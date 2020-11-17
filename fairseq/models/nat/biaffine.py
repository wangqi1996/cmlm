# encoding=utf-8
import torch
from torch import nn
from torch.nn.init import orthogonal_

from fairseq.data.data_utils import collate_tokens_list
from fairseq.dep import load_dependency_head_tree
from fairseq.models import register_model_architecture, register_model
from fairseq.models.nat import CMLMNATransformerModel, cmlm_base_architecture
from fairseq.util2 import get_base_mask, set_all_token, set_diff_tokens


class BiaffineAttentionDependency(nn.Module):
    def __init__(self, args, input_dim, contain_lstm=False, padding_idx=0):

        super().__init__()

        self.mlp_dim = 500
        self.args = args
        self.pad = padding_idx
        mlp_input_dim = input_dim
        if contain_lstm:
            mlp_input_dim = 300

        self.arc_head_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, self.mlp_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.33))

        self.arc_dep_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, self.mlp_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.33))

        self.W_arc = nn.Parameter(orthogonal_(
            torch.empty(self.mlp_dim + 1, self.mlp_dim).cuda()
        ), requires_grad=True)

        self.input_mapping_pai = 0.3  # 暂时不用这个参数

        # 获取依存树用作计算损失
        self.train_dependency_tree_head = load_dependency_head_tree(
            dependency_tree_path="/home/data_ti5_c/wangdq/data/distill/iwslt14_de_en/data-bin/dependency_head.train.log",
            add_one=True)
        self.valid_dependency_tree_head = load_dependency_head_tree(
            dependency_tree_path="/home/data_ti5_c/wangdq/data/distill/iwslt14_de_en/data-bin/dependency_head.valid.log",
            add_one=True)

        self.contain_lstm = contain_lstm

        if contain_lstm:
            self.LSTM = nn.GRU(input_size=input_dim, hidden_size=300, num_layers=1, bias=True,
                               batch_first=True)

    def get_dep_reference(self, sample_ids):
        if self.training:
            tree = self.train_dependency_tree_head
        else:
            tree = self.valid_dependency_tree_head

        tree = [[0] + tree[id] + [0] for id in sample_ids]  # 特殊处理bos和eos
        head_label = collate_tokens_list(tree, pad_idx=self.pad)
        head_label = torch.from_numpy(head_label).long().cuda()

        return head_label

    def forward(self, hidden_state, reference):

        reference_mask = reference != self.pad
        length = reference_mask.long().sum(-1)

        if self.contain_lstm:
            unpaded_reference = nn.utils.rnn.pack_padded_sequence(hidden_state, lengths=length,
                                                                  batch_first=True, enforce_sorted=False)
            decoder_input, _ = self.LSTM(unpaded_reference)
            decoder_input, _ = nn.utils.rnn.pad_packed_sequence(decoder_input, batch_first=True, padding_value=self.pad)
        else:
            decoder_input = hidden_state

        h_arc_dep = self.arc_dep_mlp(decoder_input)  # batch * max_trg * mlp_dim
        h_arc_head = self.arc_head_mlp(decoder_input)  # batch * max_trg * mlp_dim

        batch_size, max_trg_len, decoder_dim = h_arc_head.size()

        arc_dep = torch.cat((h_arc_dep, torch.ones(batch_size, max_trg_len, 1).cuda()),
                            dim=-1)  # batch * trg_len * (dim+1)

        head_dep_result = arc_dep.matmul(self.W_arc).matmul(h_arc_head.transpose(1, 2))  # batch * trg_len * trg_len

        return head_dep_result

    def compute_accuracy(self, head_dep_result, head_mask, head_dep_reference):
        pred_head = head_dep_result.argmax(-1)

        output = pred_head[head_mask]
        ref = head_dep_reference[head_mask]

        all_predict_head, = output.size()
        correct_predict_head = (ref == output).sum().item()

        return all_predict_head, correct_predict_head

    def compute_accuracy_nosum(self, head_dep_result, head_mask, head_dep_reference):
        pred_head = head_dep_result.argmax(-1)

        diff = (pred_head != head_dep_reference) & head_mask
        s = diff.sum(-1)

        return s


@register_model("biaffine")
class BiaffineAttention(CMLMNATransformerModel):
    """
    https://arxiv.org/abs/1611.01734
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        self.mlp_dim = 500
        self.pad = self.tgt_dict.pad
        self.biaffine_input = args.biaffine_input
        print("biaffine_input: ", self.biaffine_input)

        mlp_input_dim = self.args.encoder_embed_dim
        if self.biaffine_input in ["reference_lstm", "hidden_lstm"]:
            mlp_input_dim = 300

        self.arc_head_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, self.mlp_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.33))

        self.arc_dep_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, self.mlp_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.33))

        self.W_arc = nn.Parameter(orthogonal_(
            torch.empty(self.mlp_dim + 1, self.mlp_dim).cuda()
        ), requires_grad=True)

        self.input_mapping_pai = 0.3  # 暂时不用这个参数
        # self.optimizer = torch.optim.Adam()

        # 获取依存树用作计算损失
        self.train_dependency_tree_head = load_dependency_head_tree(
            dependency_tree_path="/home/data_ti5_c/wangdq/data/distill/iwslt14_de_en/data-bin2/dependency_head.train.log",
            add_one=True)
        self.valid_dependency_tree_head = load_dependency_head_tree(
            dependency_tree_path="/home/data_ti5_c/wangdq/data/distill/iwslt14_de_en/data-bin2/dependency_head.valid.log",
            add_one=True)

        if args.froze_nmt_model:
            self.froze_nmt_model()

        if self.biaffine_input in ["reference_lstm", "hidden_lstm"]:
            self.LSTM = nn.GRU(input_size=self.args.encoder_embed_dim, hidden_size=300, num_layers=1, bias=True,
                               batch_first=True)

        self.load_enc_dec = args.load_enc_dec
        if self.load_enc_dec:
            print("only load encoder and decoder")

        self.no_word_loss = args.no_word_loss
        if self.no_word_loss:
            print("not compute the word loss!!")

    def load_state_dict(self, state_dict, strict=False, args=None):
        # if self.load_enc_dec:
        #     print("only load encoder and decoder")
        #     new_state_dict = {}
        #     for key, value in state_dict.items():
        #         if key.startswith("encoder.") or key.startswith("decoder."):
        #             new_state_dict[key] = value
        #
        #     state_dict = new_state_dict

        super().load_state_dict(state_dict, strict)

    def get_dep_reference(self, sample_ids):
        if self.training:
            tree = self.train_dependency_tree_head
        else:
            tree = self.valid_dependency_tree_head

        tree = [[0] + tree[id] + [0] for id in sample_ids]  # eos和bos都不计算loss,使用mask进行控制
        head_label = collate_tokens_list(tree, pad_idx=self.pad)  # 因为计算loss的方式，pad=几 也无所谓
        head_label = torch.from_numpy(head_label).long().cuda()  # head label应该没有梯度

        return head_label

    def get_dep_loss(self, head_dep_result, reference, dep_head_reference):
        # 仅仅计算部分位置的loss
        return {
            "dep_head": {
                "out": head_dep_result,
                "tgt": dep_head_reference,
                "mask": get_base_mask(reference),  # 所有的token都要学习根节点
                "ls": 0.1
            }
        }

    # def get_decoder_input(self, loss):
    #     # 根据encoder out构造decoder input，暂时不用
    #     encoder_out = loss['train_need']['encoder_out'].encoder_out  # batch x max_src x dim
    #     source_masks = loss['train_need']['encoder_out'].encoder_padding_mask
    #     reference: torch.Tensor = loss['word_ins']['tgt']
    #     decoder_masks = reference == self.pad
    #
    #     INF = 1e10
    #
    #     max_src_len = source_masks.size(1)
    #     max_trg_len = decoder_masks.size(1)
    #     src_lens = (~source_masks).sum(-1).float()  # batchsize
    #     trg_lens = (~decoder_masks).sum(-1).float()  # batchsize
    #     steps = src_lens / trg_lens  # batchsize
    #     index_t = torch.arange(0, max_trg_len).float().cuda()  # max_trg_len
    #
    #     index_t = steps[:, None] @ index_t[None, :]  # batch x max_trg_len
    #     index_s = torch.arange(0, max_src_len).float().cuda()  # max_src_len
    #
    #     indexxx_ = (index_s[None, None, :] - index_t[:, :, None]) ** 2  # batch x max_trg x max_src
    #     weight = F.softmax(Variable(
    #         -indexxx_.float() / self.input_mapping_pai - INF * (source_masks[:, None, :].float())),
    #         dim=-1)  # batch x max_trg x max_src 同时进行了mask操作
    #
    #     decoder_input = weight.matmul(encoder_out.transpose(0, 1))  # batch * max_trg * dim
    #
    #     return decoder_input

    def forward_biaffine(self, loss, dep_head_reference):
        if self.biaffine_input == -1:
            return {}

        decoder_input = None
        if self.biaffine_input == "0":
            decoder_input = self.get_decoder_input(loss=loss)  # batch_size * max_trg * dim
        elif self.biaffine_input == 'hidden':
            hidden_state = loss['train_need']['hidden_state']
            decoder_input = hidden_state  # check hidden是否有梯度
        elif self.biaffine_input == "hidden_lstm":
            # 依赖太长，对NMT模型的更新不好
            hidden_state = loss['train_need']['hidden_state']
            reference = loss['word_ins']['tgt']  # [batch, seq_len]
            reference_mask = reference != self.pad
            reference_length = reference_mask.long().sum(-1)

            unpaded_reference = nn.utils.rnn.pack_padded_sequence(hidden_state, lengths=reference_length,
                                                                  batch_first=True, enforce_sorted=False)
            decoder_input, _ = self.LSTM(unpaded_reference)
            decoder_input, _ = nn.utils.rnn.pad_packed_sequence(decoder_input, batch_first=True, padding_value=self.pad)

        elif self.biaffine_input == "reference":
            reference = loss['word_ins']['tgt']  # [batch, seq_len]
            reference_embed, _ = self.decoder.forward_embedding(reference)
            decoder_input = reference_embed

        elif self.biaffine_input == 'reference_lstm':
            reference = loss['word_ins']['tgt']  # [batch, seq_len]
            reference_mask = reference != self.pad
            reference_length = reference_mask.long().sum(-1)

            reference_embed, _ = self.decoder.forward_embedding(reference)

            unpaded_reference = nn.utils.rnn.pack_padded_sequence(reference_embed, lengths=reference_length,
                                                                  batch_first=True, enforce_sorted=False)
            decoder_input, _ = self.LSTM(unpaded_reference)
            decoder_input, _ = nn.utils.rnn.pad_packed_sequence(decoder_input, batch_first=True, padding_value=self.pad)

        h_arc_dep = self.arc_dep_mlp(decoder_input)  # batch * max_trg * mlp_dim
        h_arc_head = self.arc_head_mlp(decoder_input)  # batch * max_trg * mlp_dim

        batch_size, max_trg_len, decoder_dim = h_arc_head.size()

        arc_dep = torch.cat((h_arc_dep, torch.ones(batch_size, max_trg_len, 1).cuda()),
                            dim=-1)  # batch * trg_len * (dim+1)

        head_dep_result = arc_dep.matmul(self.W_arc).matmul(h_arc_head.transpose(1, 2))  # batch * trg_len * trg_len

        # training
        reference: torch.Tensor = loss['word_ins']['tgt']
        dep_loss = self.get_dep_loss(head_dep_result, reference, dep_head_reference)

        return dep_loss

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        loss = super().forward(src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs)

        sample_ids = loss['train_need']['sample_ids']
        dep_head_reference = self.get_dep_reference(sample_ids)

        dep_loss = self.forward_biaffine(loss, dep_head_reference)

        if self.no_word_loss:
            train_need = loss.pop("train_need")
            del loss
            loss = dep_loss
            loss.update({
                "train_need": train_need
            })
        else:
            loss.update(dep_loss)

        if kwargs.get("eval_accuracy", False):
            all_predict_head, correct_predict_head = self._compute_accuracy(dep_loss)
            loss.setdefault('train_need', {})
            loss['train_need'].update({
                "print": {
                    "all_predict_head": all_predict_head,
                    "correct_predict_head": correct_predict_head
                }})

        return loss

    def _compute_accuracy(self, dep_loss):
        head_dep_result = dep_loss['dep_head']['out']
        pred_head = head_dep_result.argmax(-1)

        head_mask = dep_loss['dep_head']['mask']
        output = pred_head[head_mask]
        ref = dep_loss['dep_head']['tgt'][head_mask]

        all_predict_head, = output.size()
        correct_predict_head = (ref == output).sum().item()

        return all_predict_head, correct_predict_head

    def compute_accuracy(self, hidden_state, encoder_out, references, dep_head_reference):
        loss = {
            "train_need": {
                "hidden_state": hidden_state,
                "encoder_out": encoder_out,
            },
            "word_ins": {
                "tgt": references
            },

        }
        dep_loss = self.forward_biaffine(loss, dep_head_reference)
        all_predict_head, correct_predict_head = self._compute_accuracy(dep_loss)
        return all_predict_head, correct_predict_head

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        decoder_out, hidden_state = super().forward_decoder(decoder_out, encoder_out, decoding_format=decoding_format,
                                                            return_hidden_state=True, **kwargs)

        # # # # only compute at step=i

        if kwargs.get('compute_dep_accuracy', False):

            step = decoder_out.step

            max_step = decoder_out.max_step
            if step != max_step - 1:
                return decoder_out

            references = kwargs['references']
            head_label = kwargs['biaffine_tree']

            all_predict_head, correct_predict_head = self.compute_accuracy(hidden_state, encoder_out,
                                                                           references, head_label)
            set_diff_tokens(correct_predict_head)
            set_all_token(all_predict_head)

        return decoder_out


@register_model_architecture("biaffine", "biaffine")
def biaffine_architecture(args):
    cmlm_base_architecture(args)
