# encoding=utf-8
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import orthogonal_

from fairseq.data.data_utils import collate_tokens_list
from fairseq.dep import load_dependency_head_tree, DepHeadTree
from fairseq.models import register_model_architecture, register_model
from fairseq.models.nat import CMLMNATransformerModel, cmlm_base_architecture
from fairseq.util2 import get_base_mask, set_all_token, set_diff_tokens, mean_ds


class Tarjan:
    """
    adopted from : https://github.com/jcyk/Dynet-Biaffine-dependency-parser/blob/master/lib/tarjan.py
    """

    def __init__(self, prediction, tokens):
        """
        :param prediction: A predicted dependency tree where prediction[dep_idx] = head_idx
        :param tokens: The tokens we care about (i.e. exclude GO, EOS, PAD)
        """
        self._edges = defaultdict(set)
        self._vertices = set((0,))
        for dep, head in enumerate(prediction[tokens]):
            self._vertices.add(dep + 1)
            self._edges[head].add(dep + 1)
        self._indices = {}
        self._lowlinks = {}
        self._onstack = defaultdict(lambda: False)
        self._SCCs = []

        index = 0
        stack = []
        for v in self.vertices:
            if v not in self.indices:
                self.strongconnect(v, index, stack)

    def strongconnect(self, v, index, stack):
        # Reference : https://comzyh.com/blog/archives/517/
        self._indices[v] = index
        self._lowlinks[v] = index
        index += 1
        stack.append(v)
        self._onstack[v] = True
        for w in self.edges[v]:
            if w not in self.indices:
                self.strongconnect(w, index, stack)
                self._lowlinks[v] = min(self._lowlinks[v], self._lowlinks[w])
            elif self._onstack[w]:
                self._lowlinks[v] = min(self._lowlinks[v], self._indices[w])

        if self._lowlinks[v] == self._indices[v]:
            self._SCCs.append(set())
            while stack[-1] != v:
                w = stack.pop()
                self._onstack[w] = False
                self._SCCs[-1].add(w)
            w = stack.pop()
            self._onstack[w] = False
            self._SCCs[-1].add(w)
        return

    @property
    def edges(self):
        return self._edges

    @property
    def vertices(self):
        return self._vertices

    @property
    def indices(self):
        return self._indices

    @property
    def SCCs(self):
        return self._SCCs


class BiaffineAttentionDependency(nn.Module):
    def __init__(self, args, input_dim, head_tree=None, no_mlp=False):

        super().__init__()
        self.args = args
        self.no_mlp = no_mlp

        if not self.no_mlp:
            mlp_input_dim = input_dim
            self.mlp_dim = 500
            self.arc_head_mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, self.mlp_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.33))

            self.arc_dep_mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, self.mlp_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.33))
        else:
            self.mlp_dim = input_dim

        self.W_arc = nn.Parameter(orthogonal_(
            torch.empty(self.mlp_dim + 1, self.mlp_dim).cuda()
        ), requires_grad=True)

        if head_tree is None:
            self.head_tree = DepHeadTree(valid_subset=self.args.valid_subset)
        else:
            self.head_tree = head_tree

        self.dropout = nn.Dropout(0.33)

    def get_reference(self, sample_ids):

        # -1表示不用的节点，0表示根节点。
        tree = self.head_tree.get_sentences(sample_ids, training=self.training)
        tree = [[-1] + t + [-1] for t in tree]  # 特殊处理bos和eos
        head_label = collate_tokens_list(tree, pad_idx=-1)
        head_label = torch.from_numpy(head_label).long().cuda()

        return head_label

    def compute_loss(self, outputs, targets):
        logits = F.log_softmax(outputs, dim=-1)
        losses = F.nll_loss(logits, targets.to(logits.device), reduction='none')
        loss = mean_ds(losses)

        return loss

    def forward_classifier(self, hidden_state, return_hidden=False):
        if not self.no_mlp:
            h_arc_dep = self.arc_dep_mlp(hidden_state)  # batch * max_trg * mlp_dim
            h_arc_head = self.arc_head_mlp(hidden_state)  # batch * max_trg * mlp_dim
        else:
            h_arc_dep, h_arc_head = hidden_state, hidden_state

        batch_size, max_trg_len, decoder_dim = h_arc_head.size()

        arc_dep = torch.cat((h_arc_dep, torch.ones(batch_size, max_trg_len, 1).cuda()),
                            dim=-1)  # batch * trg_len * (dim+1)

        head_dep_result = arc_dep.matmul(self.W_arc).matmul(h_arc_head.transpose(1, 2))  # batch * trg_len * trg_len

        if not return_hidden:
            return head_dep_result
        else:
            return head_dep_result, hidden_state

    # def forward(self, hidden_state, reference):
    #
    #     reference_mask = reference != self.pad
    #     length = reference_mask.long().sum(-1)
    #
    #     if self.contain_lstm:
    #         unpaded_reference = nn.utils.rnn.pack_padded_sequence(hidden_state, lengths=length,
    #                                                               batch_first=True, enforce_sorted=False)
    #         decoder_input, _ = self.LSTM(unpaded_reference)
    #         decoder_input, _ = nn.utils.rnn.pad_packed_sequence(decoder_input, batch_first=True, padding_value=self.pad)
    #         size = decoder_input.size()
    #         decoder_input = self.batch_norm(decoder_input.contiguous().view(-1, size[2])).contiguous().view(size)
    #     else:
    #         decoder_input = hidden_state
    #
    #     h_arc_dep = self.arc_dep_mlp(decoder_input)  # batch * max_trg * mlp_dim
    #     h_arc_head = self.arc_head_mlp(decoder_input)  # batch * max_trg * mlp_dim
    #
    #     batch_size, max_trg_len, decoder_dim = h_arc_head.size()
    #
    #     arc_dep = torch.cat((h_arc_dep, torch.ones(batch_size, max_trg_len, 1).cuda()),
    #                         dim=-1)  # batch * trg_len * (dim+1)
    #
    #     head_dep_result = arc_dep.matmul(self.W_arc).matmul(h_arc_head.transpose(1, 2))  # batch * trg_len * trg_len
    #
    #     return head_dep_result

    def compute_accuracy(self, head_dep_result, head_mask, head_dep_reference):
        pred_head = head_dep_result.argmax(-1)

        output = pred_head[head_mask]
        ref = head_dep_reference[head_mask]

        all_predict_head, = output.size()
        correct_predict_head = (ref == output).sum().item()

        return all_predict_head, correct_predict_head

    # def compute_accuracy_nosum(self, head_dep_result, head_mask, head_dep_reference):
    #     pred_head = head_dep_result.argmax(-1)
    #
    #     diff = (pred_head != head_dep_reference) & head_mask
    #     s = diff.sum(-1)
    #
    #     return s

    def parser_tree(self, parse_probs, length, tokens_to_keep, ensure_tree=True):
        """
        adopted from : https://github.com/jcyk/Dynet-Biaffine-dependency-parser/blob/master/lib/utils.py
        """
        I = np.eye(len(tokens_to_keep))
        parse_probs = parse_probs * tokens_to_keep * (1 - I)  # 去除mask和自身节点
        parse_preds = np.argmax(parse_probs, axis=1)
        tokens = np.arange(1, length)
        roots = np.where(parse_preds[tokens] == 0)[0] + 1
        # ensure at least one root
        if len(roots) < 1:
            root_probs = parse_probs[tokens, 0]
            old_head_probs = parse_probs[tokens, parse_preds[tokens]]
            new_root_probs = root_probs / old_head_probs
            new_root = tokens[np.argmax(new_root_probs)]
            parse_preds[new_root] = 0
        elif len(roots) > 1:
            root_probs = parse_probs[roots, 0]
            parse_probs[roots, 0] = 0
            new_heads = np.argmax(parse_probs[roots][:, tokens], axis=1) + 1
            new_head_probs = parse_probs[roots, new_heads] / root_probs
            new_root = roots[np.argmin(new_head_probs)]
            parse_preds[roots] = new_heads
            parse_preds[new_root] = 0
        tarjan = Tarjan(parse_preds, tokens)
        cycles = tarjan.SCCs
        for SCC in tarjan.SCCs:
            if len(SCC) > 1:
                dependents = set()
                to_visit = set(SCC)
                # In my tuition, it is all the node related to this SCC
                while len(to_visit) > 0:
                    node = to_visit.pop()
                    if not node in dependents:
                        dependents.add(node)
                        to_visit.update(tarjan.edges[node])
                # The indices of the nodes that participate in the cycle
                cycle = np.array(list(SCC))
                # The probabilities of the current heads
                old_heads = parse_preds[cycle]
                old_head_probs = parse_probs[cycle, old_heads]
                # Set the probability of depending on a non-head to zero
                non_heads = np.array(list(dependents))
                parse_probs[np.repeat(cycle, len(non_heads)), np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
                # Get new potential heads and their probabilities
                new_heads = np.argmax(parse_probs[cycle][:, tokens], axis=1) + 1
                new_head_probs = parse_probs[cycle, new_heads] / old_head_probs
                # Select the most probable change
                change = np.argmax(new_head_probs)
                changed_cycle = cycle[change]
                old_head = old_heads[change]
                new_head = new_heads[change]
                # Make the change
                parse_preds[changed_cycle] = new_head
                tarjan.edges[new_head].add(changed_cycle)
                tarjan.edges[old_head].remove(changed_cycle)
            return parse_preds


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
