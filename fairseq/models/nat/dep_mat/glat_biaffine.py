# encoding=utf-8
import torch
import torch.nn as nn

from fairseq.dep import DepHeadTree
from fairseq.models import BaseFairseqModel
from fairseq.models.nat import BiaffineAttentionDependency, BlockedDecoderLayer, build_relative_embeddings
from fairseq.util2 import set_key_value, new_arange


class DepEncoder(nn.Module):
    """ 对输入进行编码 使用自注意力 """

    def __init__(self, args, num_layers):
        super().__init__()

        rel_keys = build_relative_embeddings(args)
        rel_vals = build_relative_embeddings(args)

        self.layers = nn.ModuleList(
            [BlockedDecoderLayer(args, no_encoder_attn=False, relative_keys=rel_keys, relative_vals=rel_vals
                                 ) for _ in range(num_layers)]
        )

    def forward(self, encoder_out, hidden_state, decoder_padding_mask):
        for layer in self.layers:
            hidden_state, layer_attn, _ = layer(
                hidden_state,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
        return hidden_state.transpose(0, 1)


class DepHeadClassifier(BaseFairseqModel):

    def __init__(self, args, relative_dep_mat=None, dep_file="", **kwargs):

        super().__init__()
        self.args = args
        self.dep_loss_factor = getattr(self.args, "dep_loss_factor", 1.0)

        self.relative_dep_mat = relative_dep_mat
        self.encoder = DepEncoder(args, num_layers=2)
        self.dropout = nn.Dropout(0.33)

        self.mlp_input_dim = args.decoder_embed_dim * 2
        self.padding_idx = -1
        head_tree = DepHeadTree(valid_subset=self.args.valid_subset, dep_file=dep_file)

        self.mlp_input_dim = args.decoder_embed_dim * 2
        self.biaffine_parser = BiaffineAttentionDependency(input_dim=self.mlp_input_dim, head_tree=head_tree)

        self.glat_training = getattr(args, "glat_training", False)
        if self.glat_training:
            print("glat training!")

    def _forward_classifier(self, hidden_state, position_embedding, decoder_padding_mask, encoder_out, **kwargs):
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.encoder(encoder_out, hidden_state, decoder_padding_mask)

        hidden_state2 = self.add_position(hidden_state, position_embedding)
        hidden_state2 = self.dropout(hidden_state2)

        output = self.biaffine_parser.forward_classifier(hidden_state2)  # [b, tgt_len, tgt_len]
        return output

    def get_random_mask_output(self, mask_length=None, target_token=None, hidden_state=None, reference_embedding=None,
                               reference_mask=None):
        # mask_length大，说明模型性能不行，所以多提供reference  ==> mask_length代表的是reference的数目
        hidden_state = hidden_state.transpose(0, 1)
        reference_embedding = reference_embedding.transpose(0, 1)

        target_score = target_token.clone().float().uniform_()
        target_score.masked_fill_(~reference_mask, 2.0)

        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < mask_length[:, None].long()
        mask = target_cutoff.scatter(1, target_rank, target_cutoff)  # [b, l]
        non_mask = ~mask
        full_mask = torch.cat((non_mask.unsqueeze(-1), mask.unsqueeze(-1)), dim=-1)

        full_embedding = torch.cat((hidden_state.unsqueeze(-1), reference_embedding.unsqueeze(-1)), dim=-1)
        output_embedding = (full_embedding * full_mask.unsqueeze(-2)).sum(-1)

        return ~mask, output_embedding.transpose(0, 1)

    def get_mask_num(self, label, reference, reference_mask, update_nums):
        ratio = (1 - update_nums / 300000) * 0.2 + 0.3
        diff = ((label != reference) & reference_mask).sum(-1).detach()
        mask_length = (diff * ratio).round()
        return mask_length

    def forward_classifier(self, hidden_state=None, reference=None, target_tokens=None, ref_embedding=None,
                           **kwargs):
        if self.glat_training:
            with torch.no_grad():
                score = self._forward_classifier(hidden_state=hidden_state, **kwargs)
                score = score.detach()
                reference_mask = reference != self.padding_idx
                label = self.get_head(score)
                mask_length = self.get_mask_num(label, reference, reference_mask, kwargs.get('update_nums', 300000))
            _, hidden_state = self.get_random_mask_output(mask_length, target_tokens, hidden_state,
                                                          ref_embedding, reference_mask=reference_mask)
            score = self._forward_classifier(hidden_state=hidden_state, **kwargs)

        else:
            score = self._forward_classifier(hidden_state=hidden_state, **kwargs)
        return score

    def get_head(self, score):
        return score.argmax(-1)

    def get_label(self, score, target_tokens=None, use_MST=True, **kwargs):

        all_head = self.get_head(score)
        dep_mat = score.new_zeros(score.size()).long()
        batch_size, tgt_len = target_tokens.shape
        for i in range(batch_size):
            ref_len = (target_tokens[i] != 1).sum().item()
            head = all_head[i]
            dep_mat[i][1:ref_len - 1, 1:ref_len - 1] = 2  # 不相关

            for j in range(1, ref_len - 1):
                dep_mat[i][j][j] = 1
                h = head[j]
                if h == 0:  # 0是根节点
                    continue

                dep_mat[i][h][j] = 1
                dep_mat[i][j][h] = 1

                grand = head[h]
                if grand == 0:
                    continue
                dep_mat[i][grand][j] = 1
                dep_mat[i][j][grand] = 1

        return dep_mat

    @staticmethod
    def add_args(parser):
        parser.add_argument('--dep-loss-factor', default=1.0, type=float)
        parser.add_argument('--glat-training', action="store_true")

    def inference_accuracy(self, **kwargs):
        score, predict, head_ref = self.predict(**kwargs)
        loss = self.compute_loss(predict, head_ref)

        return loss

    def compute_loss(self, outputs, targets, teacher_score=None, other=None, **kwargs):
        # 计算损失的肯定是依存树预测损失
        b_loss = self.biaffine_parser.compute_loss(outputs, targets) * self.dep_loss_factor
        loss = {"dep_classifier": {"loss": b_loss}}
        return loss

    def get_reference(self, sample_ids, target_tokens=None):
        head_ref = self.biaffine_parser.get_reference(sample_ids)
        return head_ref

    def predict(self, sample=None, target_tokens=None, **kwargs):

        decoder_padding_mask = target_tokens.eq(1)

        sample_ids = sample['id'].cpu().tolist()
        reference = self.get_reference(sample_ids, target_tokens)
        score = self.forward_classifier(target_tokens=target_tokens, sample=sample,
                                        decoder_padding_mask=decoder_padding_mask,
                                        reference=reference,
                                        **kwargs)

        reference_mask = reference != self.padding_idx

        predict = score[reference_mask]
        dependency_mat = reference[reference_mask]

        return score, predict, dependency_mat

    def add_position(self, hidden_state, position):
        return torch.cat((hidden_state, position), dim=-1)

    def inference(self, hidden_state, position_embedding, target_tokens, perturb=0.0, **kwargs):

        batch_size, _ = target_tokens.size()
        assert batch_size == 1, u"infernece仅仅支持batch=1"
        score = self._forward_classifier(hidden_state, position_embedding,
                                         target_tokens=target_tokens,
                                         decoder_padding_mask=target_tokens.eq(1),
                                         **kwargs)
        _label = self.get_label(score, target_tokens=target_tokens)
        _label[0][0] = 0
        _label[0][-1] = 0
        _label[0][:, 0] = 0
        _label[0][:, -1] = 0

        if kwargs.get("eval_accuracy", False):
            # sample = kwargs['sample']
            # sample_ids = sample['id'].cpu().tolist()
            # head_ref = self.get_reference(sample_ids)
            #
            # head_mask = head_ref != -1
            #
            # predict = score[head_mask]
            # head_ref = head_ref[head_mask]
            # predict = self.get_head(predict)
            # target = head_ref
            # all = len(predict)
            # correct = (target == predict).sum().item()
            # set_key_value("all", all)
            # set_key_value("correct", correct)
            #
            # # 计算和位置的相关性
            # position = torch.range(1, all).cuda()
            # predict_pos = _position(predict)
            # target_pos = _position(target)
            # a = ((target == predict) & (predict == (position - 1))).sum().item()
            # b = ((target == predict) & (predict == (position + 1))).sum().item()
            # set_key_value("predict_pos", predict_pos)
            # set_key_value("target_pos", target_pos)
            # set_key_value("correct_pos", a + b)
            sample = kwargs.get("sample", None)
            sample_ids = sample['id'].cpu().tolist()
            reference = sample["target"]
            dependency_mat = self.relative_dep_mat.get_dependency_mat(sample_ids, reference, training=self.training,
                                                                      contain_eos=True)

            predict = _label
            target = dependency_mat
            b, l, _ = predict.size()
            all = b * l * l

            # correct = (target == predict).sum().item()
            # set_diff_tokens(correct)
            # set_all_token(all)

            # 阈值分析
            # score = F.sigmoid(score)
            # class_1 = score[dependency_mat == 1].cpu().tolist()
            # class_2 = score[dependency_mat == 2].cpu().tolist()
            # set_key_value_list("class_1", class_1)
            # set_key_value_list("class_2", class_2)

            def _count(i):
                predict_i = (predict == i).sum().item()  # 1 是相关
                target_i = (target == i).sum().item()
                correct_i = ((predict == i) & (target == i)).sum().item()
                return predict_i, target_i, correct_i

            # def _perturb(label):
            #     predict_i, target_i, correct_i = _count(label)
            #     co_score = score + (target == label).long()
            #     co_score = co_score.view(-1)
            #     _, target_rank = co_score.sort(descending=True)
            #     mask = target_rank.new_zeros(target_rank.size()).bool().fill_(False)
            #     mask_length = target_i - correct_i
            #     mask[target_rank[:mask_length]] = True
            #     mask = mask.view(l, l)
            #     return mask

            name = ["pad", "positive", "negative", "same"]
            # if perturb != 0.0:
            # _label = target.clone()
            # score = _label.clone().float().uniform_()
            # _label.masked_fill_(_perturb(1), 2)
            # _label.masked_fill_(_perturb(2), 1)
            # predict = _label

            for i in [1, 2]:  # 0 pad 1 相关 2 不相关 3 相似
                predict_i, target_i, correct_i = _count(i)
                set_key_value("predict_" + name[i], predict_i)
                set_key_value("target_" + name[i], target_i)
                set_key_value("correct_" + name[i], correct_i)

        return _label
