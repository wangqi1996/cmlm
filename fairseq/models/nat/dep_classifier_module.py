# encoding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.dep import RelativeDepMat
from fairseq.models import BaseFairseqModel
from fairseq.models.lstm import LSTM
from fairseq.models.nat import BiaffineAttentionDependency, set_diff_tokens, set_all_token, get_base_mask, \
    BlockedDecoderLayer, build_relative_embeddings
from fairseq.util2 import mean_ds, set_key_value


def build_dep_classifier(model_name, **kwargs):
    if model_name == "none":
        return None

    if model_name == "head":
        return DepHeadClassifier(**kwargs)

    if model_name == "relative":
        return DepCoarseClassifier(**kwargs)


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


class LSTMEncoder(nn.Module):
    """ 对输入进行编码 使用LSTM """

    def __init__(self, args, num_layers):
        super().__init__()

        self.args = args
        self.lstm = LSTM(
            input_size=args.decoder_embed_dim,
            hidden_size=400,
            num_layers=1,
            dropout=0.,
            bidirectional=True,
            batch_first=False
        )

        args.decoder_embed_dim = 400 * 2
        args.decoder_ffn_embed_dim = 400 * 4
        rel_keys = build_relative_embeddings(args)
        rel_vals = build_relative_embeddings(args)
        self.attention = BlockedDecoderLayer(args, no_encoder_attn=False, relative_keys=rel_keys,
                                             relative_vals=rel_vals)

    def forward(self, encoder_out, hidden_state, decoder_padding_mask):
        length = (~decoder_padding_mask).long().sum(-1)
        unpaded_reference = nn.utils.rnn.pack_padded_sequence(hidden_state, lengths=length,
                                                              batch_first=False, enforce_sorted=False)
        hidden_state, _ = self.lstm(unpaded_reference)
        hidden_state, _ = nn.utils.rnn.pad_packed_sequence(hidden_state, batch_first=False, padding_value=1)

        hidden_state, layer_attn, _ = self.attention(
            hidden_state,
            encoder_out.encoder_out if encoder_out is not None else None,
            encoder_out.encoder_padding_mask if encoder_out is not None else None,
            self_attn_mask=None,
            self_attn_padding_mask=decoder_padding_mask,
        )
        return hidden_state.transpose(0, 1)


class DepClassifier(BaseFairseqModel):

    def __init__(self, args, relative_dep_mat=None, use_two_class=False, **kwargs):
        super().__init__()
        self.args = args
        self.dep_loss_factor = getattr(self.args, "dep_loss_factor", 1.0)
        if relative_dep_mat is None:
            self.relative_dep_mat = RelativeDepMat(valid_subset=self.args.valid_subset, use_two_class=use_two_class)
        else:
            self.relative_dep_mat = relative_dep_mat

        self.encoder_dep_input = getattr(self.args, "encoder_dep_input", "transformer")
        self.encoder = None
        if self.encoder_dep_input != "none":
            print("使用decoder blocks来编码分类器的input: ", self.encoder_dep_input)
            if self.encoder_dep_input == "lstm":
                self.encoder = LSTMEncoder(args, num_layers=1)
            else:
                self.encoder = DepEncoder(args, num_layers=2)

        self.dropout = nn.Dropout(0.33)

        self._add_position = getattr(self.args, "add_position_method", "none")
        self._add_position = "cat"
        if self._add_position != "none":
            print("添加position信息: ", self._add_position)

        self.relax_dep_mat = getattr(self.args, "relax_dep_mat", False)
        if self.relax_dep_mat:
            print("generate时, relax模型")

        if self._add_position == "cat":
            self.mlp_input_dim = args.decoder_embed_dim + 278
        else:
            self.mlp_input_dim = args.decoder_embed_dim * 1

    def inference(self, hidden_state, position_embedding, reference, perturb=0.0, **kwargs):

        batch_size, _ = reference.size()
        assert batch_size == 1, u"infernece仅仅支持batch=1"
        score = self.forward_classifier(hidden_state, position_embedding, decoder_padding_mask=reference.eq(1),
                                        encoder_out=kwargs.get("encoder_out", None))
        _label = self.get_label(score, reference=reference)
        _label[0][0] = 0
        _label[0][-1] = 0
        _label[0][:, 0] = 0
        _label[0][:, -1] = 0

        if self.relax_dep_mat:
            mask = (_label == 1) | (_label.transpose(1, 2) == 1)
            _label.masked_fill_(mask, 1)

        if kwargs.get("eval_accuracy", False) or perturb != 0.0:
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

            sample = kwargs.get("sample", None)
            sample_ids = sample['id'].cpu().tolist()
            reference = sample["target"]
            dependency_mat = self.relative_dep_mat.get_dependency_mat(sample_ids, reference, training=self.training,
                                                                      contain_eos=True)

            predict = _label
            target = dependency_mat
            b, l, _ = predict.size()
            all = b * l * l
            correct = (target == predict).sum().item()
            set_diff_tokens(correct)
            set_all_token(all)

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

            def _perturb(label):
                predict_i, target_i, correct_i = _count(label)
                co_score = score + (target == label).long()
                co_score = co_score.view(-1)
                _, target_rank = co_score.sort(descending=True)
                mask = target_rank.new_zeros(target_rank.size()).bool().fill_(False)
                mask_length = target_i - correct_i
                mask[target_rank[:mask_length]] = True
                mask = mask.view(l, l)
                return mask

            name = ["pad", "positive", "negative", "same"]
            if perturb != 0.0:
                _label = target.clone()
                score = _label.clone().float().uniform_()
                _label.masked_fill_(_perturb(1), 2)
                _label.masked_fill_(_perturb(2), 1)
                predict = _label

            for i in [1, 2]:  # 0 pad 1 相关 2 不相关 3 相似
                predict_i, target_i, correct_i = _count(i)
                set_key_value("predict_" + name[i], predict_i)
                set_key_value("target_" + name[i], target_i)
                set_key_value("correct_" + name[i], correct_i)

        return _label

    @staticmethod
    def add_args(parser):
        parser.add_argument('--dep-loss-factor', default=1.0, type=float)
        parser.add_argument('--encoder-dep-input', type=str, default="transformer")  # lstm

        """ 二分类器相关"""
        parser.add_argument('--positive-class-factor', type=float, default=1.0)
        parser.add_argument('--classifier-mutual-method', type=str, default="none", choices=['none', 'bias', 'logit'])
        parser.add_argument('--dep-focal-loss', action="store_true")
        parser.add_argument('--relax-dep-mat', action="store_true")
        parser.add_argument('--gumbel-softmax-mat', action="store_true")
        parser.add_argument('--threshold', type=float, default=0.556)

        """ 依存树相关"""
        parser.add_argument('--add-position-method', type=str, default="none")  # 如何使用position信息 add、none、cat
        parser.add_argument('--use-MST', action="store_true")

    def inference_accuracy(self, hidden_state, position_embedding, compute_loss, target_token, sample, eval_accuracy,
                           result_mat, encoder_out, **kwargs):
        loss, all, correct = 0, 0, 0
        score, predict, head_ref, train_mat = self.predict(hidden_state, position_embedding, sample, target_token,
                                                           result_mat=result_mat, encoder_out=encoder_out)
        if compute_loss:
            loss = self.compute_loss(predict, head_ref)
        if eval_accuracy:
            all, correct = self.compute_accuracy(predict, head_ref, score, target_token)

        return loss, all, correct, train_mat

    def add_position(self, hidden_state, position):
        # 编码之后调用
        if self._add_position == "none":
            return hidden_state

        if self._add_position == "cat":
            return torch.cat((hidden_state, position), dim=-1)

        if self._add_position == "add":
            return hidden_state + position


class DepHeadClassifier(DepClassifier):
    def __init__(self, args, relative_dep_mat=None, use_two_class=False, **kwargs):
        super().__init__(args, relative_dep_mat, use_two_class, **kwargs)

        self.use_MST = getattr(self.args, "use_MST", False)
        if self.use_MST:
            print("使用最小生成树，目前仅仅在生成矩阵（计算准确率+generate时）使用")

        self.biaffine_attention = BiaffineAttentionDependency(args, input_dim=self.mlp_input_dim)

    def compute_loss(self, outputs, targets):
        # 计算损失的肯定是依存树预测损失
        return self.biaffine_attention.compute_loss(outputs, targets) * self.dep_loss_factor

    def forward_classifier(self, hidden_state, position_embedding, decoder_padding_mask, encoder_out):
        hidden_state = self.dropout(hidden_state)
        if self.encoder is not None:
            hidden_state = self.encoder(encoder_out, hidden_state, decoder_padding_mask)
        else:
            hidden_state = hidden_state.transpose(0, 1)

        hidden_state = self.add_position(hidden_state, position_embedding)
        hidden_state = self.dropout(hidden_state)
        output = self.biaffine_attention.forward_classifier(hidden_state)  # [b, tgt_len, tgt_len]
        return output

    def get_reference(self, sample_ids):
        head_ref = self.biaffine_attention.get_reference(sample_ids)
        return head_ref

    def get_head(self, score):
        return score.argmax(-1)

    def predict(self, hidden_state, position_embedding, sample, reference, result_mat=False, encoder_out=None,
                **kwargs):

        score = self.forward_classifier(hidden_state, position_embedding, decoder_padding_mask=(reference.eq(1)),
                                        encoder_out=encoder_out)

        sample_ids = sample['id'].cpu().tolist()
        head_ref = self.get_reference(sample_ids)

        head_mask = head_ref != -1

        predict = score[head_mask]
        head_ref = head_ref[head_mask]

        mat = None
        if result_mat:
            mat = self.get_label(score, reference)

        return score, predict, head_ref, mat

    def compute_accuracy(self, predict_head, head_ref, score, reference):
        if self.use_MST:
            predict = []
            batch_size, tgt_len, tgt_len = score.size()
            base_mask = get_base_mask(reference)
            base_mask[:, 0] = True
            score = F.softmax(score, dim=-1)
            for b in range(batch_size):
                head = self.biaffine_attention.parser_tree(score[b].squeeze(0).cpu().numpy(), base_mask[b].sum().item(),
                                                           base_mask[b].int().cpu().numpy())
                predict.append(head)
            predict = head_ref.new_tensor(predict)
            base_mask[:, 0] = False
            predict = predict[base_mask]
        else:
            predict = self.get_head(predict_head)
        target = head_ref
        all = len(predict)
        correct = (target == predict).sum().item()
        return all, correct

    def get_label(self, score, reference):
        # score: [b, l, l]
        dep_mat = score.new_zeros(score.size()).long()

        if not self.use_MST:
            all_head = self.get_head(score)

        batch_size, tgt_len = reference.shape
        score = F.softmax(score, dim=-1)
        for i in range(batch_size):
            ref_len = (reference[i] != 1).sum().item()
            base_mask = get_base_mask(reference[i])
            base_mask[0] = True
            if self.use_MST:
                head = self.biaffine_attention.parser_tree(score.squeeze(0).cpu().numpy(), base_mask.sum().item(),
                                                           base_mask.int().cpu().numpy())
            else:
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


class DepCoarseClassifier(DepClassifier):
    def __init__(self, args, relative_dep_mat=None, use_two_class=False):
        super().__init__(args, relative_dep_mat, use_two_class)

        self.dep_focal_loss = getattr(self.args, "dep_focal_loss", False)
        if self.dep_focal_loss:
            print("use focal loss: ", self.dep_focal_loss)

        self.gumbel_softmax_mat = getattr(self.args, "gumbel_softmax_mat", False)
        if self.gumbel_softmax_mat:
            print("使用gumbel-softmax获得依存矩阵")

        self.classifier_mutual_method = getattr(self.args, "classifier_mutual_method", "none")
        if self.classifier_mutual_method != "none":
            print("是否使用互信息来解决类别不均衡问题: ", self.classifier_mutual_method)

            if self.classifier_mutual_method in ["logit", "bias"]:
                self.log_prior = self.mutual_logits()

        self.threshold = getattr(self.args, "threshold", 0.556)
        print("threshold: ", self.threshold)

        self.mlp_dim = 400
        # self.mlp_dim = self.mlp_input_dim
        self.class_num = 3
        if use_two_class and not self.gumbel_softmax_mat:
            self.class_num = 1
            self.loss = nn.BCEWithLogitsLoss(reduction='none')

        if self.gumbel_softmax_mat:
            self.class_num = 2

        self.mlp_output = nn.Sequential(
            nn.Linear(self.mlp_input_dim * 2, self.mlp_dim),
            nn.Tanh(),
            nn.Linear(self.mlp_dim, self.class_num)
        )
        self.positive_class_factor = getattr(self.args, "positive_class_factor", 1.0)

    def mutual_logits(self):
        prior = [0.143, 0.857]
        log_prior = np.log(prior)
        tau = 1.0
        delta = (log_prior[1] - log_prior[0]) * tau
        return delta

    def apply_init_bias(self):
        self.mlp_output[-1].bias.data = torch.FloatTensor([self.log_prior])

    def compute_loss(self, outputs, targets):

        targets = targets - 1  # 去除pad类别
        # 多分类
        if self.class_num != 1:
            logits = F.log_softmax(outputs, dim=-1)
            losses = F.nll_loss(logits, targets.to(logits.device), reduction='none')
        else:
            losses = self.loss(outputs.squeeze(-1), targets.float())

        if self.positive_class_factor != 1.0:
            weights = losses.new_ones(size=losses.size()).float()
            weights.masked_fill_(targets == 0, self.positive_class_factor)
            losses = losses * weights

        if self.dep_focal_loss:
            p_logits = F.sigmoid(outputs)
            neg_logits = 1 - p_logits
            p_logits = p_logits.masked_fill(targets == 1, 1).detach()
            p_logits = p_logits * p_logits
            neg_logits = neg_logits.masked_fill(targets == 0, 1).detach()
            neg_logits = neg_logits * neg_logits
            losses = losses * p_logits * neg_logits

        loss = mean_ds(losses)

        loss = loss * self.dep_loss_factor
        return loss

    def forward_classifier(self, hidden_state, position_embedding, decoder_padding_mask, encoder_out):

        hidden_state = self.dropout(hidden_state)
        if self.encoder is not None:
            hidden_state = self.encoder(encoder_out, hidden_state, decoder_padding_mask)
        else:
            hidden_state = hidden_state.transpose(0, 1)

        hidden_state = self.add_position(hidden_state, position_embedding)
        hidden_state = self.dropout(hidden_state)
        batch_size, seq_len, hidden_dim = hidden_state.size()
        a = hidden_state.unsqueeze(1).repeat(1, seq_len, 1, 1)  # [b, l, l, d]
        b = hidden_state.unsqueeze(2).repeat(1, 1, seq_len, 1)  # [b, l, l, d]
        feature = torch.cat((a, b), dim=-1)

        score = self.mlp_output(feature)  # [b, l, l, 3]

        if self.classifier_mutual_method == "logit":
            assert self.class_num == 1, "只支持二分类打分"
            score = score + self.log_prior
        return score.squeeze(-1)

    def sample_mat(self, score):
        mat = F.gumbel_softmax(score, hard=True, tau=1)
        if len(mat.size()) == 4:
            result = mat[:, :, :, 1] + 1  # 相关
        if len(mat.size()) == 3:
            result = mat[:, :, 1] + 1
        if len(mat.size()) == 2:
            result = mat[:, 1] + 1
        return result.long()

    def get_label(self, score, reference_mat=None, **kwargs):
        """预测->依存矩阵"""
        if self.gumbel_softmax_mat:
            assert self.class_num == 2, "gumbel softmax只支持分类呀~"
            mat = self.sample_mat(score)
            return mat
        if self.class_num == 1:
            mat = (F.sigmoid(score) > self.threshold).long() + 1
        else:
            _score, _label = F.log_softmax(score, dim=-1).max(-1)
            mat = _label + 1

        return mat

    def predict(self, hidden_state, position_embedding, sample, reference, result_mat=False, encoder_out=None):

        score = self.forward_classifier(hidden_state, position_embedding, decoder_padding_mask=reference.eq(1),
                                        encoder_out=encoder_out)

        sample_ids = sample['id'].cpu().tolist()
        dependency_mat = self.relative_dep_mat.get_dependency_mat(sample_ids, reference, training=self.training)

        reference_mask = dependency_mat != 0
        mat = None
        if result_mat:
            _label = self.get_label(score)
            mat = _label.masked_fill(~reference_mask, 0)

        predict = score[reference_mask]
        dependency_mat = dependency_mat[reference_mask]

        return score, predict, dependency_mat, mat

    def compute_accuracy(self, predict, target, score, target_token, **kwargs):
        _label = self.get_label(predict)

        all = len(_label)
        correct = (target == _label).sum().item()
        return all, correct
