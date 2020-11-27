# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.dep import RelativeDepMat
from fairseq.models import BaseFairseqModel
from fairseq.util2 import mean_ds, set_diff_tokens, set_all_token, set_key_value


def build_dep_classifier(model_name, **kwargs):
    if model_name == "none":
        return None

    if model_name == "head":
        return DepHeadClassifier(**kwargs)

    if model_name == "relative":
        return DepCoarseClassifier(**kwargs)


class DepClassifier(BaseFairseqModel):

    def __init__(self, args, relative_dep_mat=None, use_two_class=False, **kwargs):
        super().__init__()
        self.args = args
        self.dep_loss_factor = getattr(self.args, "dep_loss_factor", 1.0)
        if relative_dep_mat is None:
            self.relative_dep_mat = RelativeDepMat(valid_subset=self.args.valid_subset, use_two_class=use_two_class)
        else:
            self.relative_dep_mat = relative_dep_mat

    def inference(self, hidden_state, position_embedding, reference, **kwargs):

        batch_size, _ = reference.size()
        assert batch_size == 1, u"infernece仅仅支持batch=1"
        score = self.forward_classifier(hidden_state, position_embedding)
        _label = self.get_label(score, reference=reference)
        _label[0][0] = 0
        _label[0][-1] = 0
        _label[0][:, 0] = 0
        _label[0][:, -1] = 0

        if kwargs.get("eval_accuracy", False):
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

            name = ["pad", "positive", "negative", "same"]
            for i in [0, 1, 2, 3]:  # 0 pad 1 相关 2 不相关 3 相似
                predict_i = (predict == i).sum().item()  # 1 是相关
                target_i = (target == i).sum().item()
                correct_i = ((predict == i) & (target == i)).sum().item()
                set_key_value("predict_" + name[i], predict_i)
                set_key_value("target_" + name[i], target_i)
                set_key_value("correct_" + name[i], correct_i)

        return _label

    @staticmethod
    def add_args(parser):
        parser.add_argument('--positive-class-factor', type=float, default=1.0)
        parser.add_argument('--dep-loss-factor', default=1.0, type=float)


class DepHeadClassifier(DepClassifier):
    """
    预测父节点 --> 预测依存矩阵
    仅仅预测父节点时，计算的准确率是父节点的
    预测依存矩阵时，计算的准确率是依存矩阵的
    """

    def __init__(self, args, relative_dep_mat=None, use_two_class=False, **kwargs):
        super().__init__(args, relative_dep_mat, use_two_class, **kwargs)

        self.biaffine_attention = BiaffineAttentionDependency(args, input_dim=self.args.decoder_embed_dim,
                                                              contain_lstm=False,
                                                              padding_idx=0)

    def compute_loss(self, outputs, targets):
        # 计算损失的肯定是依存树预测损失
        return self.biaffine_attention.compute_loss(outputs, targets) * self.dep_loss_factor

    def forward_classifier(self, hidden_state: torch.Tensor, position_embedding: torch.Tensor):
        input = hidden_state.transpose(0, 1) + position_embedding
        output = self.biaffine_attention.forward_classifier(input)  # [b, tgt_len, tgt_len]
        return output

    def get_reference(self, sample_ids):
        head_ref = self.biaffine_attention.get_reference(sample_ids)
        return head_ref

    def get_head(self, score):
        return score.argmax(-1)

    def predict(self, hidden_state, position_embedding, sample, **kwargs):

        score = self.forward_classifier(hidden_state, position_embedding)

        sample_ids = sample['id'].cpu().tolist()
        head_ref = self.get_reference(sample_ids)

        head_mask = head_ref != -1

        predict = score[head_mask]
        head_ref = head_ref[head_mask]

        return score, predict, head_ref

    def forward(self, hidden_state, position_embedding, sample, reference):

        _, predict, target, = self.predict(hidden_state, position_embedding, sample)  # predict: =2的概率
        loss = self.compute_loss(predict, target)
        return {"dep_classifier_loss": {
            "loss": loss,
        }}, predict, target

    def inference_accuracy(self, hidden_state, position_embedding, compute_loss, target_token, sample, eval_accuracy,
                           result_mat):
        loss, all, correct = 0, 0, 0
        train_mat = None
        score, predict, head_ref = self.predict(hidden_state, position_embedding, sample, result_mat=result_mat)
        if compute_loss:
            loss = self.compute_loss(predict, head_ref)
        if eval_accuracy:
            all, correct = self.compute_accuracy(predict, head_ref)

        return loss, all, correct, train_mat

    def compute_accuracy(self, predict_head, head_ref):

        predict = predict_head.argmax(-1)
        target = head_ref
        all = len(predict)
        correct = (target == predict).sum().item()
        return all, correct

    def get_label(self, score, reference):
        # score: [b, l, l]
        dep_mat = score.new_tensor(score.size()).fill_(0)
        head = self.get_head(score)

        batch_size, tgt_len = score.shape
        for i in range(batch_size):
            ref_len = (reference[i] != self.padding_idx).sum().item()
            dep_mat[i][:ref_len, :ref_len] = 2  # 不相关
            for j in range(1, ref_len):
                h = head[i][j]
                if h == 0:
                    continue

                dep_mat[i][h][j] = 1
                dep_mat[i][j][h] = 1
                grand = head[i][h]
                if grand == 0:
                    continue
                dep_mat[i][grand][j] = 1
                dep_mat[i][j][grand] = 1

        return dep_mat


class DepCoarseClassifier(DepClassifier):
    def __init__(self, args, relative_dep_mat=None, use_two_class=False):
        super().__init__(args, relative_dep_mat, use_two_class)

        self.mlp_dim = 400
        self.class_num = 3
        if use_two_class:
            self.class_num = 1
            self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mlp_output = nn.Sequential(
            nn.Linear(self.args.decoder_embed_dim * 4, self.mlp_dim),
            nn.Tanh(),
            nn.Linear(self.mlp_dim, self.class_num)
        )

        self.positive_class_factor = getattr(self.args, "positive_class_factor", 1.0)

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

        loss = mean_ds(losses)

        loss = loss * self.dep_loss_factor
        return loss

    def forward_classifier(self, hidden_state: torch.Tensor, position_embedding: torch.Tensor):

        seq_len, batch_size, hidden_dim = hidden_state.size()
        input = torch.cat((hidden_state.transpose(0, 1), position_embedding), dim=-1)
        a = input.unsqueeze(1).repeat(1, seq_len, 1, 1)  # [b, l, l, d]
        b = input.unsqueeze(2).repeat(1, 1, seq_len, 1)  # [b, l, l, d]
        feature = torch.cat((a, b), dim=-1)

        score = self.mlp_output(feature)  # [b, l, l, 3]
        return score.squeeze(-1)

    def get_label(self, score, **kwargs):
        """预测->依存矩阵"""
        if self.class_num == 1:
            return (F.sigmoid(score) > 0.5).long() + 1
        else:
            _score, _label = F.log_softmax(score, dim=-1).max(-1)
            return _label + 1

    def predict(self, hidden_state, position_embedding, sample, reference, result_mat=False):

        score = self.forward_classifier(hidden_state, position_embedding)

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

    def inference_accuracy(self, hidden_state, position_embedding, compute_loss, reference, sample, eval_accuracy,
                           result_mat):

        loss, all, correct = 0, 0, 0
        score, predict, dependency_mat, train_mat = self.predict(hidden_state, position_embedding, sample, reference,
                                                                 result_mat=result_mat)
        if compute_loss:
            loss = self.compute_loss(predict, dependency_mat)
        if eval_accuracy:
            all, correct = self.compute_accuracy(predict, dependency_mat)

        return loss, all, correct, train_mat

    def compute_accuracy(self, predict, target):
        _label = self.get_label(predict)

        all = len(_label)
        correct = (target == _label).sum().item()
        return all, correct
