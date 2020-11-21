# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import default_restore_location

from fairseq.dep import RelativeDepMat
from fairseq.file_io import PathManager
from fairseq.models import BaseFairseqModel
from fairseq.util2 import mean_ds, set_diff_tokens, set_all_token, set_key_value


class DepCoarseClassifier(BaseFairseqModel):
    def __init__(self, args, relative_dep_mat=None, use_two_class=False):
        super().__init__()
        self.args = args

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

        if relative_dep_mat is None:
            self.relative_dep_mat = RelativeDepMat(valid_subset=self.args.valid_subset, use_two_class=use_two_class)
        else:
            self.relative_dep_mat = relative_dep_mat

        if getattr(args, "load_checkpoint", "") != "":
            self.load_parameters(self.args.load_checkpoint)

        self.dep_loss_factor = getattr(self.args, "dep_loss_factor", 1.0)
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
        return score

    def get_label(self, score):
        """score: logits"""
        if self.class_num == 1:
            return (F.sigmoid(score) > 0.5 + 1).long()
        else:
            _score, _label = F.log_softmax(score, dim=-1).max(-1)
            return _label + 1

    def predict(self, hidden_state, position_embedding, sample, reference, result_mat=False):

        score = self.forward_classifier(hidden_state, position_embedding)

        sample_ids = sample['id'].cpu().tolist()
        dependency_mat = self.relative_dep_mat.get_dependency_mat(sample_ids, reference, training=self.training,
                                                                  contain_eos=True)

        reference_mask = dependency_mat != 0
        mat = None
        if result_mat:
            _label = self.get_label(score)
            mat = _label.masked_fill(~reference_mask, 0)

        predict = score[reference_mask]
        dependency_mat = dependency_mat[reference_mask]

        return score, predict, dependency_mat, mat

    def forward(self, hidden_state, position_embedding, sample, reference):
        """
        :param hidden_state:  [batch_size, tgt_len, dim]
        :param position_embedding: [batch_size, tgt_len, dim]
        :param sample:
        :return:
        """
        _, predict, target, _ = self.predict(hidden_state, position_embedding, sample, reference)  # predict: =2的概率
        loss = self.compute_loss(predict, target)
        return {"dep_classifier_loss": {
            "loss": loss,
        }}, predict, target

    def inference_accuracy(self, hidden_state, position_embedding, compute_loss, target_token, sample, eval_accuracy,
                           result_mat):

        loss, all, correct = 0, 0, 0
        score, predict, dependency_mat, train_mat = self.predict(hidden_state, position_embedding, sample, target_token,
                                                                 result_mat=result_mat)
        if compute_loss:
            loss = self.compute_loss(predict, dependency_mat)
        if eval_accuracy:
            all, correct = self.compute_accuracy(predict, dependency_mat)

        return loss, all, correct, train_mat

    def inference(self, hidden_state, position_embedding, **kwargs):
        """
        依然是二分类，是同一个词这种关系本身应该是先验。
        计算损失时只计算不是0（pad） or 3（同一个词）的位置
        """
        score = self.forward_classifier(hidden_state, position_embedding)
        _label = self.get_label(score)
        # batch_size=1
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

    def compute_accuracy(self, predict, target):
        _label = self.get_label(predict)

        all = len(_label)
        correct = (target == _label).sum().item()
        return all, correct

    @staticmethod
    def add_args(parser):
        parser.add_argument('--dep-mat-grain', type=str, default="coarse", choices=['fine', 'coarse'])
        parser.add_argument('--load-checkpoint', type=str, default="")
        parser.add_argument('--dep-loss-factor', default=1.0, type=float)

    def load_parameters(self, model_path):
        with PathManager.open(model_path, "rb") as f:
            state = torch.load(
                f, map_location=lambda s, l: default_restore_location(s, "cpu")
            )

        new_state = {}
        for k, v in state['model'].items():
            if "dep_classifier." in k:
                k = ".".join(k.split('.')[1:])
                new_state[k] = v
        self.load_state_dict(new_state)

        print("load parameters for dep_classifier!")
