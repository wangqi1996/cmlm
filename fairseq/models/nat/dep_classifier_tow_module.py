# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import default_restore_location

from fairseq.dep import RelativeDepMat
from fairseq.file_io import PathManager
from fairseq.models.nat import DepCoarseClassifier
from fairseq.util2 import mean_ds, set_diff_tokens, set_all_token, set_key_value


class DepCoarseTwoClassifier(DepCoarseClassifier):
    def __init__(self, args, relative_dep_mat=None):
        super().__init__(args, relative_dep_mat=relative_dep_mat)

        self.mlp_output = nn.Sequential(
            nn.Linear(self.args.decoder_embed_dim * 4, self.mlp_dim),
            nn.Tanh(),
            nn.Linear(self.mlp_dim, 1)
        )

        if relative_dep_mat is None:
            self.relative_dep_mat = RelativeDepMat(valid_subset=self.args.valid_subset, use_two_class=True)
        else:
            self.relative_dep_mat = relative_dep_mat

    def compute_loss(self, outputs, targets):
        # 多分类
        losses = self.loss(outputs, targets - 1)

        if self.positive_class_factor != 1.0:
            weigths = losses.new_ones(size=losses.size()).float()
            weigths.masked_fill_(targets == 1, self.positive_class_factor)
            losses = losses * weigths

        loss = mean_ds(losses)

        loss = loss * self.dep_loss_factor
        return loss

    def predict(self, hidden_state, position_embedding, sample, reference, result_mat=False):

        score = self.forward_classifier(hidden_state, position_embedding)

        sample_ids = sample['id'].cpu().tolist()
        dependency_mat = self.relative_dep_mat.get_dependency_mat(sample_ids, reference, training=self.training,
                                                                  contain_eos=True)

        reference_mask = dependency_mat != 0
        mat = None
        if result_mat:
            _label = F.sigmoid(score) > 0.5
            _label = _label + 1
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
        _score, _label = F.log_softmax(score, dim=-1).max(-1)
        _label = _label + 1
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
        _score, _label = F.log_softmax(predict, dim=-1).max(-1)

        all = len(_label)
        target = target - 1
        correct = (target == _label).sum().item()

        # if self.args.compute_confusion_matrix:
        #     predict_positive = (predict == 1).sum().item()  # 1 是相关
        #     target_positive = (target == 1).sum().item()
        #     correct_positive = ((predict == 1) & (target == 1)).sum().item()
        #
        #     predict_negative = (predict == 2).sum().item()
        #     target_negative = (target == 2).sum().item()
        #     correct_negative = ((predict == 2) & (target == 2)).sum().item()
        #
        #     set_value1(predict_positive)
        #     set_value2(target_positive)
        #     set_value3(correct_positive)
        #     set_value4(predict_negative)
        #     set_value5(target_negative)
        #     set_value6(correct_negative)

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
