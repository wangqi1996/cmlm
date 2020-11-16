# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import default_restore_location

from fairseq.dep import RelativeDepMat
from fairseq.file_io import PathManager
from fairseq.models import BaseFairseqModel
from fairseq.util2 import mean_ds


class DepCoarseClassifier(BaseFairseqModel):
    def __init__(self, args, relative_dep_mat=None):
        super().__init__()
        self.args = args

        self.mlp_dim = 400
        self.mlp_output = nn.Sequential(
            nn.Linear(self.args.decoder_embed_dim * 4, self.mlp_dim),
            nn.Tanh(),
            nn.Linear(self.mlp_dim, 3)
        )

        if relative_dep_mat is None:
            self.relative_dep_mat = RelativeDepMat(valid_subset=self.args.valid_subset)
        else:
            self.relative_dep_mat = relative_dep_mat

        if getattr(args, "load_checkpoint", "") != "":
            self.load_parameters(self.args.load_checkpoint)

    def compute_loss(self, outputs, targets):
        # 多分类
        logits = F.log_softmax(outputs, dim=-1)
        targets = targets - 1  # 去除pad类别
        losses = F.nll_loss(logits, targets.to(logits.device), reduction='none')

        loss = mean_ds(losses)
        return loss

    def forward_classifier(self, hidden_state: torch.Tensor, position_embedding: torch.Tensor):

        seq_len, batch_size, hidden_dim = hidden_state.size()
        input = torch.cat((hidden_state.transpose(0, 1), position_embedding), dim=-1)
        a = input.unsqueeze(1).repeat(1, seq_len, 1, 1)  # [b, l, l, d]
        b = input.unsqueeze(2).repeat(1, 1, seq_len, 1)  # [b, l, l, d]
        feature = torch.cat((a, b), dim=-1)

        score = self.mlp_output(feature)  # [b, l, l, 3]
        return score

    def predict(self, hidden_state, position_embedding, sample, reference):

        score = self.forward_classifier(hidden_state, position_embedding)
        sample_ids = sample['id'].cpu().tolist()
        dependency_mat = self.relative_dep_mat.get_dependency_mat(sample_ids, reference, training=self.training,
                                                                  contain_eos=True)

        reference_mask = dependency_mat != 0  # 计算损失时仅仅计算相关和不相关两类。
        predict = score[reference_mask]
        dependency_mat = dependency_mat[reference_mask]

        return score, predict, dependency_mat

    def forward(self, hidden_state, position_embedding, sample, reference):
        """
        :param hidden_state:  [batch_size, tgt_len, dim]
        :param position_embedding: [batch_size, tgt_len, dim]
        :param sample:
        :return:
        """
        _, predict, target = self.predict(hidden_state, position_embedding, sample, reference)  # predict: =2的概率
        loss = self.compute_loss(predict, target)
        return {"dep_classifier_loss": {
            "loss": loss,
        }}, predict, target

    def inference_accuracy(self, hidden_state, position_embedding, compute_loss, target_token, sample, eval_accuracy):

        loss, all, correct = 0, 0, 0
        score, predict, dependency_mat = self.predict(hidden_state, position_embedding, sample, target_token)
        if compute_loss:
            loss = self.compute_loss(predict, dependency_mat)
        if eval_accuracy:
            all, correct = self.compute_accuracy(predict, dependency_mat)
        return loss, all, correct

    def inference(self, hidden_state, position_embedding, **kwargs):
        """
        依然是二分类，是同一个词这种关系本身应该是先验。
        计算损失时只计算不是0（pad） or 3（同一个词）的位置
        """
        score = self.forward_classifier(hidden_state, position_embedding)
        _score, _label = F.log_softmax(score, dim=-1).max(-1)
        return _label + 1

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
        parser.add_argument('--compute-confusion-matrix', action="store_true")
        parser.add_argument('--load-checkpoint', type=str, default="")

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
