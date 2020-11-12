import torch
from torch import nn

from fairseq.models import register_model, BaseFairseqModel
from fairseq.models.nat import NAT, orthogonal_, DepChildTree, DepHeadTree, get_dependency_mat, \
    get_coarse_dependency_mat, register_model_architecture, nat_iwslt16_de_en
import torch.nn.functional as F

class DepCoarseClassifier(BaseFairseqModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.W_arc = nn.Parameter(orthogonal_(
            torch.empty(self.args.decoder_embed_dim * 2, self.args.decoder_embed_dim * 2).cuda()
        ), requires_grad=True)

        self.child_tree = DepChildTree(valid_subset=self.args.valid_subset)
        self.head_tree = DepHeadTree(valid_subset=self.args.valid_subset)

        self.dep_mat_grain = getattr(args, "dep_mat_grain", "fine")
        print(self.dep_mat_grain)

        self.loss = nn.BCEWithLogitsLoss(reduction="none")

    def get_dependency_mat(self, sample_ids, target_token):
        if self.head_tree is not None and self.child_tree is not None:
            if self.dep_mat_grain == "fine":
                dependency_mat = get_dependency_mat(self.head_tree, self.child_tree, sample_ids, self.training,
                                                    target_token)
            elif self.dep_mat_grain == "coarse":
                dependency_mat = get_coarse_dependency_mat(self.head_tree, self.child_tree, sample_ids, self.training,
                                                           target_token, contain_eos=True)
        else:
            dependency_mat = None

        return dependency_mat

    def forward(self, hidden_state, position_embedding, sample, reference):
        """
        :param hidden_state:  [batch_size, tgt_len, dim]
        :param position_embedding: [batch_size, tgt_len, dim]
        :param sample:
        :return:
        """

        predict, target = self.predict(hidden_state, position_embedding, sample, reference)  # predict: =2的概率
        loss = self.loss(predict, (target - 1).float())
        loss = self.mean_ds(loss)
        return {"dep_classifier_loss": {
            "loss": loss,
        }}, predict, target

    def mean_ds(self, x: torch.Tensor, dim=None) -> torch.Tensor:
        return (
            x.float().mean().type_as(x)
            if dim is None
            else x.float().mean(dim).type_as(x)
        )

    def predict(self, hidden_state, position_embedding, sample, reference):
        sample_ids = sample['id'].cpu().tolist()
        input = torch.cat((hidden_state.transpose(0, 1), position_embedding), dim=-1)

        score = input.matmul(self.W_arc).matmul(input.transpose(1, 2))  # [b, l, l]

        dependency_mat = self.get_dependency_mat(sample_ids, reference)  # [b,l,l]

        reference_mask = ~(dependency_mat == 0)
        predict = score[reference_mask]
        dependency_mat = dependency_mat[reference_mask]

        return predict, dependency_mat

    def compute_accuracy(self, predict, target, threshold=0.5):
        predict = F.sigmoid(predict) > threshold + 1

        all = len(predict)
        correct = (predict == target).sum().item()

        # predict_1 = (predict == 1).sum().item() # 1 是相关
        # dependency_1 = (dependency_mat == 1).sum().item()
        #
        # correct_1 = ((predict == 1) & (dependency_mat == 1)).sum().item()

        return all, correct

    @staticmethod
    def add_args(parser):
        parser.add_argument('--dep-mat-grain', type=str, default="coarse", choices=['fine', 'coarse'])


@register_model("dep_classifier")
class DepClassifierModel(NAT):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        self.dep_classifier = DepCoarseClassifier(args)

        if self.args.froze_nmt_model:
            self.froze_nmt_model()

    @staticmethod
    def add_args(parser):
        NAT.add_args(parser)
        DepCoarseClassifier.add_args(parser)

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        # 获取hidden state + position embedding
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=True, **kwargs)

        # decoding
        word_ins_out, other = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            inner=True,
            return_position_embedding=True
        )
        hidden_state = other['inner_states'][-1]
        position_embedding = other['position_embedding']
        sample = kwargs['sample']

        loss, predict, target = self.dep_classifier.forward(hidden_state, position_embedding, sample, tgt_tokens)

        if kwargs.get("eval_accuracy", False):
            all_predict_head, correct_predict_head = self.dep_classifier.compute_accuracy(predict, target)
            loss.setdefault('train_need', {})
            loss['train_need'].update({
                "print": {
                    "all_predict_head": all_predict_head,
                    "correct_predict_head": correct_predict_head
                }})

        return loss

    def froze_nmt_model(self):
        print("froze nmt model !!!!!!")

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.decoder.parameters():
            param.requires_grad = False


@register_model_architecture('dep_classifier', 'dep_classifier_iwslt16_en_de')
def dep_classifier_iwslt16_en_de(args):
    nat_iwslt16_de_en(args)
