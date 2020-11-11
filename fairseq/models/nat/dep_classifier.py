from fairseq.models import register_model, BaseFairseqModel
from fairseq.models.nat import NAT


class DepCoarseClassifier(BaseFairseqModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

    #     self.arc_head_mlp = nn.Sequential(
    #         nn.Linear(mlp_input_dim, self.mlp_dim),
    #         nn.LeakyReLU(),
    #         nn.Dropout(0.33))
    #
    #     self.arc_dep_mlp = nn.Sequential(
    #         nn.Linear(mlp_input_dim, self.mlp_dim),
    #         nn.LeakyReLU(),
    #         nn.Dropout(0.33))
    #
    #     self.W_arc = nn.Parameter(orthogonal_(
    #         torch.empty(self.mlp_dim + 1, self.mlp_dim).cuda()
    #     ), requires_grad=True)
    #
    #
    # def forward(self, losses, sample):


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
        DepClassifierModel.add_args(parser)

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        losses = super().forward(src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs)

        predict_loss = self.dep_classifier.forward(losses)
        losses.update(predict_loss)

        return losses

    def froze_nmt_model(self):
        print("froze nmt model !!!!!!")

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.decoder.parameters():
            param.requires_grad = False

    def compute_accuracy(self, ):
        self.dep_classifier.compute_accuracy()
