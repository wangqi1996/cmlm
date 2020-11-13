from fairseq.models import register_model
from fairseq.models.nat import NAT, register_model_architecture, nat_iwslt16_de_en, DepCoarseClassifier

SuperClass = NAT

"""
可以是joint training NMT任务和相关判别任务。
也可以仅仅训练一个二分类器。
"""


@register_model("dep_classifier")
class DepClassifierModel(SuperClass):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        self.dep_classifier = DepCoarseClassifier(args)

        if self.args.froze_nmt_model:
            self.froze_nmt_model()

        self.hidden_state_layer = getattr(args, "hidden_state_layer", -2)
        print(self.hidden_state_layer)

    @staticmethod
    def add_args(parser):
        SuperClass.add_args(parser)
        DepCoarseClassifier.add_args(parser)
        parser.add_argument('--hidden_state_layer', type=int, default=-2)  # -1

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
            return_position_embedding=True,
            **kwargs
        )
        # 定义nmt loss
        loss = {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True
            }
        }
        # length prediction
        if self.decoder.length_loss_factor > 0:
            length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
            length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)
            loss["length"] = {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor
            }

        hidden_state = other['inner_states'][self.hidden_state_layer]
        position_embedding = other['position_embedding']
        sample = kwargs['sample']

        dep_loss, predict, target = self.dep_classifier.forward(hidden_state, position_embedding, sample, tgt_tokens)
        loss.update(dep_loss)

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
