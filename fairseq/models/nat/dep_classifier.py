from fairseq.models import register_model
from fairseq.models.nat import register_model_architecture, nat_iwslt16_de_en, DepCoarseClassifier, GLAT

SuperClass = GLAT

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

        self.predict_dep_relative_layer = getattr(args, "predict_dep_relative_layer", -2)
        print("predict_dep_relative_layer: ", self.predict_dep_relative_layer)

    @staticmethod
    def add_args(parser):
        SuperClass.add_args(parser)
        DepCoarseClassifier.add_args(parser)
        # decoder=100 表示decoder_input
        parser.add_argument('--predict-dep-relative-layer', type=int, default=100)

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=True, **kwargs)

        # decoding
        word_ins_out, other = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            inner=True,
            **kwargs
        )
        word_ins_out.detach_()
        _score, predict = word_ins_out.max(-1)
        mask_num = self.get_mask_num(tgt_tokens, predict)

        decoder_input = other['embedding']
        samples = kwargs['sample']
        output_token, output_embedding = self.get_mask_output(decoder_input=decoder_input, reference=tgt_tokens,
                                                              mask_length=mask_num, samples=samples, encoder_out=None)

        # decoder
        word_ins_out, other = self.decoder(
            normalize=False,
            prev_output_tokens=output_token,
            encoder_out=encoder_out,
            inner=True,
            prev_target_embedding=output_embedding,
            post_process_function=self.post_process_after_layer,
            **kwargs
        )

        # 计算hamming距离
        losses = None
        if not self.args.froze_nmt_model:
            losses = {
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
                losses["length"] = {
                    "out": length_out,
                    "tgt": length_tgt,
                    "factor": self.decoder.length_loss_factor
                }

        if self.predict_dep_relative_layer == 100:
            hidden_state = other['inner_states'][0]
        else:
            hidden_state = other['inner_states'][self.predict_dep_relative_layer + 1]
        position_embedding = other['position_embedding']
        sample = kwargs['sample']

        loss, predict, reference_mat = self.dep_classifier(hidden_state, position_embedding, sample,
                                                           reference=tgt_tokens)

        if kwargs.get("eval_accuracy", False):
            all_predict_head, correct_predict_head = self.dep_classifier.compute_accuracy(predict, reference_mat)
            loss.setdefault('train_need', {})
            loss['train_need'].update({
                "print": {
                    "all_predict_head": all_predict_head,
                    "correct_predict_head": correct_predict_head
                }})

        if losses is not None:
            loss.update(losses)
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
