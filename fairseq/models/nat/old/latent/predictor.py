import torch
import torch.nn.functional as F
from torch import nn

from fairseq.models.nat.old.latent.predictor_module import LogisticModel, BCELoss, DecoderLayer, EncoderLayer, \
    EncoderDecoder, LSTMLayer, mean_ds


# def get_loss(self, hidden_state, logits, decoder_input, reference, length_out, length_tgt, encoder_out):
#
#     loss = {}
#     target_mask = decoder_input.eq(self.unk)
#
#     if self.predictor_loss in self.loss_function:
#         output_score, predict_words = F.log_softmax(logits, -1).max(-1)  # 做一个log_softmax并不会改变单调性，所以最大的还是最大的
#         reference_mask = get_reference_mask(reference, predict_words)  # true表示需要mask的字符
#
#         # train
#         predict_mask = self.decoder.mask_predictor(encoder_out=encoder_out,
#                                                    hidden_state=hidden_state,
#                                                    decoder_input=decoder_input,
#                                                    logits=logits,
#                                                    nmt_model=self,
#                                                    predict_word=predict_words,
#                                                    normalize=False,
#                                                    output_score=output_score)
#
#         # loss
#         loss_value = self.decoder.mask_predictor.loss(target_mask, predict_mask, reference_mask)
#
#         mask_predict = {
#             "loss": loss_value
#         }
#         loss.update({"mask_predict": mask_predict})
#
#     if self.nmt_loss in self.loss_function:
#         word_in = {
#             "out": logits, "tgt": reference,
#             "mask": target_mask, "ls": self.args.label_smoothing,
#             "nll_loss": True
#         }
#         loss.update({"word_in": word_in})
#
#     if self.length_loss in self.loss_function:
#         length = {
#             "out": length_out, "tgt": length_tgt,
#             "factor": self.decoder.length_loss_factor
#         }
#         loss.update({"length": length})
#
#     return loss
class _Predictor(nn.Module):

    def __init__(self, args, pad=0, **unused):
        super().__init__()

        self.args = args
        self.padding_idx = pad
        self.loss_class = None
        self.predictor = None

    def loss_func(self, outputs, targets):
        return self.loss_class(outputs, targets)

    def forward(self, decoder_out=None, **unused):
        return self.predictor(decoder_out=decoder_out, **unused)


class LogisticPredict(_Predictor):
    def __init__(self, args, pad, **unused):
        super().__init__(args, pad)

        self.predictor = LogisticModel(args, activation=nn.Sigmoid, contain_logistic=True, dropout=0.2, **unused)
        self.loss_class = BCELoss()


class DecoderLayerPredictor(_Predictor):
    def __init__(self, args, pad, **unused):
        super().__init__(args, pad)

        self.predictor = DecoderLayer(args, num_layers=1, pad=pad)
        self.loss_class = BCELoss()

    def forward(self, decoder_out=None, encoder_out=None, **unused):
        return self.predictor(encoder_output=encoder_out.encoder_out,
                              encoder_padding_mask=encoder_out.encoder_padding_mask, decoder_out=decoder_out, **unused)


class EncoderPredictor(_Predictor):
    def __init__(self, args, pad, **unused):
        super().__init__(args, pad)

        self.predictor = EncoderLayer(args, num_layers=1, pad=pad)
        self.loss_class = BCELoss()


class EncoderDecoderPredictor(_Predictor):
    def __init__(self, args, pad, **unused):
        super().__init__(args, pad)
        self.predictor = EncoderDecoder(args, pad=pad, encoder_layers=3, decoder_layers=3)
        self.loss_class = BCELoss()


class LSTMPredictor(_Predictor):
    def __init__(self, args, pad, **unused):
        super().__init__(args, pad)
        self.predictor = LSTMLayer(args, num_layers=1, pad=pad)
        self.loss_class = BCELoss()


class LSTMPredictor2(_Predictor):
    def __init__(self, args, pad, **unused):
        super().__init__(args, pad)
        self.predictor = LSTMLayer(args, num_layers=2, pad=pad)
        self.loss_class = BCELoss()


class MaskPredict(nn.Module):
    def __init__(self, args, pad, nmt_model=None, **unused):
        super().__init__()
        self.args = args
        mask_method = self.args.mask_predictor_method
        self.predictor: _Predictor = MAPPING[mask_method](args, pad)

    def loss(self, masks, outputs, targets, factor=1.0):
        if self.args.mask_predictor_method in ["margin"]:
            return self.predictor.loss_func(outputs, targets, masks)
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]  # mask表示需要计算的，mask=False表示之前预测好的，不在本轮计算损失

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            losses = self.predictor.loss_func(outputs, targets)
            nll_loss = mean_ds(losses)
            loss = nll_loss

        loss = loss * factor
        return loss

    # reference=1表示要被mask, outputs=1表示不被mask

    def forward(self, encoder_out, hidden_state, decoder_input, logits, nmt_model, predict_word=None,
                normalize=False, output_score=None):
        # mask
        # 学习目标：得分越高，越不容易mask
        decoder_out = self.get_input(hidden_state, logits, nmt_model, predict_word, output_score)
        return self.predictor(decoder_input=decoder_input, decoder_out=decoder_out, normalize=normalize,
                              encoder_out=encoder_out)

    def get_input(self, hidden_state, logits, nmt_model, predict_word=None, output_score=None):

        embedding_layer = nmt_model.decoder.embed_tokens

        if not hasattr(self.args, "predictor_input") or self.args.predictor_input == "hidden-state":
            return hidden_state

        elif self.args.predictor_input == 'expect-embedding':
            probability = F.softmax(logits, dim=-1)  # [b, len, vocab_size]
            embedding_weight = embedding_layer.weight  # [vocab_size, dim]
            expect = torch.matmul(probability, embedding_weight)  # [batch, len, embed_size]
            return expect

        elif self.args.predictor_input == 'token-embedding':
            if predict_word is None:
                predict_word = logits.max(-1)[1]
            embed = embedding_layer(predict_word)  # [batch, len, embed_size]
            return embed

        elif self.args.predictor_input == "score":
            input = hidden_state.mul(output_score.unsqueeze(-1))
            return input


# encoding=utf-8
MAPPING = {
    # "mix-attention": MixAttentionPredict,
    # "linear": LinearPredict,
    "logistic": LogisticPredict,
    "re-logistic": LogisticPredict,
    # "margin": MarginPredict,
    'decoder': DecoderLayerPredictor,
    'encoder': EncoderPredictor,
    'lstm': LSTMPredictor,
    "encoderdecoder": EncoderDecoderPredictor,
    'lstm2': LSTMPredictor2,
}
