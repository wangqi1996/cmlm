# encoding=utf-8

import torch.nn.functional as F

from fairseq.models.nat import CMLMNATransformerModel, register_model, cmlm_base_architecture, \
    register_model_architecture
from fairseq.models.nat.old.latent.latent import get_prior, \
    get_posterior
from fairseq.util2 import get_base_mask


@register_model("dep_nat")
class DepNAT(CMLMNATransformerModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        if not hasattr(args, "loss_function"):
            setattr(args, "loss_function", "nmt_loss,length_loss")
        self.loss_function = self.args.loss_function.strip().split(',')
        self.predictor_loss = 'mask_predictor'
        self.nmt_loss = 'nmt_loss'
        self.length_loss = 'length_loss'
        self.use_mask_predictor = self.predictor_loss in self.loss_function

        arch = args.latent_arch
        self.prior = get_prior(arch, args, self.pad)
        self.posterior = get_posterior(arch, args, self.pad)

    def get_loss(self, prior_out, posterior_out, base_mask):
        """ logits"""
        prior_out = prior_out.log_softmax(dim=-1)
        posterior_out = posterior_out.softmax(dim=-1)

        loss = F.kl_div(prior_out, posterior_out, reduction='none')
        loss = loss.sum(-1)

        useful_loss = loss[base_mask]
        useful_loss = useful_loss.mean()
        return useful_loss

    def froze_prior(self, froze=True):
        if froze:
            for param in self.prior.parameters():
                param.requires_grad = False
        else:
            for param in self.prior.parameters():
                param.requires_grad = True

    def froze_posterior(self, froze=True):
        if froze:
            for param in self.posterior.parameters():
                param.requires_grad = False
        else:
            for param in self.posterior.parameters():
                param.requires_grad = True

    @staticmethod
    def add_args(parser):
        CMLMNATransformerModel.add_args(parser)

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, reference, need_sample=True, **kwargs
    ):
        loss = super().forward(src_tokens, src_lengths, prev_output_tokens, reference, **kwargs)

        logits = loss['word_ins']['out']

        if need_sample:
            # predict_word
            predict_word = logits.max(-1)[1]
            src_embed, _ = self.encoder.forward_embedding(src_tokens)
            tgt_embed, decoder_padding_mask = self.decoder.forward_embedding(predict_word, states=None)
            reference_embed, _ = self.decoder.forward_embedding(reference, states=None)  # embedding会更新
            # prior
            encoder_out = loss['train_need']['encoder_out']
            prior_out = self.prior(src_embed=src_embed, src_token=src_tokens, trg_prev_embed=tgt_embed,
                                   trg_token=predict_word, encoder_out=encoder_out)
            # posterior
            posterior_out = self.posterior(src_embed=src_embed, src_token=src_tokens, trg_prev_embed=tgt_embed,
                                           trg_token=predict_word, reference_embed=reference_embed,
                                           reference_token=reference, encoder_out=encoder_out)

            # sample multiple
            prior_sample = F.gumbel_softmax(prior_out, hard=True, tau=1)  # index=0表示mask
            posterior_sample = F.gumbel_softmax(posterior_out, hard=True, tau=1)

            # KL loss
            base_mask = get_base_mask(reference)
            kl_loss = self.get_loss(prior_out, posterior_out.detach(), base_mask)

            # posterior loss --> reference_mask
            # reference_mask = ~get_reference_mask(reference, predicted=predict_word)  # True表示mask

            loss.update({
                "latent": {
                    "loss": kl_loss
                },
                "train_need": {
                    "prior_sample": prior_sample,
                    "posterior_sample": posterior_sample,
                    "predict_word": predict_word,
                },
                # "posterior": {
                #     "out": posterior_out,
                #     "tgt": reference_mask.long(),
                #     "mask": base_mask,
                #     "factor": 0.5
                # }
            })
        return loss

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):

        reference = kwargs['references']
        use_reference_mask = kwargs['use_reference_mask']

        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # random mask references  为了统计不同的reference mask概率时的准确率
        # output_tokens = _random_mask(reference, noise_probability=0.8)

        # execute the decoder
        base_mask = get_base_mask(output_tokens)
        output_masks = output_tokens.eq(self.unk)
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
        ).max(-1)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        # output_masks = get_base_mask(reference)
        # need_predict_token = output_masks.float().sum().item()
        # correct_mask = output_masks & (output_tokens == reference)
        # correct_predict_token = (output_masks & (output_tokens == reference)).float().sum().item()
        # set_diff_tokens(correct_predict_token)
        # set_all_token(need_predict_token)
        # token_index = output_tokens[correct_mask].cpu().tolist()
        # for i in token_index:
        #     set_step_value(0, i)

        if history is not None:
            history.append(output_tokens.clone())

        src_tokens = kwargs['src_tokens']
        src_embed, _ = self.encoder.forward_embedding(src_tokens)
        tgt_embed, decoder_padding_mask = self.decoder.forward_embedding(output_tokens, states=None)
        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
            # skeptical_mask = _skeptical_unmasking(
            #     output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
            # )

            # 使用先验估计一下
            prior_out = self.prior(src_embed=src_embed, src_token=src_tokens, trg_prev_embed=tgt_embed,
                                   trg_token=output_tokens, encoder_out=encoder_out)

            # reference_embed, _ = self.decoder.forward_embedding(reference, states=None)
            # posterior_out = self.posterior(src_embed=src_embed, src_token=src_tokens, trg_prev_embed=tgt_embed,
            #                                trg_token=output_tokens, reference_embed=reference_embed,
            #                                reference_token=reference)
            # prior_out = posterior_out
            #
            prior_sample = F.gumbel_softmax(prior_out, hard=True, tau=1)

            skeptical_mask = prior_sample[:, :, 0].bool() & base_mask

            # # 计算0/1比例
            # need_predict = base_mask.sum().item()
            # predict_0 = (skeptical_mask == 1).sum().item()  # mask的概率
            # set_diff_tokens(need_predict)
            # set_all_token(predict_0)

            # # 计算先验准确率
            # reference_mask = (output_tokens == reference) & base_mask  # reference认为是正确的
            # prior_mask = (~skeptical_mask) & base_mask  # prior认为是正确的
            # common = reference_mask & prior_mask
            # set_value1(common.sum().item())
            # correct = reference_mask.sum().item()
            # set_value2(correct)
            # prior_correct = prior_mask.sum().item()
            # set_value3(prior_correct)

            # if use_reference_mask:
            #     skeptical_mask = get_reference_mask(reference, output_tokens)
            # else:
            #     skeptical_mask = _skeptical_unmasking(
            #         output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
            #     )
            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history
        )


@register_model_architecture("dep_nat", "dep_nat")
def dep_nat_architecture(args):
    cmlm_base_architecture(args)
