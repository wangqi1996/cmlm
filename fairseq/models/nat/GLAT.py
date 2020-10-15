# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from fairseq.models import register_model_architecture, register_model
from fairseq.models.nat import NATransformerModel, base_architecture
from fairseq.util2 import new_arange


@register_model('GLAT')
class GLAT(NATransformerModel):

    def hamming_distance(self, reference, predict):
        diff = (reference != predict).sum(-1).detach()
        return diff

    def random_mask(self, target_tokens, mask_num, input):
        """
        GLAT的构造input的办法
        """
        pad = self.tgt_dict.pad()
        bos = self.tgt_dict.bos()
        eos = self.tgt_dict.eos()
        unk = self.tgt_dict.unk()

        target_masks = target_tokens.ne(pad) & \
                       target_tokens.ne(bos) & \
                       target_tokens.ne(eos)
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(~target_masks, 2.0)
        target_length = mask_num
        target_length = target_length + 1  # make sure to mask at least one token.

        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < target_length[:, None].long()
        mask = target_cutoff.scatter(1, target_rank, target_cutoff)
        prev_target_tokens = target_tokens.masked_fill(mask, unk)

        # cover
        prev_target_embedding = self.decoder.forward_embedding(prev_target_tokens)
        prev_target_embedding.masked_scatter_(mask, input[mask])

        return prev_target_embedding

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        """
        GLAT: 两步解码
        1. 输入为没有梯度的decode一遍
        2. 计算hamming距离（和reference有多少token不一致
        3. 随机采样（其实是确定了mask的概率。
        4. hidden state和word embedding混合
        """
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        # first step decoding  使用embedding copy
        logits, input_embedding = self.decoder(normalize=False,
                                               prev_output_tokens=prev_output_tokens,
                                               encoder_out=encoder_out,
                                               step=0,
                                               return_input=True
                                               ).detach()
        score, predict_word = logits.max(-1)

        # 计算hamming距离
        reference = kwargs['sample']['target']
        mask_num = self.hamming_distance(reference=reference, predict=predict_word, )

        prev_output_tokens = self.random_mask(reference, mask_num, input_embedding)

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out)

        return {
            "word_ins": {
                "out": word_ins_out, "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad), "ls": self.args.label_smoothing,
                "nll_loss": True
            },
            "length": {
                "out": length_out, "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor
            }
        }


@register_model_architecture("GLAT", "GLAT")
def GLAT(args):
    base_architecture(args)
