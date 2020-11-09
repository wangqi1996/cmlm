# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import Tensor

from fairseq.models import register_model_architecture, register_model
from fairseq.models.nat import base_architecture
from fairseq.models.nat.nat_base import NATTransformerModel
from fairseq.util2 import new_arange


@register_model('GLAT')
class GLAT(NATTransformerModel):

    def hamming_distance(self, reference: Tensor, predict: Tensor):
        diff = (reference != predict).sum(-1).detach()
        return diff

    def get_ratio(self, update_num, max_steps):
        return update_num / max_steps + 0.1

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
        prev_target_embedding, _ = self.decoder.forward_embedding(prev_target_tokens)

        new_embedding = torch.cat((prev_target_embedding.unsqueeze(-1), input.unsqueeze(-1)), dim=-1)
        mask = mask.unsqueeze(-1).unsqueeze(-1)
        new_mask = torch.cat((~mask, mask), dim=-1)
        embedding = (new_embedding * new_mask).sum(-1)

        return embedding, prev_target_tokens

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
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)

        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        # first step decoding  使用embedding copy
        logits, _decoder_out = self.decoder(normalize=False,
                                            prev_output_tokens=prev_output_tokens,
                                            encoder_out=encoder_out,
                                            step=0,
                                            return_input=True,
                                            inner=True
                                            )
        input_embedding = _decoder_out['return_input']
        logits = logits.detach()
        score, predict_word = logits.max(-1)

        # 计算hamming距离
        reference = kwargs['sample']['target']
        mask_num = self.hamming_distance(reference=reference, predict=predict_word).int()
        if self.training:
            mask_num = mask_num * self.get_ratio(update_num=kwargs['update_nums'], max_steps=self.args.max_update)

        prev_target_embedding, prev_output_tokens = self.random_mask(reference, mask_num, input_embedding)

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            prev_target_embedding=prev_target_embedding,
            step=1)

        return {
            "word_ins": {
                "out": word_ins_out, "tgt": tgt_tokens,
                "mask": prev_output_tokens.eq(self.unk), "ls": self.args.label_smoothing,
                "nll_loss": True
            },
            "length": {
                "out": length_out, "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor
            }
        }

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        kwargs['tgt_tokens'] = tgt_tokens
        kwargs['extra_ret'] = False
        super().forward_dcoder()


@register_model_architecture("GLAT", "GLAT")
def GLAT(args):
    base_architecture(args)
