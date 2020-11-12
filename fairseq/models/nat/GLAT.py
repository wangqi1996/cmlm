# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import torch
from torch import Tensor

from fairseq.dep import DepLayerTree
from fairseq.models import register_model_architecture, register_model
from fairseq.util2 import get_base_mask, new_arange
from .nat_base import NAT, nat_iwslt16_de_en


@register_model('GLAT')
class GLAT(NAT):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.mask_method = args.mask_method

        if self.mask_method == "layer":
            self.dependency_tree = DepLayerTree(valid_subset=self.args.valid_subset)

    @staticmethod
    def add_args(parser):
        NAT.add_args(parser)

        parser.add_argument('--mask-method', default="random")  # random、layer

    def hamming_distance(self, reference: Tensor, predict: Tensor):
        reference_mask = get_base_mask(reference)

        diff = ((reference != predict) & reference_mask).sum(-1).detach()

        return diff

    def get_ratio(self, update_num, max_steps):
        return 0.5

    def get_mask_num(self, reference, predict):
        distance = self.hamming_distance(reference, predict)
        ratio = self.get_ratio(None, None)

        mask_num = distance * ratio  # 使用reference的数目 == 不使用decoder input的数目
        return mask_num

    def get_mask_output(self, mask_length=None, reference=None, samples=None, encoder_out=None, decoder_input=None):

        reference_embedding, _, _ = self.decoder.forward_embedding(reference)

        kwargs = {
            "mask_length": mask_length,
            "reference": reference,
            "samples": samples,
            "encoder_out": encoder_out,
            "decoder_input": decoder_input,
            "reference_embedding": reference_embedding
        }
        if self.mask_method == "random":
            return self.get_random_mask_output(**kwargs)
        elif self.mask_method == "layer":
            return self.get_layer_mask_output(**kwargs)

    def get_layer_mask_output(self, mask_length=None, reference=None, samples=None, encoder_out=None,
                              decoder_input=None, reference_embedding=None):

        sample_ids = samples['id'].cpu().tolist()
        mask = reference.new_empty(reference.size(), requires_grad=False).fill_(False).bool()

        for index, id in enumerate(sample_ids):
            mask_num = mask_length[index]
            mask = self.layer_mask(mask_num, sample_id=id, mask=mask, sentence_index=index)

        return self._mask(mask, reference_embedding, decoder_input, reference)

    def layer_mask(self, mask_num, sample_id, mask, sentence_index):
        """
        随机mask多层, 随机扰动多层
        """
        dependency_layers = self.dependency_tree.get_one_sentence(sample_id, self.training)

        num = 0
        layers = list(range(len(dependency_layers)))
        random.shuffle(layers)
        for layer in layers:
            if num >= mask_num:
                break
            for token_index in dependency_layers[layer]:
                if num >= mask_num:
                    break
                mask[sentence_index][token_index] = True
                num += 1

        return mask

    def _mask(self, mask, reference_embedding, decoder_input, reference):
        non_mask = ~mask
        full_mask = torch.cat((non_mask.unsqueeze(-1), mask.unsqueeze(-1)), dim=-1)

        # 处理token
        predict_unk = reference.new_empty(reference.shape, requires_grad=False).fill_(
            self.tgt_dict.unk())
        full_output_tokens = torch.cat((predict_unk.unsqueeze(-1), reference.unsqueeze(-1)), dim=-1)
        output_tokens = (full_output_tokens * full_mask).sum(-1).long()

        # 处理embedding
        full_embedding = torch.cat((decoder_input.unsqueeze(-1), reference_embedding.unsqueeze(-1)), dim=-1)
        output_emebdding = (full_embedding * full_mask.unsqueeze(-2)).sum(-1)

        return output_tokens, output_emebdding

    def get_random_mask_output(self, mask_length=None, reference=None, samples=None, encoder_out=None,
                               decoder_input=None, reference_embedding=None):
        reference_mask = get_base_mask(reference)

        target_score = reference.clone().float().uniform_()
        target_score.masked_fill_(~reference_mask, 2.0)

        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < mask_length[:, None].long()
        mask = target_cutoff.scatter(1, target_rank, target_cutoff)  # [b, l]

        return self._mask(mask, reference_embedding, decoder_input, reference)

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
            **kwargs
        )

        # 计算hamming距离

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

        return losses


@register_model_architecture('GLAT', 'GLAT_iwslt16_de_en')
def glat_iwslt16_de_en(args):
    nat_iwslt16_de_en(args)
