# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import random

from fairseq.dep import load_dependency_tree
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import NATransformerModel, base_architecture


@register_model('DEP_GLAT_NOJOINT')
class DEP_GLAT_NOOINT(NATransformerModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        self.dep_model = getattr(args, "dep_model", "random")

        # 加载dependency tree
        if self.dep_model in ['layer']:
            self.train_dependency_tree = load_dependency_tree(
                "/home/data_ti5_c/wangdq/data/distill/iwslt14_de_en/data-bin/dependency.train.log",
                add_one=True)
            self.valid_dependency_tree = load_dependency_tree(
                "/home/data_ti5_c/wangdq/data/distill/iwslt14_de_en/data-bin/dependency.valid.log",
                add_one=True)
        else:
            self.train_dependency_tree, self.valid_dependency_tree = None, None

        self.glat_warmup_steps = getattr(args, "glat_warmup_steps", -1)
        self.full_mask_steps = getattr(args, "full_mask_steps", -1)
        if self.full_mask_steps == -1:
            self.full_mask_steps = self.args.max_update

    def random_mask(self, length, mask_num, reference_embedding, prev_target_token, decoder_input, sentence_index, unk):
        token_indexs = list(range(1, length - 1))
        random.shuffle(token_indexs)

        for token_index in token_indexs[:int(mask_num)]:
            reference_embedding[sentence_index][token_index] = decoder_input[sentence_index][
                token_index]
            prev_target_token[sentence_index][token_index] = unk

        return reference_embedding, prev_target_token

    def layer_mask(self, sample_id, mask_num, reference_embedding, prev_target_token, decoder_input,
                   sentence_index, unk):
        """
        随机mask多层, 随机扰动多层
        """
        if self.training:
            dependency_layers = self.train_dependency_tree[sample_id]
        else:
            dependency_layers = self.valid_dependency_tree[sample_id]

        num = 0
        layers = list(range(len(dependency_layers)))
        random.shuffle(layers)
        for layer in layers:
            if num > mask_num:
                break
            for token_index in dependency_layers[layer]:
                if num > mask_num:
                    break
                reference_embedding[sentence_index][token_index] = decoder_input[sentence_index][
                    token_index]
                prev_target_token[sentence_index][token_index] = unk
                num += 1

        return reference_embedding, prev_target_token

    def dep_model_mask(self, dep_model, **kwargs):

        length = kwargs.get('length', None)
        mask_num = kwargs.get('mask_num', None)
        reference_embedding = kwargs.get('reference_embedding', None)
        prev_target_token = kwargs.get('prev_target_token', None)
        decoder_input = kwargs.get('decoder_input', None)
        sentence_index = kwargs.get('sentence_index', None)
        unk = kwargs.get('unk', None)
        sample_id = kwargs.get("sample_id", None)

        if dep_model == "random":
            reference_embedding, prev_target_token = self.random_mask(length=length, mask_num=mask_num,
                                                                      reference_embedding=reference_embedding,
                                                                      prev_target_token=prev_target_token,
                                                                      decoder_input=decoder_input,
                                                                      sentence_index=sentence_index, unk=unk)
        elif dep_model == "layer":
            reference_embedding, prev_target_token = self.layer_mask(sample_id=sample_id,
                                                                     mask_num=mask_num,
                                                                     reference_embedding=reference_embedding,
                                                                     prev_target_token=prev_target_token,
                                                                     decoder_input=decoder_input,
                                                                     sentence_index=sentence_index, unk=unk)
        else:
            assert False, "未定义的dep_model类型"
        return reference_embedding, prev_target_token

    def get_mask_token(self, hidden_state, samples, decoder_input, update_nums, dep_model="random"):
        references = samples['target']
        sample_ids = samples['id'].cpu().tolist()

        if update_nums < self.glat_warmup_steps:
            dep_model = "random"

        # 3. 要mask的token数目，需要仔细调节这个东西
        if self.training:
            if self.full_mask_steps != -1:
                ratio = self.get_ratio2(update_nums)
            else:
                ratio = self.get_ratio(update_nums)
        else:
            ratio = 0.5
        # mask_nums = (correct_predict_head * ratio).cpu().tolist()

        # 4. 从底层开始对每个句子进行mask
        reference_embedding, _ = self.decoder.forward_embedding(references)

        prev_target_token = copy.deepcopy(references)
        unk = self.tgt_dict.unk()

        # random mask
        reference_length = (references != self.pad).sum(-1).cpu().tolist()

        mask_nums = ((references != self.pad).sum(-1) * ratio).cpu().tolist()

        for sentence_index, sample_id in enumerate(sample_ids):
            mask_num = mask_nums[sentence_index]
            length = reference_length[sentence_index]

            kwargs = {
                'length': length,
                'mask_num': mask_num,
                'reference_embedding': reference_embedding,
                'prev_target_token': prev_target_token,
                'decoder_input': decoder_input,
                'sentence_index': sentence_index,
                'unk': unk,
                'sample_id': sample_id
            }
            reference_embedding, prev_target_token = self.dep_model_mask(dep_model, **kwargs)

        return reference_embedding, prev_target_token, ratio

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)

        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        samples = kwargs['sample']

        if self.training:
            update_nums = kwargs['update_nums']
        else:
            update_nums = 0

        decoder_input, decoder_input_mask = self.decoder.get_copy_embedding(encoder_out, prev_output_tokens)
        # 带有位置embedding
        prev_target_embedding, prev_output_tokens, mask_ratio = self.get_mask_token(None, samples, decoder_input,
                                                                                    update_nums=update_nums,
                                                                                    dep_model=self.dep_model)

        # decoding
        word_ins_out, _decoder_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            prev_target_embedding=prev_target_embedding,
            return_decoder_output=True,
            step=1)

        head_mask = prev_output_tokens.eq(self.unk)

        loss = {
            "word_ins": {
                "out": word_ins_out, "tgt": tgt_tokens,
                "mask": head_mask, "ls": self.args.label_smoothing,
                "nll_loss": True
            },
            "length": {
                "out": length_out, "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor
            },
            "train_need": {
                "print": {
                    "mask_ratio": mask_ratio
                }
            }

        }

        return loss

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        decoder_out, hidden_state = super().forward_decoder(decoder_out, encoder_out, decoding_format=decoding_format,
                                                            return_decoder_output=True, **kwargs)

        return decoder_out

    def get_ratio2(self, update_nums):
        max_updates = self.full_mask_steps
        ratio = min(1, max(update_nums / max_updates, 0.1))
        return ratio

    def get_ratio(self, update_nums):
        max_updates = self.args.max_update
        ratio = update_nums / max_updates
        return ratio * 0.99 + 0.1


@register_model_architecture("DEP_GLAT_NOJOINT", "DEP_GLAT_NOJOINT")
def GLAT(args):
    base_architecture(args)
