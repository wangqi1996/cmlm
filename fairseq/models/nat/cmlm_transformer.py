# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements:
Ghazvininejad, Marjan, et al.
"Constant-time machine translation with conditional masked language models."
arXiv preprint arXiv:1904.09324 (2019).
"""

import numpy as np

from fairseq.data.data_utils import collate_tokens_list
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import NATransformerModel, torch
from fairseq.util2 import new_arange, get_reference_mask, merge_mask, get_dependency_mask, load_dependency_tree, \
    get_base_mask, set_all_token, set_diff_tokens, set_step_value


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
            (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


@register_model("cmlm_transformer")
class CMLMNATransformerModel(NATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        self.freq_weight = getattr(args, "freq_weight", False)
        self.use_dependency_weight = getattr(args, "use_dependency_weight", False)

        scale = 8
        if self.freq_weight:
            print("use freq weight!!!!!!!!!!")
            weights = np.power(np.array(self.tgt_dict.count), 1 / 2)

            min_weight = min(weights)
            max_weight = max(weights)
            weights -= min_weight
            diff = max_weight - min_weight

            weights = weights / diff * scale

            weights = scale - weights + 1
            weights[:self.tgt_dict.nspecial] = 1

            self.weights = torch.tensor(weights, requires_grad=False).float().cuda()
        else:
            self.weights = None
            # self.weight_freq = torch.Tensor(get_probability(self.weights), requires_grad=False).float().cuda()
        if self.use_dependency_weight:
            print("use dependency weight!!!")
            self.train_dependency_tree = load_dependency_tree(
                dependency_tree_path="/home/data_ti5_c/wangdq/data/iwslt14.de-en.distill/data-bin/dependency_tree.train.log",
                convert=True, add_one=True, scale=scale)
            self.valid_dependency_tree = load_dependency_tree(
                dependency_tree_path="/home/data_ti5_c/wangdq/data/iwslt14.de-en.distill/data-bin/dependency_tree.valid.log",
                convert=True, add_one=True, scale=scale)
        else:
            self.train_dependency_tree = None
            self.valid_dependency_tree = None

    def get_dependency_weigth_tensor(self, ids, tree):

        select = [tree[id] + [1] for id in ids]  # 这个要check一下
        select_array = collate_tokens_list(select, pad_idx=0)  # pad的地方损失为0
        return select_array

    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        # decoding latent模型,使用这个做后验采样操作
        prev_target_embedding = None
        if "prev_target_embedding" in kwargs:
            prev_target_embedding = kwargs['prev_target_embedding']

        word_ins_out, _decoder_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            return_decoder_output=True,
            prev_target_embedding=prev_target_embedding)

        hidden_state = _decoder_out['return_decoder_output']
        word_ins_mask = prev_output_tokens.eq(self.unk)

        freq_weights = None
        dependency_weights = None
        if self.freq_weight:
            freq_weights = self.weights

        if self.use_dependency_weight:
            sentence_index = kwargs["sample"]["id"].cpu().tolist()

            if self.training:
                weights = self.get_dependency_weigth_tensor(sentence_index, self.train_dependency_tree)
            else:
                weights = self.get_dependency_weigth_tensor(sentence_index, self.valid_dependency_tree)
            dependency_weights = torch.from_numpy(weights).float().to(src_tokens.device)

        loss = {
            "word_ins": {
                "out": word_ins_out, "tgt": tgt_tokens,
                "mask": word_ins_mask, "ls": self.args.label_smoothing,
                "nll_loss": True,
                "weights": {
                    "dependency_weights": dependency_weights,
                    "freq_weights": freq_weights
                }
            },
            "length": {
                "out": length_out, "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor
            },
            "train_need": {
                "encoder_out": encoder_out,
                "sample_ids": kwargs["sample"]["id"].cpu().tolist(),
                "hidden_state": hidden_state
            }
        }

        return loss

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):

        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history
        references = kwargs["references"]

        # random mask references  为了统计不同的reference mask概率时的准确率
        # output_tokens = _random_mask(references, noise_probability=0.8)
        #
        # execute the decoder
        output_masks = output_tokens.eq(self.unk)
        logits, hidden_state = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            return_decoder_output=True
        )
        _scores, _tokens = logits.max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        # compute_accuracy  为了统计不同的reference mask概率时的准确率
        # if step == max_step:
        if kwargs.get('accuracy', False) is True:
            output_masks = get_base_mask(references)
            need_predict_token = output_masks.float().sum().item()
            correct_mask = output_masks & (output_tokens == references)
            correct_predict_token = (output_masks & (output_tokens == references)).float().sum().item()
            set_diff_tokens(correct_predict_token)
            set_all_token(need_predict_token)

            token_index = output_tokens[correct_mask].cpu().tolist()
            for i in token_index:
                set_step_value(0, i)

        if history is not None:
            history.append(output_tokens.clone())

        # 直接用self.args.XXX 会有问题，会reload训练时的参数
        # skeptical decoding (depend on the maximum decoding steps.)
        use_reference_mask = kwargs['use_reference_mask']
        # use_reference_probability = kwargs['reference_probability']

        use_baseline_mask = kwargs['use_baseline_mask']

        # use_block_mask_size = kwargs['use_block_mask_size']
        # use_block_mask_method = kwargs['use_block_mask_method']

        dependency_tree = kwargs['dependency_tree']
        dependency_modify_root = kwargs['dependency_modify_root']
        use_mask_predictor = kwargs['use_mask_predictor']

        random_modify_token = kwargs['random_modify_token']

        # use_comma_mask = kwargs['use_comma_mask']
        # comma_mask_list = kwargs['comma_mask_list']
        if (step + 1) < max_step:
            skeptical_mask = None

            if use_reference_mask:
                references_mask = get_reference_mask(references, output_tokens)
                # print(references_mask.cpu().numpy())
                skeptical_mask = merge_mask(skeptical_mask, references_mask)

            if dependency_tree is not None and len(dependency_tree) > 0:
                dependency_mask = get_dependency_mask(output_tokens, dependency_tree, step,
                                                      modify=dependency_modify_root, reference=references,
                                                      random_modify_token=random_modify_token)

                # 确保该step之前生成的mask不要被覆盖掉
                dependency_mask = merge_mask(dependency_mask, output_masks)
                skeptical_mask = merge_mask(skeptical_mask, dependency_mask)

            # comma_mask 暂时不用

            # use_reference_probability 暂时不用

            if True or use_baseline_mask:
                # 就是根据长度进行mask的
                baseline_mask = _skeptical_unmasking(
                    output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
                )

                # 暂时在这里比较一下 reference_mask和baseline_mask的区别
                # rev_references_mask = (~references_mask) & pad_mask  # 本次生成中和reference一致的token  (去除pad的字符
                # rev_baseline_mask = (~baseline_mask) & pad_mask  # 本次生成中 baseline认为正确的 (去除pad的字符
                # common = (rev_baseline_mask & rev_references_mask).sum().item()
                # ref = rev_references_mask.sum().item()
                # base = rev_baseline_mask.sum().item()
                # all_token = pad_mask.sum().item()
                # set_diff_tokens(common)  # 46887
                # set_all_token(all_token)  # 49770
                # set_value1(base)  # 147987
                # set_value2(ref)
                # set_value3(repeat_num)

                # 统计reference mask=True但是baseline mask=False的情况下, 重复token的比例。
                # diff_mask = references_mask & rev_baseline_mask  # 不是特殊字符，baseline认为不需要mask 但是和reference不一致
                # batch_size, _ = output_masks.shape
                # zero_pad = output_tokens.new_full((batch_size, 1), 0)
                #
                # _left = torch.cat((zero_pad, output_tokens[:, :-1]), dim=-1)
                # left_common = _left == output_tokens
                #
                # _right = torch.cat((output_tokens[:, 1:], zero_pad), dim=-1)
                # right_common = _right == output_tokens
                #
                # common = left_common | right_common  # 重复的token
                # repeat = (common & diff_mask)  # 预测错误的且重复的token
                # repeat_num = repeat.sum().item()

                # # 修改baseline mask
                # baseline_mask = baseline_mask | repeat  # 33.42 ->34.57

                skeptical_mask = merge_mask(skeptical_mask, baseline_mask)
            # use block mask 暂时不用

            # 暂时排他使用
            if use_mask_predictor:
                mask_predict = self.decoder.mask_predictor(encoder_out=encoder_out,
                                                           hidden_state=hidden_state,
                                                           decoder_input=output_tokens,
                                                           logits=logits,
                                                           nmt_model=self,
                                                           predict_word=None,
                                                           normalize=False,
                                                           output_score=output_scores)

                if self.args.mask_predictor_method in ['linear', 'mix-attention']:
                    predictor_mask = mask_predict.max(-1)[1].bool()  # True表示需要mask的
                else:
                    predictor_mask = _skeptical_unmasking(mask_predict, output_tokens.ne(self.pad),
                                                          1 - (step + 1) / max_step * 0.5)

                skeptical_mask = merge_mask(skeptical_mask, predictor_mask)

                # compute准确率和召回率 去掉特殊字符 skeptical_mask=True表示需要mask，在这里反转一下。
                # 准确率 = real_correct / predict_correct
                # 召回率 = real_correct / all_correct
                # base_mask = get_base_mask(output_tokens)  # False表示为特殊字符,True表示普通字符
                # reverse_mask = (~skeptical_mask) & base_mask  # predictor认为该token预测正确，且该token不是特殊字符
                # predict_correct = reverse_mask.sum().item()
                #
                # reference_mask = (output_tokens == references) & base_mask  # 和reference一致且不是特殊字符
                # all_correct = reference_mask.sum().item()
                #
                # real_correct = (reference_mask & reverse_mask).sum().item()  # 预测的和真实的一样
                #
                # all_token = output_masks.sum().item()  # 所有的需要预测的token
                # set_diff_tokens(real_correct)
                # set_value1(predict_correct)
                # set_all_token(all_correct)

            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)
            if history is not None:
                history.append(output_tokens.clone())

        decoder_out = decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history
        )

        if kwargs.get('return_hidden_state', False):
            return decoder_out, hidden_state
        else:
            return decoder_out


@register_model_architecture("cmlm_transformer", "cmlm_transformer")
def cmlm_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.ngram_predictor = getattr(args, "ngram_predictor", 1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture("cmlm_transformer", "cmlm_transformer_wmt_en_de")
def cmlm_wmt_en_de(args):
    cmlm_base_architecture(args)
