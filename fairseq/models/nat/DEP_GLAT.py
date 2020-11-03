# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import random

from fairseq.dep import load_dependency_tree, load_dependency_head_tree
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import NATransformerModel, base_architecture, BiaffineAttentionDependency
from fairseq.util2 import set_value1, set_value2, get_base_mask


@register_model('DEP_GLAT')
class DEP_GLAT(NATransformerModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        self.dep_model = getattr(args, "dep_model", "random")

        self.update_two_decoding = getattr(args, "update_two_decoding", False)
        print("--update-two-decoder: ", self.update_two_decoding)
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

        if self.dep_model in ['path', 'child']:
            self.train_dependency_tree_head = load_dependency_head_tree(
                dependency_tree_path="/home/data_ti5_c/wangdq/data/distill/iwslt14_de_en/data-bin/dependency_head.train.log",
                add_one=True)
            self.valid_dependency_tree_head = load_dependency_head_tree(
                dependency_tree_path="/home/data_ti5_c/wangdq/data/distill/iwslt14_de_en/data-bin/dependency_head.valid.log",
                add_one=True)
        else:
            self.train_dependency_tree_head, self.valid_dependency_tree_head = None, None

        if self.dep_model in ['parent']:
            # 有哪些孩子节点
            self.train_dependency_tree_child = load_dependency_tree(
                "/home/data_ti5_c/wangdq/data/distill/iwslt14_de_en/data-bin/dependency_child.train.log",
                add_one=True
            )
            self.valid_dependency_tree_child = load_dependency_tree(
                "/home/data_ti5_c/wangdq/data/distill/iwslt14_de_en/data-bin/dependency_child.valid.log"
            )
        else:
            self.train_dependency_tree_child, self.valid_dependency_tree_child = None, None
        #
        self.dependency_model = BiaffineAttentionDependency(args, input_dim=self.args.encoder_embed_dim,
                                                            contain_lstm=False, padding_idx=self.tgt_dict.pad())

        # froze_nmt_model
        if self.args.froze_nmt_model:
            self.froze_nmt_model()

    def froze_nmt_model(self):
        print("froze nmt encoder and decoder")

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.decoder.parameters():
            param.requires_grad = False

    def random_mask(self, length, mask_num, reference_embedding, prev_target_token, decoder_input, sentence_index, unk):
        token_indexs = random.sample(range(1, length - 1), min(int(mask_num), length - 2))  # 处理特殊token

        for token_index in token_indexs:
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

    def path_mask(self, sample_id, mask_num, reference_embedding, prev_target_token, decoder_input,
                  sentence_index, unk, length):
        """
        随机mask多层, 随机扰动多层
        """
        if self.training:
            dependency_heads = self.train_dependency_tree_head[sample_id]
        else:
            dependency_heads = self.valid_dependency_tree_head[sample_id]

        num = 0

        token_indexs = list(range(1, length - 1))
        random.shuffle(token_indexs)

        for index in range(len(token_indexs)):
            token_index = token_indexs[index]
            if num > mask_num:
                break
            if prev_target_token[sentence_index][token_index] == unk:
                continue
            while token_index > 0 and token_index < length and prev_target_token[sentence_index][
                token_index] != unk and num <= mask_num:
                reference_embedding[sentence_index][token_index] = decoder_input[sentence_index][
                    token_index]
                prev_target_token[sentence_index][token_index] = unk
                num += 1
                token_index = dependency_heads[token_index - 1]  # pad token

        return reference_embedding, prev_target_token

    def parent_mask(self, sample_id, mask_num, reference_embedding, prev_target_token, decoder_input,
                    sentence_index, unk, length):
        """
        孩子节点被mask的时候，保证父节点没有被mask住
        """
        if self.training:
            dependency_heads = self.train_dependency_tree_child[sample_id]
        else:
            dependency_heads = self.valid_dependency_tree_child[sample_id]

        num = 0

        token_indexs = list(range(1, length - 1))
        random.shuffle(token_indexs)

        can_mask = [1 for _ in range(length)]

        for index in range(len(token_indexs)):
            token_index = token_indexs[index]
            if num > mask_num:
                break
            if can_mask[token_index] == 0:
                continue
            reference_embedding[sentence_index][token_index] = decoder_input[sentence_index][
                token_index]
            prev_target_token[sentence_index][token_index] = unk
            num += 1

            child_indexs = dependency_heads[token_index - 1]
            for child_index in child_indexs:
                can_mask[child_index] = 0

        return reference_embedding, prev_target_token

    def child_mask(self, sample_id, mask_num, reference_embedding, prev_target_token, decoder_input,
                   sentence_index, unk, length):
        """
        孩子节点被mask的时候，保证父节点没有被mask住
        """
        if self.training:
            dependency_heads = self.train_dependency_tree_child[sample_id]
        else:
            dependency_heads = self.valid_dependency_tree_head[sample_id]

        num = 0

        token_indexs = list(range(1, length - 1))
        random.shuffle(token_indexs)

        can_mask = [1 for _ in range(length)]

        for index in range(len(token_indexs)):
            token_index = token_indexs[index]
            if num > mask_num:
                break
            if can_mask[token_index] == 0:
                continue
            reference_embedding[sentence_index][token_index] = decoder_input[sentence_index][
                token_index]
            prev_target_token[sentence_index][token_index] = unk
            num += 1
            parent_index = dependency_heads[token_index - 1]
            can_mask[parent_index] = 0

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
        elif dep_model == "path":
            reference_embedding, prev_target_token = self.path_mask(sample_id=sample_id, length=length,
                                                                    mask_num=mask_num,
                                                                    reference_embedding=reference_embedding,
                                                                    prev_target_token=prev_target_token,
                                                                    decoder_input=decoder_input,
                                                                    sentence_index=sentence_index, unk=unk)
        elif dep_model == "child":
            reference_embedding, prev_target_token = self.child_mask(sample_id=sample_id, length=length,
                                                                     mask_num=mask_num,
                                                                     reference_embedding=reference_embedding,
                                                                     prev_target_token=prev_target_token,
                                                                     decoder_input=decoder_input,
                                                                     sentence_index=sentence_index, unk=unk)
        elif dep_model == "parent":
            reference_embedding, prev_target_token = self.parent_mask(sample_id=sample_id, length=length,
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

        # 1. predict head
        # head_dep_predict = self.dependency_model(hidden_state, references)
        # head_dep_predict.detach_()
        #
        # # 2. compute accuracy
        # head_mask = references != self.pad
        # head_dep_reference = self.dependency_model.get_dep_reference(sample_ids)
        #
        # correct_predict_head = self.dependency_model.compute_accuracy_nosum(head_dep_predict, head_mask,
        #                                                                     head_dep_reference)  # 预测错误的token数目

        # 3. 要mask的token数目，需要仔细调节这个东西
        if self.training:
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
        # tree = dependency_tree[sentence_index]
        # mask_num = mask_nums[sentence_index]
        # layer_nums = len(tree)
        # num = 0
        # for layer_index in range(layer_nums - 1, -1, -1):
        #     layer = tree[layer_index]
        #     if num < mask_num:
        #         for token_index in layer:
        #             if num < mask_num:
        #                 reference_embedding[sentence_index][token_index] = decoder_input[sentence_index][
        #                     token_index]
        #                 prev_target_token[sentence_index][token_index] = unk
        #                 num += 1
        #     else:
        #         break

        return reference_embedding, prev_target_token

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
        word_ins_out, _decoder_out = self.decoder(normalize=False,
                                                  prev_output_tokens=prev_output_tokens,
                                                  encoder_out=encoder_out,
                                                  step=0,
                                                  return_decoder_output=True,
                                                  return_input=True,
                                                  )
        samples = kwargs['sample']
        hidden_state = _decoder_out['return_decoder_output']
        decoder_input = _decoder_out['return_input']
        #
        head_mask = get_base_mask(tgt_tokens)

        hidden_state2 = None
        head_mask2 = None
        if not self.args.only_joint_training:

            if self.training:
                update_nums = kwargs['update_nums']
            else:
                update_nums = 0

            # decoder_input, decoder_input_mask = self.decoder.get_copy_embedding(encoder_out, prev_output_tokens)
            prev_target_embedding, prev_output_tokens = self.get_mask_token(hidden_state, samples, decoder_input,
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

            hidden_state2 = _decoder_out['return_decoder_output']
            head_mask2 = prev_output_tokens.eq(self.unk)

        loss = {
            "word_ins": {
                "out": word_ins_out, "tgt": tgt_tokens,
                "mask": prev_output_tokens.eq(self.unk), "ls": self.args.label_smoothing,
                "nll_loss": True
            },
            "length": {
                "out": length_out, "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor
            },

        }

        # compute the joint training loss
        if self.args.joint_training or self.args.only_joint_training:
            # 如果仅仅更新一次，肯定是更新第二次的decoder
            # 如果不是仅仅更新一次，则联合训练更新第一次的decoder，nmt训练更新第二次的decoder。
            if not self.update_two_decoding and hidden_state2 is not None:
                hidden_state.contiguous().detach_()
                hidden_state = hidden_state2
                head_mask = head_mask2

            head_dep_predict = self.dependency_model(hidden_state, samples['target'])
            head_dep_reference = self.dependency_model.get_dep_reference(samples['id'].cpu().tolist())
            dep_loss = {
                "dep_loss": {
                    "out": head_dep_predict,
                    "tgt": head_dep_reference,
                    "mask": head_mask,
                    "ls": 0.1
                }
            }
            loss.update(dep_loss)

        if kwargs.get("eval_accuracy", False):
            all_predict_head, correct_predict_head = self.dependency_model.compute_accuracy(head_dep_predict, head_mask,
                                                                                            head_dep_reference)
            loss.setdefault('train_need', {})
            loss['train_need'].update({
                "print": {
                    "all_predict_head": all_predict_head,
                    "correct_predict_head": correct_predict_head
                }})

        return loss

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        decoder_out, hidden_state = super().forward_decoder(decoder_out, encoder_out, decoding_format=decoding_format,
                                                            return_decoder_output=True, **kwargs)

        # only compute at step=i
        if kwargs.get('compute_dep_accuracy', False):

            step = decoder_out.step

            max_step = decoder_out.max_step
            if step != max_step - 1:
                return decoder_out

            references = kwargs['references']

            head_dep_predict = self.dependency_model(hidden_state, references)

            # 2. compute accuracy
            head_mask = references != self.pad
            head_dep_reference = kwargs['biaffine_tree']

            all_predict_head, correct_predict_head = self.dependency_model.compute_accuracy(head_dep_predict, head_mask,
                                                                                            head_dep_reference)

            set_value1(all_predict_head)
            set_value2(correct_predict_head)

        return decoder_out

    def get_ratio(self, update_nums):
        max_updates = self.args.max_update
        step = update_nums / max_updates
        return step * 0.99 + 0.1


@register_model_architecture("DEP_GLAT", "DEP_GLAT")
def GLAT(args):
    base_architecture(args)
