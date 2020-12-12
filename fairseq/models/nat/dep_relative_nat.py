# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import nn

from fairseq.dep import RelativeDepMat
from fairseq.models import register_model_architecture, register_model
from fairseq.models.nat import BlockedDecoderLayer, NATDecoder, init_bert_params, \
    build_relative_embeddings, DepCoarseClassifier, build_dep_classifier, NAT, nat_wmt_en_de
from fairseq.modules.dep_attention import DepRelativeMultiheadAttention
from .nat_base import nat_iwslt16_de_en


class DEPRelativeGLATDecoderLayer(BlockedDecoderLayer):
    def __init__(self, args, **kwargs):

        relative_layers = getattr(args, "relative_layers", "all")

        self.relative_layers = relative_layers
        self.layer_ids = []
        if relative_layers != "all":
            self.layer_ids = [int(i) for i in relative_layers.split(',')]

        super().__init__(args, **kwargs)

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False, layer_id=0, **kwargs):

        if self.relative_layers == "all" or layer_id in self.layer_ids:
            return DepRelativeMultiheadAttention(
                embed_dim=embed_dim,
                num_heads=args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size
            )
        else:
            return super().build_self_attention(embed_dim, args, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
                                                layer_id=layer_id, **kwargs)


"""
关于使用oracle矩阵还是使用predict矩阵的问题:
1. 模型的状态分为三种: train、eval、inference; train和eval总是同种处理流程; 
2. inference负责调选模型，use_oracle_dep_generate控制inference使用oracel vs generate
3. 关于train和eval
    3.1 若传了use_oracle_dep，则在tune等操作之前使用oracle矩阵; 否则是predict矩阵
    3.2 tune等操作之后，按照tune的意愿来操作
"""


class DEPRelativeDecoder(NATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        # 依存矩阵相关
        # use_two_class = getattr(self.args, "use_two_class", False)
        # print("三分类转为二分类，建议True: ", use_two_class)
        use_two_class = True

        relative_layers = getattr(args, "relative_layers", "all")

        self.relative_layers = relative_layers
        self.layer_ids = []
        if relative_layers != "all":
            self.layer_ids = [int(i) for i in relative_layers.split(',')]

        if getattr(self, "relative_dep_mat", None) is None and self.relative_layers != -2:
            dep_file = getattr(self.args, "dep_file", "iwslt16")
            self.relative_dep_mat = RelativeDepMat(valid_subset=self.args.valid_subset, use_two_class=use_two_class,
                                                   args=args, dep_file=dep_file)

        # 分类器相关
        predict_dep_model = getattr(args, "predict_dep_model", "none")
        self.dep_classifier = build_dep_classifier(predict_dep_model, args=args, relative_dep_mat=self.relative_dep_mat,
                                                   use_two_class=use_two_class, dep_file=dep_file)
        self.predict_dep_relative_layer = getattr(args, "predict_dep_relative_layer", -2)
        # print("使用哪层预测依存矩阵，建议100: ", self.predict_dep_relative_layer)

        self.dependency_classifier_loss = getattr(self.args, "dependency_classifier_loss", False)

        # 训练和测试策略
        self.use_oracle_dep = getattr(self.args, "use_oracle_dep", False)
        self.use_oracle_dep_generate = getattr(self.args, "use_oracle_dep_generate", False)
        print("是否使用oracle的矩阵训练nmt模型： ", self.use_oracle_dep)
        print("是否使用oracle的矩阵infernece: ", self.use_oracle_dep_generate)
        self.dep_warmup_steps = getattr(self.args, "dep_warmup_steps", -1)
        self.dep_train_method = getattr(self.args, "dep_train_method", "none")
        print("使用何种方式消除train和infernec的gap: ", self.dep_train_method)

        self.random_dep_mat = False
        # self.random_dep_mat = getattr(self.args, "random_dep_mat", False)  # 测试时使用
        # print("使用随机的random矩阵: ", self.random_dep_mat)

        self.dep_input_ref = getattr(self.args, "dep_input_ref", False)
        if self.dep_input_ref:
            print("使用ref embedding做为分类器的输入: ", self.dep_input_ref)

        self.random_perturb_mat = False
        # self.random_perturb_mat = getattr(self.args, "random_perturb_mat", 0.0)
        # if self.random_perturb_mat > 0.0:
        #     print("随机扰动oracle矩阵: ", self.random_perturb_mat)

        self.joint_encoder = getattr(self.args, "joint_encoder", False)  # 使用联合的模式
        if self.joint_encoder:
            self.add_encoder_hidden = getattr(self.args, "add_encoder_hidden", "none")
            print("joint encoder呀~ , add: ", self.add_encoder_hidden)
            if self.add_encoder_hidden == "gate":
                self.fusion_ffn = nn.Sequential(
                    nn.Linear(args.decoder_embed_dim * 2, args.decoder_embed_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(args.decoder_embed_dim * 4, args.decoder_embed_dim),
                    nn.Dropout(args.dropout)
                )

    def build_decoder_layer(self, args, no_encoder_attn=False, rel_keys=None, rel_vals=None, layer_id=0, **kwargs):
        if getattr(args, "self_attn_cls", "abs") != "abs":
            rel_keys = rel_keys if rel_keys is not None else build_relative_embeddings(args)
            rel_vals = rel_vals if rel_vals is not None else build_relative_embeddings(args)
        return DEPRelativeGLATDecoderLayer(args, no_encoder_attn=no_encoder_attn, relative_keys=rel_keys,
                                           relative_vals=rel_vals,
                                           layer_id=layer_id, **kwargs)

    def apply_init_bias(self):
        self.dep_classifier.apply_init_bias()

    def fusion_dep_hidden(self, dep_hidden, x):
        if self.add_encoder_hidden == "none":
            return dep_hidden.transpose(0, 1)
        if self.add_encoder_hidden == "add":
            return dep_hidden.transpose(0, 1) + x

        if self.add_encoder_hidden == "gate":
            x = x.transpose(0, 1)
            input = torch.cat((dep_hidden, x), dim=-1)  # [b, l, d]
            g = self.fusion_ffn(input).sigmoid()  # [b, l, d*2]
            output = g * x + dep_hidden * (1 - g)
            return output.transpose(0, 1)

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out=None,
            early_exit=None,
            embedding_copy=False,
            prev_target_embedding=None,
            **unused
    ):

        x, decoder_padding_mask, position = self.forward_decoder_inputs(prev_output_tokens, encoder_out=encoder_out,
                                                                        prev_target_embedding=prev_target_embedding)

        embedding = x.clone()
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder_input=100
        # 分类器预测relative dep
        dep_classifier_loss = None
        post_process = self.forward_classifier(layer_id=100, hidden_state=x, position_embedding=position,
                                               target_tokens=prev_output_tokens, encoder_out=encoder_out, **unused)
        if post_process is not None:
            unused.update(post_process)
            dep_classifier_loss = post_process.get('dep_classifier_loss', None)
            if self.joint_encoder:
                dep_hidden = post_process['dep_hidden']
                x = self.fusion_dep_hidden(dep_hidden, x)
        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if not self.layerwise_attn else encoder_out.encoder_states[i],
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
                prev_output_tokens=prev_output_tokens,
                **unused
            )
            inner_states.append(x)

            post_process = self.forward_classifier(layer_id=i, hidden_state=x, position_embedding=position,
                                                   target_tokens=prev_output_tokens, encoder_out=encoder_out, **unused)
            if post_process is not None:
                unused.update(post_process)
                dep_classifier_loss = post_process.get('dep_classifier_loss', None)
                if self.joint_encoder:
                    dep_hidden = post_process['dep_hidden']
                    x = self.fusion_dep_hidden(dep_hidden, x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        other = {"attn": attn, "inner_states": inner_states, "embedding": embedding, "position_embedding": position,
                 }
        if dep_classifier_loss is not None:
            other["dep_classifier_loss"] = dep_classifier_loss

        return x, other

    def compute_oracle_dep(self, update_nums, generate=False):
        if self.relative_layers == -2:
            return False

        if generate:
            return self.use_oracle_dep_generate

        if self.use_oracle_dep is False or self.dep_train_method == "none" or update_nums < self.dep_warmup_steps:
            return self.use_oracle_dep

        if self.dep_train_method == "tune":
            return False

        if self.dep_train_method == "schedule":
            import numpy as np
            p = np.random.rand()
            if p < update_nums / 300000:  # 这个ratio可以再改变
                return False
            else:
                return True

    def compute_predict_dep(self, **kwargs):
        if self.relative_layers == -2:
            return False

        if "dependency_mat" in kwargs:
            return False
        else:
            return True

    def get_special_input(self, samples, generate=False, **kwargs):
        sample_ids = samples['id']
        target_token = samples['prev_target']

        if self.random_dep_mat:
            dependency_mat = self.relative_dep_mat.get_random_mat(sample_ids, target_token, training=self.training)
            return {"dependency_mat": dependency_mat}

        use_oracle = self.compute_oracle_dep(update_nums=kwargs['update_num'], generate=generate)
        if use_oracle:
            dependency_mat = self.relative_dep_mat.get_dependency_mat(sample_ids, target_token, training=self.training,
                                                                      perturb=self.random_perturb_mat)
            return {"dependency_mat": dependency_mat}
        else:
            return {}

    def forward_classifier(self, layer_id, hidden_state, sample, **kwargs):

        if layer_id == self.predict_dep_relative_layer and self.dep_classifier is not None:
            ref_embedding, _, _ = self.forward_embedding(sample['target'])
            ref_embedding = ref_embedding.transpose(0, 1)
            if self.dep_input_ref:
                hidden_state = ref_embedding

            generate_mat = self.compute_predict_dep(**kwargs)

            if kwargs.get("generate", False):
                if not generate_mat:
                    return {}

                dependency_mat, dep_hidden = self.dep_classifier.inference(sample=sample, hidden_state=hidden_state,
                                                                           ref_embedding=ref_embedding,
                                                                           perturb=self.random_perturb_mat,
                                                                           **kwargs)
                return {'dependency_mat': dependency_mat, "dep_hidden": dep_hidden}
            else:
                dependency_classifier_loss = self.dependency_classifier_loss
                loss, all, correct, dependency_mat, dep_hidden = self.dep_classifier.inference_accuracy(sample=sample,
                                                                                                        hidden_state=hidden_state,
                                                                                                        compute_loss=dependency_classifier_loss,
                                                                                                        result_mat=generate_mat,
                                                                                                        generate=False,
                                                                                                        ref_embedding=ref_embedding,
                                                                                                        **kwargs)
                torch.cuda.empty_cache()
                if kwargs.get("eval_accuracy", False):
                    loss.setdefault('train_need', {})
                    loss['train_need'].update({
                        "print": {
                            "all_predict_head": all,
                            "correct_predict_head": correct
                        }})

                result = {"dep_classifier_loss": loss, "dep_hidden": dep_hidden}

                if generate_mat:
                    result['dependency_mat'] = dependency_mat
                return result

        return None


SuperClass, model_name = NAT, "dep_relative_nat"


# 在nat base上没有做头
# SuperClass, model_name = GLAT, "dep_relative_glat"


@register_model(model_name)
class DEPRelativeNAT(SuperClass):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # 联合训练损失如何计算
        self.dependency_classifier_loss = getattr(self.args, "dependency_classifier_loss", False)
        self.nmt_loss = getattr(self.args, "nmt_loss", False)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = DEPRelativeDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)

        if args.classifier_mutual_method == "bias":
            decoder.apply_init_bias()
        return decoder

    @staticmethod
    def add_args(parser):
        SuperClass.add_args(parser)
        DepCoarseClassifier.add_args(parser)

        parser.add_argument('--relative-direction', default=True)

        # 依存矩阵相关
        parser.add_argument('--relative-layers', type=str, default="0")
        parser.add_argument('--use-dependency-mat-type', default="parent")  # grandparent
        parser.add_argument('--use-two-class', action="store_true")  # 将相同类别归于相关类别
        parser.add_argument('--random-dep-mat', action="store_true")
        parser.add_argument('--random-perturb-mat', type=float, default=0.0)
        parser.add_argument('--dep-file', type=str, default="iwslt16")  # wmt16

        # 分类器相关
        parser.add_argument('--predict-dep-model', type=str, default="none",
                            choices=['none', 'head', 'relative', "distill_head"],
                            help="the model used to predict the dependency matrix")
        parser.add_argument('--predict-dep-relative-layer', type=int,
                            default=-2)
        parser.add_argument('--dep-input-ref', action="store_true")
        parser.add_argument('--joint-encoder', action="store_true")
        parser.add_argument('--add-encoder-hidden', type=str, default="none")  # gate add none

        # 联合训练相关
        parser.add_argument('--dependency-classifier-loss', action="store_true",
                            default="calculate the dependency classifier losses")
        parser.add_argument('--nmt-loss', action="store_true", default="Calculate nmt model losses")

        # 训练时和valid时 / inference 时 使用oracle信息
        parser.add_argument('--use-oracle-dep', action="store_true")
        parser.add_argument('--use-oracle-dep-generate', action="store_true")

        parser.add_argument('--dep-warmup-steps', type=int, default=-1)
        parser.add_argument('--dep-train-method', type=str, default="none")  # tune、schedule

    def inference_special_input(self, special_input, not_terminated):
        if special_input is None:
            return special_input

        keys = ['dependency_mat']
        for k in keys:
            v = special_input.get(k, None)
            if v is not None:
                v = v[not_terminated]
                special_input[k] = v
        return special_input

    def get_special_input(self, samples, generate=False, **kwargs):
        return self.decoder.get_special_input(samples, generate=generate, **kwargs)

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

        # decoding 不计算分类器的准确率和loss，只计算矩阵
        word_ins_out, other = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            inner=True,
            **kwargs
        )

        losses = {}
        if self.nmt_loss:
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

        if self.dependency_classifier_loss:
            dep_classifier_loss = other['dep_classifier_loss']
            losses.update(dep_classifier_loss)

        return losses

    # def forward(
    #         self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    # ):
    #     """
    #     GLAT: 两步解码
    #     1. 输入为没有梯度的decode一遍
    #     2. 计算hamming距离（和reference有多少token不一致
    #     3. 随机采样（其实是确定了mask的概率。
    #     4. hidden state和word embedding混合
    #     """
    #
    #     # encoding
    #     encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=True, **kwargs)
    #
    #     # decoding 不计算分类器的准确率和loss，只计算矩阵
    #     eval_accuracy = kwargs.get("eval_accuracy", False)
    #     kwargs['eval_accuracy'] = False
    #     word_ins_out, other = self.decoder(
    #         normalize=False,
    #         prev_output_tokens=prev_output_tokens,
    #         encoder_out=encoder_out,
    #         inner=True,
    #         **kwargs
    #     )
    #     word_ins_out.detach_()
    #     _score, predict = word_ins_out.max(-1)
    #     mask_num = self.get_mask_num(tgt_tokens, predict)
    #
    #     decoder_input = other['embedding']
    #     samples = kwargs['sample']
    #     output_token, output_embedding = self.get_mask_output(decoder_input=decoder_input, reference=tgt_tokens,
    #                                                           mask_length=mask_num, samples=samples, encoder_out=None)
    #
    #     # decoder
    #     kwargs['eval_accuracy'] = eval_accuracy
    #     word_ins_out, other = self.decoder(
    #         normalize=False,
    #         prev_output_tokens=output_token,
    #         encoder_out=encoder_out,
    #         inner=True,
    #         prev_target_embedding=output_embedding,
    #         **kwargs
    #     )
    #
    #     losses = {}
    #     if self.nmt_loss:
    #         losses = {
    #             "word_ins": {
    #                 "out": word_ins_out,
    #                 "tgt": tgt_tokens,
    #                 "mask": tgt_tokens.ne(self.pad),
    #                 "ls": self.args.label_smoothing,
    #                 "nll_loss": True
    #             }
    #         }
    #         # length prediction
    #         if self.decoder.length_loss_factor > 0:
    #             length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
    #             length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)
    #             losses["length"] = {
    #                 "out": length_out,
    #                 "tgt": length_tgt,
    #                 "factor": self.decoder.length_loss_factor
    #             }
    #
    #     if self.dependency_classifier_loss:
    #         dep_classifier_loss = other['dep_classifier_loss']
    #         losses.update(dep_classifier_loss)
    #
    #     return losses


@register_model_architecture(model_name, model_name + '_iwslt16_de_en')
def dep_relative_glat_iwslt16_de_en(args):
    nat_iwslt16_de_en(args)


@register_model_architecture(model_name, model_name + '_wmt')
def dep_relative_glat_wmt(args):
    nat_wmt_en_de(args)
