# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.dep import RelativeDepMat
from fairseq.models import register_model_architecture, register_model
from fairseq.models.nat import BlockedDecoderLayer, NATDecoder, init_bert_params, \
    build_relative_embeddings, DepCoarseClassifier, GLAT, build_dep_classifier
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
        use_two_class = getattr(self.args, "use_two_class", False)
        print("三分类转为二分类，建议True: ", use_two_class)

        if getattr(self, "relative_dep_mat", None) is None:
            self.relative_dep_mat = RelativeDepMat(valid_subset=self.args.valid_subset, use_two_class=use_two_class,
                                                   args=args)

        # 分类器相关
        predict_dep_model = getattr(args, "predict_dep_model", "none")
        self.dep_classifier = build_dep_classifier(predict_dep_model, args=args, relative_dep_mat=self.relative_dep_mat,
                                                   use_two_class=use_two_class)
        self.predict_dep_relative_layer = getattr(args, "predict_dep_relative_layer", -2)
        print("使用哪层预测依存矩阵，建议1: ", self.predict_dep_relative_layer)

        # 联合训练损失如何计算
        self.dependency_classifier_loss = getattr(self.args, "dependency_classifier_loss", False)
        self.nmt_loss = getattr(self.args, "nmt_loss", False)

        # 训练和测试策略
        self.use_oracle_dep = getattr(self.args, "use_oracle_dep", False)
        self.use_oracle_dep_generate = getattr(self.args, "use_oracle_dep_generate", False)
        print("是否使用oracle的矩阵训练nmt模型： ", self.use_oracle_dep)
        print("是否使用oracle的矩阵infernece: ", self.use_oracle_dep_generate)
        self.dep_warmup_steps = getattr(self.args, "dep_warmup_steps", -1)
        self.dep_train_method = getattr(self.args, "dep_train_method", "none")
        print("使用何种方式消除train和infernec的gap: ", self.dep_train_method)

    def build_decoder_layer(self, args, no_encoder_attn=False, rel_keys=None, rel_vals=None, layer_id=0, **kwargs):
        if getattr(args, "self_attn_cls", "abs") != "abs":
            rel_keys = rel_keys if rel_keys is not None else build_relative_embeddings(args)
            rel_vals = rel_vals if rel_vals is not None else build_relative_embeddings(args)
        return DEPRelativeGLATDecoderLayer(args, no_encoder_attn=no_encoder_attn, relative_keys=rel_keys,
                                           relative_vals=rel_vals,
                                           layer_id=layer_id, **kwargs)

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
                                               reference=prev_output_tokens, **unused)
        if post_process is not None:
            unused.update(post_process)
            dep_classifier_loss = post_process.get('dep_classifier_loss', None)

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
                                                   reference=prev_output_tokens, **unused)
            if post_process is not None:
                unused.update(post_process)
                dep_classifier_loss = post_process.get('dep_classifier_loss', None)

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
        if generate:
            return self.use_oracle_dep_generate

        if self.use_oracle_dep is False or self.dep_train_method == "none" or update_nums < self.dep_warmup_steps:
            return self.use_oracle_dep

        if self.dep_train_method == "tune":
            return False

        if self.dep_train_method == "schedule":
            import numpy as np
            p = np.random.rand()
            if p < update_nums / 200000:  # 这个ratio可以再改变
                return False
            else:
                return True

    def compute_predict_dep(self, **kwargs):
        if "dependency_mat" in kwargs:
            return False
        else:
            return True

    def get_special_input(self, samples, generate=False, **kwargs):
        use_oracle = self.compute_oracle_dep(update_nums=kwargs['update_num'], generate=generate)
        if use_oracle:
            sample_ids = samples['id']
            target_token = samples['prev_target']

            dependency_mat = self.relative_dep_mat.get_dependency_mat(sample_ids, target_token, training=self.training)
            return {"dependency_mat": dependency_mat}
        else:
            return {}

    def forward_classifier(self, layer_id, hidden_state, position_embedding, reference, sample, **kwargs):

        if layer_id == self.predict_dep_relative_layer and self.dep_classifier is not None:

            generate_mat = self.compute_predict_dep(update_nums=kwargs.get('update_num', -1), **kwargs)

            if kwargs.get("generate", False):
                if not generate_mat:
                    return {}

                dependency_mat = self.dep_classifier.inference(hidden_state, position_embedding, sample=sample,
                                                               reference=reference,
                                                               **kwargs)
                return {'dependency_mat': dependency_mat}
            else:
                eval_accuracy = kwargs.get('eval_accuracy', False)
                dependency_classifier_loss = self.dependency_classifier_loss and kwargs.get(
                    "dependency_classifier_loss", True)
                loss, all, correct, dependency_mat = self.dep_classifier.inference_accuracy(hidden_state,
                                                                                            position_embedding,
                                                                                            dependency_classifier_loss,
                                                                                            reference, sample,
                                                                                            eval_accuracy,
                                                                                            result_mat=generate_mat)
                loss = {"dep_classifier_loss": {"loss": loss}}

                if eval_accuracy:
                    loss.setdefault('train_need', {})
                    loss['train_need'].update({
                        "print": {
                            "all_predict_head": all,
                            "correct_predict_head": correct
                        }})

                result = {"dep_classifier_loss": loss}

                if generate_mat:
                    result['dependency_mat'] = dependency_mat
                return result

        return None


# SuperClass, model_name = NAT, "dep_relative_nat"


# 在nat base上没有做头
SuperClass, model_name = GLAT, "dep_relative_glat"


@register_model(model_name)
class DEPRelativeNAT(SuperClass):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = DEPRelativeDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def add_args(parser):
        SuperClass.add_args(parser)
        DepCoarseClassifier.add_args(parser)

        parser.add_argument('--relative-direction', default=True)

        # 依存矩阵相关
        parser.add_argument('--relative-layers', type=str, default="0")
        parser.add_argument('--use-dependency-mat-type', default="parent")  # grandparent
        parser.add_argument('--use-two-class', action="store_true")  # 将相同类别归于相关类别

        # 分类器相关
        parser.add_argument('--predict-dep-model', type="str", default="none", choices=['none', 'head', 'relative'],
                            help="the model used to predict the dependency matrix")
        parser.add_argument('--predict-dep-relative-layer', type=int,
                            default=-2)

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
            eval_accuracy=False,
            dependency_classifier_loss=False,
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


@register_model_architecture(model_name, model_name + '_iwslt16_de_en')
def dep_relative_glat_iwslt16_de_en(args):
    nat_iwslt16_de_en(args)
