# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.dep import RelativeDepMat
from fairseq.models import register_model_architecture, register_model
from fairseq.models.nat import BlockedDecoderLayer, NATDecoder, init_bert_params, \
    build_relative_embeddings, DepCoarseClassifier, GLAT
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


class DEPRelativeDecoder(NATDecoder):
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
        post_process_function = unused.get('post_process_function', None)

        if post_process_function is not None:
            post_process = post_process_function(layer_id=100, hidden_state=x, position_embedding=position,
                                                 target_token=prev_output_tokens, **unused)
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

            if post_process_function is not None:
                post_process = post_process_function(layer_id=i, hidden_state=x, position_embedding=position,
                                                     target_token=prev_output_tokens, **unused)
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

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # 依存矩阵计算的相关参数
        use_two_class = getattr(self.args, "use_two_class", False)
        print("use_tow_class: ", use_two_class)
        use_dependency_mat_type = getattr(self.args, "use_dependency_mat_type", False)
        if getattr(self, "relative_dep_mat", None) is None:
            self.relative_dep_mat = RelativeDepMat(valid_subset=self.args.valid_subset,
                                                   use_dependency_mat_type=use_dependency_mat_type,
                                                   use_two_class=use_two_class)

        """
        分类器
        """
        # relative dep 分类器
        self.dep_classifier = None
        if getattr(args, "predict_dep_relative", False):
            print("use dependency classifier!!!")
            self.dep_classifier = DepCoarseClassifier(args, self.relative_dep_mat, use_two_class=use_two_class)

        # 使用哪层的hidden state算dep relative
        self.predict_dep_relative_layer = getattr(args, "predict_dep_relative_layer", -2)
        print("predict_dep_relative_layer", self.predict_dep_relative_layer)

        # 是否计算classifier的损失
        self.dependency_classifier_loss = getattr(self.args, "dependency_classifier_loss", False)
        self.nmt_loss = getattr(self.args, "nmt_loss", False)

        # 在训练时使用oracle的relative dependency mat
        self.use_oracle_dep = getattr(self.args, "use_oracle_dep", False)
        self.use_oracle_dep_generate = getattr(self.args, "use_oracle_dep_generate", False)
        print("use_oracle_dep： ", self.use_oracle_dep)
        print("use_oracle_dep_generate: ", self.use_oracle_dep_generate)

        # 固定住nmt模型
        if self.args.froze_nmt_model:
            self.froze_nmt_model()

    def add_args(parser):
        SuperClass.add_args(parser)
        DepCoarseClassifier.add_args(parser)

        parser.add_argument('--relative-direction', default=True)
        parser.add_argument('--relative-layers', type=str, default="0")  # 0

        # 使用分类器
        parser.add_argument('--predict-dep-relative', action="store_true")
        parser.add_argument('--predict-dep-relative-layer', type=int,
                            default=-2)  # relative-layers比这个搞一个 decoder_input=100
        parser.add_argument('--dependency-classifier-loss', action="store_true")
        parser.add_argument('--nmt-loss', action="store_true")

        # 方案1：
        parser.add_argument('--positive-class-factor', type=float, default=1.0)

        # 训练时和valid时 使用oracle信息
        parser.add_argument('--use-oracle-dep', action="store_true")

        # 计算bleu的inference阶段使用oracle信息，默认时分类器的预测
        parser.add_argument('--use-oracle-dep-generate', action="store_true")

        """
        依存矩阵
        """
        # 使用哪一个矩阵
        parser.add_argument('--use-dependency-mat-type', default="parent")  # grandparent
        # 三分类转二分类
        parser.add_argument('--use-two-class', action="store_true")

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
        sample_ids = samples['id']
        target_token = samples['prev_target']

        use_oracle = (not generate and self.use_oracle_dep) or (generate and self.use_oracle_dep_generate)
        if use_oracle:
            dependency_mat = self.relative_dep_mat.get_dependency_mat(sample_ids, target_token, training=self.training,
                                                                      contain_eos=True)
            return {"dependency_mat": dependency_mat}
        else:
            return {}

    def post_process_after_layer(self, layer_id, hidden_state, position_embedding, target_token, sample, **kwargs):

        if layer_id == self.predict_dep_relative_layer and self.dep_classifier is not None:

            generate_mat = kwargs.get("generate", False) and not self.use_oracle_dep_generate
            train_mat = not kwargs.get("generate") and not self.use_oracle_dep
            if generate_mat:
                dependency_mat = self.dep_classifier.inference(hidden_state, position_embedding, sample=sample,
                                                               **kwargs)
                return {'dependency_mat': dependency_mat}
            else:
                eval_accuracy = kwargs.get('eval_accuracy', False)
                dependency_classifier_loss = self.dependency_classifier_loss
                loss, all, correct, dependency_mat = self.dep_classifier.inference_accuracy(hidden_state,
                                                                                            position_embedding,
                                                                                            dependency_classifier_loss,
                                                                                            target_token, sample,
                                                                                            eval_accuracy,
                                                                                            result_mat=train_mat)
                loss = {"dep_classifier_loss": {"loss": loss}}

                if eval_accuracy:
                    loss.setdefault('train_need', {})
                    loss['train_need'].update({
                        "print": {
                            "all_predict_head": all,
                            "correct_predict_head": correct
                        }})

                result = {"dep_classifier_loss": loss}
                if train_mat:
                    result['dependency_mat'] = dependency_mat
                return result
        return None

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
            post_process_function=self.post_process_after_layer,
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
            post_process_function=self.post_process_after_layer,
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
