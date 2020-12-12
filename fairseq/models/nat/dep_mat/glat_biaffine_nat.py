# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.dep import RelativeDepMat
from fairseq.models import register_model_architecture, register_model
from fairseq.models.nat import BlockedDecoderLayer, NATDecoder, init_bert_params, \
    NAT, nat_wmt_en_de
from fairseq.models.nat.dep_mat.glat_biaffine import DepHeadClassifier
from fairseq.modules.dep_attention import DepRelativeMultiheadAttention


class DEPRelativeGLATDecoderLayer(BlockedDecoderLayer):

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False, layer_id=0, **kwargs):

        if layer_id == 0:
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
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        dep_file = getattr(self.args, "dep_file", "iwslt16")
        self.relative_dep_mat = RelativeDepMat(valid_subset=self.args.valid_subset, args=args, dep_file=dep_file)

        self.dep_classifier = DepHeadClassifier(relative_dep_mat=self.relative_dep_mat, args=args, dep_file=dep_file)

        self.dep_input_ref = getattr(self.args, "dep_input_ref", False)
        if self.dep_input_ref:
            print("使用ref embedding做为分类器的输入: ", self.dep_input_ref)

        self.glat_training = getattr(args, "glat_training", False)

    def build_decoder_layer(self, args, no_encoder_attn=False, rel_keys=None, rel_vals=None, layer_id=0, **kwargs):
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

        dep_loss, dep_mat = self.forward_classifier(hidden_state=x, position_embedding=position,
                                                    target_tokens=prev_output_tokens, encoder_out=encoder_out, **unused)
        if dep_mat is not None:
            unused['dependency_mat'] = dep_mat

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

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        other = {"attn": attn, "inner_states": inner_states, "embedding": embedding, "position_embedding": position,
                 "dep_classifier_loss": dep_loss}

        return x, other

    def get_special_input(self, samples, generate=False, **kwargs):
        sample_ids = samples['id']
        target_token = samples['prev_target']

        if not generate:
            dependency_mat = self.relative_dep_mat.get_dependency_mat(sample_ids, target_token, training=self.training)
            return {"dependency_mat": dependency_mat}
        else:
            return {}

    def forward_classifier(self, hidden_state, sample, **kwargs):
        ref_embedding = None
        if self.dep_input_ref or self.glat_training:
            ref_embedding, _, _ = self.forward_embedding(sample['target'])
            ref_embedding = ref_embedding.transpose(0, 1)
            hidden_state = ref_embedding

        if kwargs.get("generate", False):

            dependency_mat = self.dep_classifier.inference(sample=sample, hidden_state=hidden_state, **kwargs)
            return {}, dependency_mat
        else:

            loss = self.dep_classifier.inference_accuracy(sample=sample, hidden_state=hidden_state,
                                                          ref_embedding=ref_embedding, **kwargs)

            return loss, None


SuperClass, model_name = NAT, "dep_relative_classifier"


@register_model(model_name)
class DEPRelativeNAT(SuperClass):

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = DEPRelativeDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)

        return decoder

    @staticmethod
    def add_args(parser):
        SuperClass.add_args(parser)
        DepHeadClassifier.add_args(parser)

        parser.add_argument('--dep-file', type=str, default="iwslt16")  # wmt16
        parser.add_argument('--dep-input-ref', action="store_true")

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

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=True, **kwargs)

        word_ins_out, other = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            inner=True,
            **kwargs
        )

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

        dep_classifier_loss = other['dep_classifier_loss']
        losses.update(dep_classifier_loss)

        return losses


@register_model_architecture(model_name, model_name + '_wmt')
def dep_relative_glat_wmt(args):
    nat_wmt_en_de(args)
