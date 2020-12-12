# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.dep import RelativeDepMat
from fairseq.models import register_model_architecture, register_model
from fairseq.models.nat import BlockedDecoderLayer, NATDecoder, init_bert_params, \
    NAT, nat_wmt_en_de
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
        self.relative_dep_mat = RelativeDepMat(valid_subset=self.args.valid_subset, use_two_class=True,
                                               args=args, dep_file=dep_file)

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
                 }

        return x, other

    def get_special_input(self, samples, generate=False, **kwargs):
        sample_ids = samples['id']
        target_token = samples['prev_target']

        dependency_mat = self.relative_dep_mat.get_dependency_mat(sample_ids, target_token, training=self.training,
                                                                  perturb=0)
        return {"dependency_mat": dependency_mat}


SuperClass, model_name = NAT, "dep_relative_oracle"


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
        parser.add_argument('--dep-file', type=str, default="iwslt16")  # wmt16

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


@register_model_architecture(model_name, model_name + '_wmt')
def dep_relative_glat_wmt(args):
    nat_wmt_en_de(args)
