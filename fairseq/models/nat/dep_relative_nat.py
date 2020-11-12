# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.dep import DepChildTree, DepHeadTree, get_dependency_mat, get_coarse_dependency_mat
from fairseq.models import register_model_architecture, register_model
from fairseq.models.nat import GLAT, BlockedDecoderLayer, DepRelativeMultiheadAttention, NATDecoder, init_bert_params, \
    build_relative_embeddings
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


@register_model('dep_relative_glat')
class DEPRelativeGLAT(GLAT):

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = DEPRelativeDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        self.child_tree = DepChildTree(valid_subset=self.args.valid_subset)
        self.head_tree = DepHeadTree(valid_subset=self.args.valid_subset)

        self.dep_mat_grain = getattr(args, "dep_mat_grain", "fine")
        print(self.dep_mat_grain)

    def add_args(parser):
        GLAT.add_args(parser)

        parser.add_argument('--relative-direction', default=True)
        parser.add_argument('--dep-mat-grain', type=str, default="coarse", choices=['fine', 'coarse'])
        parser.add_argument('--relative-layers', type=str, default="0")

    def get_dependency_mat(self, sample_ids, target_token):
        if self.head_tree is not None and self.child_tree is not None:
            if self.dep_mat_grain == "fine":
                dependency_mat = get_dependency_mat(self.head_tree, self.child_tree, sample_ids, self.training,
                                                    target_token)
            elif self.dep_mat_grain == "coarse":
                dependency_mat = get_coarse_dependency_mat(self.head_tree, self.child_tree, sample_ids, self.training,
                                                           target_token, contain_eos=True)
        else:
            dependency_mat = None

        return dependency_mat

    def inference_special_input(self, special_input, not_terminated):
        keys = ['dependency_mat']
        for k in keys:
            v = special_input.get(k, None)
            if v is not None:
                v = v[not_terminated]
                special_input[k] = v
        return special_input

    def get_special_input(self, samples):
        dependency_mat = self.get_dependency_mat(samples["id"].cpu().tolist(),
                                                 samples['prev_target'],
                                                 )
        return {"dependency_mat": dependency_mat}


@register_model_architecture('dep_relative_glat', 'dep_relative_glat_iwslt16_de_en')
def dep_relative_glat_iwslt16_de_en(args):
    nat_iwslt16_de_en(args)
