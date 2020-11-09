from fairseq.dep import DepChildTree, DepHeadTree, get_dependency_mat, get_coarse_dependency_mat
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.nat import NATransformerDecoder, NATransformerModel, init_bert_params
from fairseq.modules import MultiheadAttention, TransformerDecoderLayer
from fairseq.modules.dep_attention import DepRelativeMultiheadAttention
from fairseq.modules.relative_multihead_attention import RelativeMultiheadAttention


class RelativeNonTransformerDecoderLayer(TransformerDecoderLayer):

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False, layer_id=0):

        positional_embeddings = args.positional_embeddings
        relative_layers = getattr(args, "relative_layers", "all")
        if relative_layers != "all":
            layers = [int(i) for i in relative_layers.split(',')]
            if layer_id not in layers:
                positional_embeddings = 'abl'

        if positional_embeddings == 'abl':
            return MultiheadAttention(
                embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not getattr(args, "cross_self_attention", False),
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )
        elif positional_embeddings == "dep+abl":
            return DepRelativeMultiheadAttention(
                max_relative_position=args.max_relative_position,
                relative_direction=args.relative_direction,
                embed_dim=embed_dim,
                num_heads=args.encoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size
            )
        else:
            return RelativeMultiheadAttention(
                max_relative_position=args.max_relative_position,
                relative_direction=args.relative_direction,
                embed_dim=embed_dim,
                num_heads=args.encoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size
            )


class RelativeNonTransformerDecoder(NATransformerDecoder):

    def build_decoder_layer(self, args, no_encoder_attn=False, layer_id=0):
        return RelativeNonTransformerDecoderLayer(args, no_encoder_attn, layer_id=layer_id)

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out=None,
            early_exit=None,
            embedding_copy=False,
            **unused
    ):
        # embedding
        if embedding_copy:
            x, decoder_padding_mask = self.get_copy_embedding(encoder_out, prev_output_tokens)

        elif "prev_target_embedding" in unused and unused['prev_target_embedding'] is not None:
            x = unused['prev_target_embedding']
            decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        else:

            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

        input_embedding = x
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
                **unused
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}, input_embedding


@register_model("relative_non_transformer")
class RelativeNonTransformerModel(NATransformerModel):
    def __init__(self, args, encoder, decoder):
        super(RelativeNonTransformerModel, self).__init__(args, encoder, decoder)

        # dependency_treee
        self.head_tree, self.child_tree, = None, None
        if self.args.positional_embeddings == "dep+abl":
            self.child_tree = DepChildTree(self.args.valid_subset)
            self.head_tree = DepHeadTree(self.args.valid_subset)

        self.dep_mat_grain = getattr(args, "dep_mat_grain", "fine")
        print(self.dep_mat_grain)

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

    def get_special_input(self, samples):
        dependency_mat = self.get_dependency_mat(samples["id"].cpu().tolist(),
                                                 samples['prev_target'],
                                                 )
        return {"dependency_mat": dependency_mat}

    def inference_special_input(self, special_input, not_terminated):
        keys = ['dependency_mat']
        for k in keys:
            v = special_input.get(k, None)
            if v is not None:
                v = v[not_terminated]
                special_input[k] = v
        return special_input

    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)
        parser.add_argument('--positional-embeddings', choices=['rel', 'abl', 'rel+abl', 'dep+abl'])
        parser.add_argument('--max-relative-position', type=int, default=16)
        parser.add_argument('--relative-direction', default=True)
        parser.add_argument('--dep-mat-grain', type=str, default="fine", choices=['fine', 'coarse'])  # fine coarse
        parser.add_argument('--relative-layers', type=str, default="all")

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = RelativeNonTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, 'no_cross_attention', False),
        )
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder


@register_model_architecture("relative_non_transformer", "relative_non_transformer")
def base_architecture(args):
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
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)

    if 'abl' not in args.positional_embeddings:
        args.no_token_positional_embeddings = True


@register_model_architecture("relative_non_transformer", "relative_non_transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)

    base_architecture(args)
