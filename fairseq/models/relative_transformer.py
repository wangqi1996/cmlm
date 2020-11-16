from typing import Dict, List, Optional

from torch import Tensor

from fairseq.dep import DepChildTree, DepHeadTree
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder, TransformerModel
from fairseq.modules import MultiheadAttention
from fairseq.modules.dep_attention import DepRelativeMultiheadAttention
from fairseq.modules.relative_multihead_attention import RelativeMultiheadAttention
from fairseq.modules.transformer_layer import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)


class RelativeTransformerEncoderLayer(TransformerEncoderLayer):

    def build_self_attention(self, embed_dim, args):
        """
        Optionally use relative position representation.
        """
        if args.positional_embeddings in ['abl', 'dep+abl']:
            return MultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size)
        else:
            return RelativeMultiheadAttention(
                max_relative_position=args.max_relative_position,
                relative_direction=args.relative_direction,
                embed_dim=embed_dim,
                num_heads=args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size
            )


class RelativeTransformerDecoderLayer(TransformerDecoderLayer):

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        if args.positional_embeddings == 'abl':
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
        elif args.positional_embeddings == "dep+abl":
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


class RelativeTransformerEncoder(TransformerEncoder):

    def build_encoder_layer(self, args):
        return RelativeTransformerEncoderLayer(args)


class RelativeTransformerDecoder(TransformerDecoder):

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return RelativeTransformerDecoderLayer(args, no_encoder_attn)

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            **kwargs
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        # # embed length positions
        # length_positions = (
        #     self.embed_positions(
        #         prev_output_tokens, incremental_state=incremental_state
        #     )
        #     if self.embed_positions is not None
        #     else None
        # )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer), ),
                **kwargs
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}


@register_model("relative_transformer")
class RelativeTransformerModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super(RelativeTransformerModel, self).__init__(args, encoder, decoder)

        # dependency_treee
        self.head_tree, self.child_tree, = None, None
        if self.args.positional_embeddings == "dep+abl":
            self.child_tree = DepChildTree(self.args.valid_subset)
            self.head_tree = DepHeadTree(self.args.valid_subset)

        self.dep_mat_grain = getattr(args, "dep_mat_grain", "fine")
        print(self.dep_mat_grain)

    def get_special_input(self, samples, **kwargs):
        dependency_mat = self.get_dependency_mat(samples["id"].cpu().tolist(),
                                                 samples['net_input']['prev_output_tokens'])
        return {"dependency_mat": dependency_mat}

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--positional-embeddings', choices=['rel', 'abl', 'rel+abl', 'dep+abl'])
        parser.add_argument('--max-relative-position', type=int, default=16)
        parser.add_argument('--relative-direction', default=True)
        parser.add_argument('--dep-mat-grain', type=str, default="fine", choices=['fine', 'coarse'])  # fine coarse

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return RelativeTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return RelativeTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, 'no_cross_attention', False)
        )

    def forward(self,
                src_tokens,
                src_lengths,
                prev_output_tokens,
                return_all_hiddens: bool = True,
                features_only: bool = False,
                alignment_layer: Optional[int] = None,
                alignment_heads: Optional[int] = None,
                dependency_mat=None):
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            dependency_mat=dependency_mat
        )

        return x, extra


@register_model_architecture("relative_transformer", "relative_transformer")
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


@register_model_architecture("relative_transformer", "relative_transformer_iwslt_de_en")
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
