# encoding=utf-8

from torch import nn

from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import TransformerEncoderLayer, LayerDropModuleList, LayerNorm, TransformerDecoderLayer


class EncoderLayer(nn.Module):

    def __init__(self, args, num_layers=1, pad=0, **unused):
        super().__init__()
        self.padding_idx = pad
        self.layers = LayerDropModuleList(p=args.encoder_layerdrop,
                                          modules=[TransformerEncoderLayer(args) for _ in range(num_layers)]
                                          )
        self.num_layers = num_layers

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    def forward(self, src_embed, src_token):
        # B * T * C -> T * B * C
        x = src_embed.transpose(0, 1)
        encoder_padding_mask = src_token.eq(self.padding_idx)

        for layer in self.layers:
            x = layer(x, encoder_padding_mask=encoder_padding_mask)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=src_embed,  # B x T x C
            encoder_states=None,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )


class DecoderLayer(nn.Module):

    def __init__(self, args, num_layers=1, pad=0, no_encoder_attn=True, **unused):

        super().__init__()
        self.padding_idx = pad
        self.args = args

        self.layers = LayerDropModuleList(p=args.decoder_layerdrop,
                                          modules=[TransformerDecoderLayer(args, no_encoder_attn=no_encoder_attn) for _
                                                   in
                                                   range(num_layers)])

        self.num_layers = num_layers

        if args.decoder_normalize_before and not getattr(
                args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(args.decoder_embed_dim)
        else:
            self.layer_norm = None

    def forward(self, prev_embed, encoder_out, target_token):
        # B * T * C -> T * B * C
        x = prev_embed.transpose(0, 1)
        decoder_padding_mask = target_token.eq(self.padding_idx)

        for layer in self.layers:
            x, _, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        x = x.transpose(0, 1)

        return x


class PriorTransformNN(nn.Module):
    def __init__(self, args, pad):
        super().__init__()

        self.encoder = EncoderLayer(args, num_layers=2, pad=pad)
        self.decoder = DecoderLayer(args, num_layers=2, pad=pad)
        self.linear = nn.Linear(args.decoder_embed_dim, 2)

        self.args = args

        self.padding_idx = pad

    def forward(self, src_embed, src_token, trg_prev_embed, trg_token, **unused):
        encoder_out = self.encoder(src_embed, src_token)

        decoder_out = self.decoder(prev_embed=trg_prev_embed, encoder_out=encoder_out, target_token=trg_token)

        out = self.linear(decoder_out)
        return out


class PosteriorTransformNN(nn.Module):

    def __init__(self, args, pad):
        super().__init__()

        self.encoder = EncoderLayer(args, num_layers=1, pad=pad)
        self.target_encoder = EncoderLayer(args, num_layers=1, pad=pad)

        self.decoder = DecoderLayer(args, num_layers=1, pad=pad, no_encoder_attn=False)
        self.linear = nn.Linear(args.decoder_embed_dim, 2)
        self.args = args

        self.padding_idx = pad

    def forward(self, src_embed, src_token, trg_prev_embed, trg_token, reference_embed, reference_token, **unused):
        """ trg_mask: pad mask
        暂时把encoder和decoder拼接起来计算
        """
        # encoder_out = self.encoder(src_embed, src_token)
        target_out = self.target_encoder(reference_embed, reference_token)
        new_out = EncoderOut(
            encoder_out=reference_embed,  # time dim
            encoder_padding_mask=target_out.encoder_padding_mask,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None

        )

        decoder_out = self.decoder(prev_embed=trg_prev_embed, encoder_out=new_out, target_token=trg_token)

        out = self.linear(decoder_out)

        return out


class PriorDecoderNN(nn.Module):
    def __init__(self, args, pad):
        super().__init__()

        self.decoder = DecoderLayer(args, num_layers=1, pad=pad)
        self.linear = nn.Linear(args.decoder_embed_dim, 2)

        self.args = args

        self.padding_idx = pad

    def forward(self, encoder_out, trg_prev_embed, trg_token, **unused):
        decoder_out = self.decoder(prev_embed=trg_prev_embed, encoder_out=encoder_out, target_token=trg_token)

        out = self.linear(decoder_out)
        return out


class PosteriorDecoderNN(nn.Module):
    def __init__(self, args, pad):
        super().__init__()

        self.target_encoder = EncoderLayer(args, num_layers=1, pad=pad)

        self.decoder = DecoderLayer(args, num_layers=1, pad=pad, no_encoder_attn=False)
        self.linear = nn.Linear(args.decoder_embed_dim, 2)
        self.args = args

        self.padding_idx = pad

    def forward(self, trg_prev_embed, trg_token, reference_embed, reference_token, **unused):
        """ trg_mask: pad mask
        暂时把encoder和decoder拼接起来计算
        """
        target_out = self.target_encoder(reference_embed, reference_token)
        # new_out = EncoderOut(
        #     encoder_out=torch.cat([encoder_out.encoder_out, target_out.encoder_out], dim=0),  # time dim
        #     encoder_padding_mask=torch.cat([encoder_out.encoder_padding_mask, target_out.encoder_padding_mask], dim=-1),
        #     encoder_embedding=None,
        #     encoder_states=None,
        #     src_tokens=None,
        #     src_lengths=None
        #
        # )
        # reference_mask = reference_token.eq(self.padding_idx)
        # new_out = EncoderOut(
        #     encoder_out=reference_embed,
        #     encoder_padding_mask=reference_mask,
        #     encoder_embedding=None,
        #     encoder_states=None,
        #     src_tokens=None,
        #     src_lengths=None
        # )

        # 这个还是非自回归地预测目标
        decoder_out = self.decoder(prev_embed=trg_prev_embed, encoder_out=target_out, target_token=trg_token)

        out = self.linear(decoder_out)

        return out


class LSTMLayer(nn.Module):
    def __init__(self, args, num_layers=1, dropout=0.2, pad=0, bidirectional=True):
        super().__init__()

        self.padding_idx = pad
        self.num_layers = num_layers

        self.lstm = nn.GRU(input_size=args.encoder_embed_dim, hidden_size=300, num_layers=num_layers,
                           bidirectional=bidirectional, dropout=dropout, batch_first=False)

        self.bidirectional = bidirectional

    def combine_bidir(self, outs, bsz: int):
        out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
        return out

    def forward(self, src_embed, src_token, h_0=None, process_hidden=True):
        """
        融合encoder有两种方案，1是h_0, 2是使用attention。这个是h_0的方案。
        :param src_embed: seq_len * batch * dim
        :param src_token:
        :param h_0:
        :return:
        """
        src_embed = src_embed.transpose(0, 1)
        batch_size = src_embed.size(1)

        x_length = src_token.ne(self.padding_idx).sum(-1)

        packed_x = nn.utils.rnn.pack_padded_sequence(src_embed, x_length, enforce_sorted=False)
        lstm_output, final_hidden = self.lstm(packed_x, hx=h_0)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, padding_value=self.padding_idx)

        if self.bidirectional and process_hidden:
            final_hidden = self.combine_bidir(final_hidden, batch_size)
        return lstm_output, final_hidden


class PriorLSTMNN(nn.Module):
    def __init__(self, args, pad):
        super().__init__()

        self.encoder = LSTMLayer(args, num_layers=1, pad=pad)
        self.decoder = LSTMLayer(args, num_layers=1, pad=pad, bidirectional=False)
        self.linear = nn.Linear(300, 2)

        self.args = args

        self.padding_idx = pad

    def forward(self, src_embed, src_token, trg_prev_embed, trg_token, **unused):
        encoder_out, hidden_state = self.encoder(src_embed, src_token, process_hidden=True)

        h_0 = hidden_state.mean(-2)
        decoder_out, _ = self.decoder(src_embed=trg_prev_embed, src_token=trg_token, h_0=h_0)

        out = self.linear(decoder_out)
        return out.transpose(0, 1)


class PosteriorLSTMNN(nn.Module):

    def __init__(self, args, pad):
        super().__init__()

        self.encoder = LSTMLayer(args, num_layers=1, pad=pad)
        self.reference_encoder = LSTMLayer(args, num_layers=1, pad=pad)
        self.decoder = LSTMLayer(args, num_layers=1, pad=pad, bidirectional=False)
        self.linear = nn.Linear(300, 2)
        self.args = args

        self.padding_idx = pad

    def forward(self, src_embed, src_token, trg_prev_embed, trg_token, reference_embed, reference_token, **unused):
        """ trg_mask: pad mask
        暂时把encoder和decoder拼接起来计算
        """
        # source_hidden: [num_layers, batch, bidirection, dim]
        # encoder_out, source_hidden = self.encoder(src_embed, src_token, process_hidden=False)

        reference_out, reference_hidden = self.reference_encoder(reference_embed, reference_token, h_0=None)

        decoder_out, target_hidden = self.decoder(src_embed=trg_prev_embed, src_token=trg_token,
                                                  h_0=reference_hidden.mean(-2))

        out = self.linear(decoder_out)

        return out.transpose(0, 1)


REGISTER_PRIOR_ARCH = {
    "transformer": PriorTransformNN,
    "lstm": PriorLSTMNN,
    "decoder": PriorDecoderNN
}

REGISTER_POSEERIOR_ARCH = {
    "transformer": PosteriorTransformNN,
    "lstm": PosteriorLSTMNN,
    "decoder": PosteriorDecoderNN,
}


def get_prior(arch, args, pad):
    return REGISTER_PRIOR_ARCH[arch](args, pad=pad)


def get_posterior(arch, args, pad):
    return REGISTER_POSEERIOR_ARCH[arch](args, pad=pad)
