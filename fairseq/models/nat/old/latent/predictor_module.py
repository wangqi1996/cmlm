from torch import nn

from fairseq.modules import TransformerEncoderLayer, TransformerDecoderLayer


def mean_ds(x, dim=None):
    return (
        x.float().mean().type_as(x)
        if dim is None
        else x.float().mean(dim).type_as(x)
    )


class Perceptron(nn.Module):
    """
    1. 是否激活 通过是否有激活层来控制，最后一层都没有激活层
    """

    def __init__(self, input_dim, output_dim, activation=None, drouput=0.1, **unused):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

        self.activate = None
        if activation is not None:
            self.activate = activation()

        self.dropout = None
        if drouput > 0:
            self.dropout = nn.Dropout(p=drouput)

    def forward(self, x, dropout=True):

        # inference时关掉dropout开关
        if not self.training:
            dropout = False

        out = self.linear(x)

        if self.activate:
            out = self.activate(out)

        if dropout:
            out = self.dropout(out)
        return out


class LogisticModel(nn.Module):
    """ 两层感知机 """

    def __init__(self, args, activation=None, dropout=0.1, contain_normalize=False, **unused):
        """ 如果Logistic是模型的最后一层，contain_normalize=True; 否则，设置为False"""
        super().__init__()

        self.layers = nn.Sequential(
            Perceptron(args.encoder_embed_dim, int(args.encoder_embed_dim / 2), drouput=dropout,
                       activation=activation),
            Perceptron(int(args.encoder_embed_dim / 2), 1, drouput=dropout, activation=None)
        )

        self.activation = None
        if contain_normalize:
            self.activation = nn.Sigmoid()

    def forward(self, decoder_out=None, normalize=False, **unused):
        out = self.layers(decoder_out)

        if self.activation is not None and normalize:
            out = self.activation(out)

        return out.squeeze(-1)


"""
TODO: AT Decoder
"""


class DecoderLayer(nn.Module):

    def __init__(self, args, pad, num_layers, contain_normalize=True, **unused):
        super().__init__()

        self.padding_idx = pad
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(args, no_encoder_attn=False) for _ in range(num_layers)]
        )
        self.predictor = LogisticModel(args, activation=nn.Sigmoid, dropout=0.3, contain_normalize=contain_normalize,
                                       **unused)

    def forward(self, encoder_output, encoder_padding_mask, decoder_out, decoder_input, normalize=False, **unused):
        # NAT的decoder

        decoder_padding_mask = decoder_input.eq(self.padding_idx)

        batch_size, _ = encoder_padding_mask.shape

        b, _, _ = decoder_out.shape
        if b == batch_size:
            decoder_out = decoder_out.transpose(0, 1)

        b, _, _ = encoder_output.shape
        if b == batch_size:
            encoder_output = encoder_output.transpose(0, 1)

        input = decoder_out
        for layer in self.layers:
            input, _, _ = layer(
                input,
                encoder_output,
                encoder_padding_mask,
                incremental_state=None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask)

        out = self.predictor(decoder_out=input.transpose(0, 1), normalize=normalize)

        # 使用logistic做了normalize的处理啦
        return out


class EncoderLayer(nn.Module):

    def __init__(self, args, num_layers=1, contain_logistic=True, contain_normalize=True, pad=0, **unused):
        """ contain_logistic和contain_normalize其实是绑定的"""
        super().__init__()
        self.padding_idx = pad
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(num_layers)]
        )

        self.predictor = None
        if contain_logistic:
            self.predictor = LogisticModel(args, activation=nn.Sigmoid, dropout=0.3,
                                           contain_normalize=contain_normalize, **unused)

    def forward(self, input, mask=None, decoder_input=None, normalize=False):
        """
        1. 用于encoder: mask=encoder, decoder_input=None
        2. 用于decoder: mask=None, decoder_input=decoder.input
        """
        if mask is None:
            mask = decoder_input.eq(self.padding_idx)

        for layer in self.layers:
            input = layer(input, encoder_padding_mask=mask)

        if self.predictor is not None:
            out = self.predictor(input.transpose(0, 1), normalize=normalize)

            return out

        return input


class LSTMLayer(nn.Module):
    def __init__(self, args, num_layers=1, activation=nn.Sigmoid, dropout=0.2, pad=0, contain_logistic=True):
        super().__init__()

        self.padding_idx = pad
        self.lstm = nn.GRU(input_size=args.encoder_embed_dim, hidden_size=300, num_layers=num_layers,
                           bidirectional=True, dropout=dropout, batch_first=True)
        self.predictor = None
        if contain_logistic:
            self.predictor = nn.Sequential(
                Perceptron(input_dim=600, output_dim=300, activation=activation, drouput=0.2),
                Perceptron(input_dim=300, output_dim=1, activation=None, drouput=0.2)  # 最后一层不用激活
            )

        self.activate = activation()

    def forward(self, decoder_out=None, normalize=False, decoder_input=None, h_0=None, **unused):
        x_length = decoder_input.ne(self.padding_idx).sum(-1)
        packed_x = nn.utils.rnn.pack_padded_sequence(decoder_out, x_length, batch_first=True, enforce_sorted=False)

        lstm_output, _ = self.lstm(packed_x, h_0)

        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, padding_value=self.padding_idx, batch_first=True)

        if self.activate:
            out = self.predictor(lstm_output).squeeze(-1)
        else:
            out = lstm_output
        return self.activate(out) if normalize else out


class EncoderDecoder(nn.Module):
    def __init__(self, args, encoder_layers=3, decoder_layers=3, pad=0):
        # 三层block
        super().__init__()
        self.encoder = EncoderLayer(args, num_layers=encoder_layers, contain_logistic=False)

        self.decoder = DecoderLayer(args, num_layers=decoder_layers, pad=pad)

    def forward(self, encoder_out, decoder_out, decoder_input, normalize=False, **unused):
        encoder_output = self.encoder(encoder_out.encoder_out, encoder_out.encoder_padding_mask, normalize=False)
        decoder_output = self.decoder(encoder_output, encoder_out.encoder_padding_mask, decoder_out,
                                      decoder_input, normalize=normalize)

        return decoder_output


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_class = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, outputs, targets):
        # target: 1表示需要mask
        # output：得分越高表示不需要mask
        targets = (~targets).float()
        return self.loss_class(outputs, targets)
