# encoding=utf-8


import torch
import torch.nn.functional as F
from torch import nn

from fairseq.models.nat.old.latent.predictor_module import _Predictor
from fairseq.modules import MultiheadAttention


class MarginPredict(_Predictor):
    def __init__(self, args, pad, **unused):
        super().__init__(args, pad)

        self.predictor = LogisticModel(args, activation=nn.Sigmoid, dropout=0.2, **unused)
        self.loss_class = nn.MarginRankingLoss(reduction="none")

    def loss_func(self, outputs, targets, masks):
        # outputs 是对应的分数,得分越高表示不需要mask 和targets相反
        # target: 1表示需要mask，0表示和reference一致，1表示和reference不一致
        # mask False=reference，True的表示本轮预测的。只关心mask=True的
        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
            return loss

        low_index = (targets & masks).nonzero(as_tuple=True)  # 提取和reference不一致的，对应的mask肯定=False
        low_feature = outputs[low_index]
        # 现在1表示不需要mask, 理应对应的得分要高
        targets = (~targets) & (masks)  # 提取在本轮预测的且和reference一致的。
        high_index = targets.nonzero(as_tuple=True)
        high_feature = outputs[high_index]
        # 随机选择一些pair margin_loss: (x1,x2,y) y=1表示x1的结果大于x2
        pairs_x1 = []
        pairs_x2 = []

        # 整合low score
        batch, _ = targets.shape
        low_feature_list = [[] for _ in range(batch)]
        for row, feature in zip(low_index[0].cpu().tolist(), low_feature):
            low_feature_list[row].append(feature)

        for row, feature in zip(high_index[0].cpu().tolist(), high_feature):
            low = low_feature_list[row]
            if len(low) == 0:
                continue
            for l in low:
                pairs_x1.append(feature)
                pairs_x2.append(l)
        pairs_x1 = torch.stack(pairs_x1, dim=-1)
        pairs_x2 = torch.stack(pairs_x2, dim=-1)
        b, = pairs_x1.shape
        y = pairs_x1.new_full((b, 1), fill_value=1)
        losses = self.loss_class(pairs_x1, pairs_x2, y.squeeze(-1))

        nll_loss = mean_ds(losses)
        loss = nll_loss

        return loss


class LinearPredict(_Predictor):
    def __init__(self, args, pad=0):
        super().__init__(super().__init__(args, pad))
        self.predictor = nn.Linear(args.encoder_embed_dim, 2)

    def loss_func(self, outputs, targets):
        logits = F.log_softmax(outputs, dim=-1)
        return F.nll_loss(logits, targets.to(logits.device).long(), reduction='none')

    def forward(self, decoder_out=None, decoder_input=None, normalize=False, encoder_out=None, **unused):
        predict_input = decoder_out
        predict = self.predictor(predict_input)
        return F.log_softmax(predict, -1) if normalize else predict


class MixAttentionPredict(_Predictor):
    def __init__(self, args, pad=0, **unused):
        super().__init__(args, pad)

        self.mix_attention = MultiheadAttention(
            args.encoder_embed_dim,
            args.decoder_attention_heads,
            kdim=args.encoder_embed_dim,
            vdim=args.encoder_embed_dim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True
        )

        self.mix_attention.reset_parameters()
        self.predictor = nn.Linear(args.encoder_embed_dim * 2, 2)

    def loss_func(self, outputs, targets):
        logits = F.log_softmax(outputs, dim=-1)
        return F.nll_loss(logits, targets.to(logits.device).long(), reduction='none')

    def forward(self, decoder_out=None, decoder_input=None, normalize=False, encoder_out=None, **unused):
        # mask
        encoder_padding_mask = encoder_out.encoder_padding_mask
        target_mask = decoder_input.eq(self.padding_idx)
        key_padding_mask = torch.cat((encoder_padding_mask, target_mask), dim=1)

        # key,value,query
        decoder_out = decoder_out.transpose(0, 1)
        encoder_hidden_state = encoder_out.encoder_out
        input = torch.cat((encoder_hidden_state, decoder_out), dim=0)

        attn_out, attn_weigth = self.mix_attention(query=decoder_out, key=input, value=input,
                                                   key_padding_mask=key_padding_mask,
                                                   incremental_state=None, static_kv=True)

        predict_input = torch.cat((attn_out, decoder_out), dim=-1).transpose(0, 1)
        predict = self.predictor(predict_input)
        return F.log_softmax(predict, -1) if normalize else predict
