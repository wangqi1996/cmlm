# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.dep import DepHeadTree
from fairseq.models import BaseFairseqModel
from fairseq.models.lstm import LSTM
from fairseq.models.nat import BiaffineAttentionDependency, get_base_mask, \
    BlockedDecoderLayer, build_relative_embeddings, new_arange
from fairseq.util2 import set_key_value


def build_dep_classifier(model_name, **kwargs):
    if model_name == "none":
        return None

    if model_name == "head":
        return DepHeadClassifier(**kwargs)

    # if model_name == "relative":
    #     return DepCoarseClassifier(**kwargs)

    if model_name == "distill_head":
        return DHeadClassifier(**kwargs)


class DepEncoder(nn.Module):
    """ 对输入进行编码 使用自注意力 """

    def __init__(self, args, num_layers):
        super().__init__()

        rel_keys = build_relative_embeddings(args)
        rel_vals = build_relative_embeddings(args)

        self.layers = nn.ModuleList(
            [BlockedDecoderLayer(args, no_encoder_attn=False, relative_keys=rel_keys, relative_vals=rel_vals
                                 ) for _ in range(num_layers)]
        )

    def forward(self, encoder_out, hidden_state, decoder_padding_mask):
        for layer in self.layers:
            hidden_state, layer_attn, _ = layer(
                hidden_state,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
        return hidden_state.transpose(0, 1)


class LSTMEncoder(nn.Module):
    """ 对输入进行编码 使用LSTM """

    def __init__(self, args, num_layers):
        super().__init__()

        self.args = args
        self.lstm = LSTM(
            input_size=args.decoder_embed_dim,
            hidden_size=400,
            num_layers=1,
            dropout=0.,
            bidirectional=True,
            batch_first=False
        )

        args.decoder_embed_dim = 400 * 2
        args.decoder_ffn_embed_dim = 400 * 4
        rel_keys = build_relative_embeddings(args)
        rel_vals = build_relative_embeddings(args)
        self.attention = BlockedDecoderLayer(args, no_encoder_attn=False, relative_keys=rel_keys,
                                             relative_vals=rel_vals)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, encoder_out, hidden_state, decoder_padding_mask):
        length = (~decoder_padding_mask).long().sum(-1)
        unpaded_reference = nn.utils.rnn.pack_padded_sequence(hidden_state, lengths=length,
                                                              batch_first=False, enforce_sorted=False)
        hidden_state, _ = self.lstm(unpaded_reference)
        hidden_state, _ = nn.utils.rnn.pad_packed_sequence(hidden_state, batch_first=False, padding_value=1)

        hidden_state = self.dropout(hidden_state)
        hidden_state, layer_attn, _ = self.attention(
            hidden_state,
            encoder_out.encoder_out if encoder_out is not None else None,
            encoder_out.encoder_padding_mask if encoder_out is not None else None,
            self_attn_mask=None,
            self_attn_padding_mask=decoder_padding_mask,
        )
        return hidden_state.transpose(0, 1)


def cos_loss(hidden1, hidden2):
    r = F.cosine_similarity(hidden1, hidden2, dim=-1)
    return r.mean().exp()


class ImitationModule(nn.Module):
    def __init__(self, input_dim, ffn_dim, class_num=512, dropout=0.1):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, class_num),
            nn.Dropout(dropout)
        )

        self.action_embedding = nn.Embedding(embedding_dim=input_dim, num_embeddings=class_num, padding_idx=0)

        self.c = torch.zeros((class_num)).float().cuda()
        self.alpha = 0.9

    def forward(self, x, tokens):
        # forward
        logit = self.ffn(x)
        output = logit.softmax(-1)  # [b, l, class_num]
        output = output.masked_fill(tokens.eq(1).unsqueeze(-1), 0)

        # regularization
        length = tokens.ne(1).sum(-1)
        c = self.alpha * self.c + (1 - self.alpha) * (output.sum(-2).div(length.unsqueeze(-1)).mean(0))  # [class_num]
        norm_output = (output.pow(2)).div(c)
        sum_output = norm_output.sum(-1)
        # 处理0
        sum_output = sum_output.masked_fill(sum_output.eq(0), 1).unsqueeze(-1)
        norm_output = norm_output.div(sum_output)

        expect_embedding = output.matmul(self.action_embedding.weight)  # [b, l, d]

        mask = tokens.ne(1)
        loss = F.kl_div(output[mask].log(), norm_output[mask], reduction="none")
        l = loss.sum(-1).mean()
        return expect_embedding, l, output


class DepClassifier(BaseFairseqModel):

    def __init__(self, args, relative_dep_mat=None, use_two_class=False, **kwargs):
        super().__init__()
        self.args = args
        self.padding_idx = 0
        self.dep_loss_factor = getattr(self.args, "dep_loss_factor", 1.0)
        # if relative_dep_mat is None:
        #     self.relative_dep_mat = RelativeDepMat(valid_subset=self.args.valid_subset, use_two_class=use_two_class,
        #                                            **kwargs)
        # else:
        #     self.relative_dep_mat = relative_dep_mat

        self.relative_dep_mat = relative_dep_mat

        self.encoder_dep_input = getattr(self.args, "encoder_dep_input", "transformer")
        self.encoder = None
        if self.encoder_dep_input != "none":
            print("使用decoder blocks来编码分类器的input: ", self.encoder_dep_input)
            if self.encoder_dep_input == "lstm":
                self.encoder = LSTMEncoder(args, num_layers=1)
            else:
                self.encoder = DepEncoder(args, num_layers=2)

        self.dropout = nn.Dropout(0.33)

        self._add_position = getattr(self.args, "add_position_method", "cat")
        if self._add_position != "cat":
            print("添加position信息: ", self._add_position)

        self.relax_dep_mat = getattr(self.args, "relax_dep_mat", False)
        if self.relax_dep_mat:
            print("generate时, relax模型")

        if self._add_position == "cat":
            self.mlp_input_dim = args.decoder_embed_dim * 2
        else:
            self.mlp_input_dim = args.decoder_embed_dim

        self.glat_classifier = getattr(self.args, "glat_classifier", "none")
        if self.glat_classifier != "none":
            print("使用glat的方式来训练分类器: ", self.glat_classifier)

        self.increase_p = getattr(self.args, "increase_p", False)
        if self.increase_p:
            print("使用increase_p")

        self.distill_method = getattr(self.args, "distill_method", "none")
        if self.distill_method != "none":
            print("蒸馏方式： ", self.distill_method)

        self.share_with_ref = getattr(self.args, "share_with_ref", "none")
        if self.share_with_ref:
            print("和以ref做为输入的encoder共享: ", self.share_with_ref)

    def get_random_mask_output(self, mask_length=None, target_token=None, hidden_state=None, reference_embedding=None):
        # mask_length大，说明模型性能不行，所以多提供reference  ==> mask_length代表的是reference的数目
        hidden_state = hidden_state.transpose(0, 1)
        reference_embedding = reference_embedding.transpose(0, 1)
        reference_mask = get_base_mask(target_token)

        target_score = target_token.clone().float().uniform_()
        target_score.masked_fill_(~reference_mask, 2.0)

        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < mask_length[:, None].long()
        mask = target_cutoff.scatter(1, target_rank, target_cutoff)  # [b, l]
        non_mask = ~mask
        full_mask = torch.cat((non_mask.unsqueeze(-1), mask.unsqueeze(-1)), dim=-1)

        full_embedding = torch.cat((hidden_state.unsqueeze(-1), reference_embedding.unsqueeze(-1)), dim=-1)
        output_emebdding = (full_embedding * full_mask.unsqueeze(-2)).sum(-1)

        return ~mask, output_emebdding.transpose(0, 1)

    def forward_classifier(self, sample, hidden_state, target_tokens, ref_embedding, position_embedding,
                           decoder_padding_mask, encoder_out, **kwargs):

        loss_mask = None
        other = {}

        mask_length = None
        if self.glat_classifier != "none":
            with torch.no_grad():
                score, dep_hidden, _ = self._forward_classifier(hidden_state, position_embedding, decoder_padding_mask,
                                                                encoder_out, target_tokens=target_tokens)

            score = score.detach()
            sample_ids = sample['id'].cpu().tolist()
            reference = self.get_reference(sample_ids, target_tokens)
            reference_mask = reference != self.padding_idx
            label = self.get_label(score, target_tokens, reference_mask, use_MST=False, return_head=True)
            mask_length = self.get_mask_num(label, reference, reference_mask, kwargs.get('update_nums', 300000))

        if self.increase_p:
            reference_length = (target_tokens != 0).sum(-1)
            update_num = kwargs.get('update_nums', 300000)
            ratio = min((update_num / 300000) * 0.5 + 0.35, 0.8)
            mask_length = reference_length * (1 - ratio)

        if mask_length != None:
            loss_mask, hidden_state = self.get_random_mask_output(mask_length, target_tokens, hidden_state,
                                                                  ref_embedding)
        score, dep_hidden, other = self._forward_classifier(hidden_state, position_embedding, decoder_padding_mask,
                                                            encoder_out, target_tokens=target_tokens)

        return score, loss_mask, dep_hidden, other

    def inference(self, hidden_state, position_embedding, target_tokens, perturb=0.0, **kwargs):

        batch_size, _ = target_tokens.size()
        assert batch_size == 1, u"infernece仅仅支持batch=1"
        score, dep_hidden, _ = self._forward_classifier(hidden_state, position_embedding,
                                                        target_tokens=target_tokens,
                                                        decoder_padding_mask=target_tokens.eq(1),
                                                        **kwargs)
        _label = self.get_label(score, target_tokens=target_tokens)
        _label[0][0] = 0
        _label[0][-1] = 0
        _label[0][:, 0] = 0
        _label[0][:, -1] = 0

        if self.relax_dep_mat:
            mask = (_label == 1) | (_label.transpose(1, 2) == 1)
            _label.masked_fill_(mask, 1)

        def _position(target):
            left = ((position - 1) == target).sum().item()
            right = ((position + 1) == target).sum().item()
            return left + right

        if kwargs.get("eval_accuracy", False) or perturb != 0.0:
            # sample = kwargs['sample']
            # sample_ids = sample['id'].cpu().tolist()
            # head_ref = self.get_reference(sample_ids)
            #
            # head_mask = head_ref != -1
            #
            # predict = score[head_mask]
            # head_ref = head_ref[head_mask]
            # predict = self.get_head(predict)
            # target = head_ref
            # all = len(predict)
            # correct = (target == predict).sum().item()
            # set_key_value("all", all)
            # set_key_value("correct", correct)
            #
            # # 计算和位置的相关性
            # position = torch.range(1, all).cuda()
            # predict_pos = _position(predict)
            # target_pos = _position(target)
            # a = ((target == predict) & (predict == (position - 1))).sum().item()
            # b = ((target == predict) & (predict == (position + 1))).sum().item()
            # set_key_value("predict_pos", predict_pos)
            # set_key_value("target_pos", target_pos)
            # set_key_value("correct_pos", a + b)
            sample = kwargs.get("sample", None)
            sample_ids = sample['id'].cpu().tolist()
            reference = sample["target"]
            dependency_mat = self.relative_dep_mat.get_dependency_mat(sample_ids, reference, training=self.training,
                                                                      contain_eos=True)

            predict = _label
            target = dependency_mat
            b, l, _ = predict.size()
            all = b * l * l

            # correct = (target == predict).sum().item()
            # set_diff_tokens(correct)
            # set_all_token(all)

            # 阈值分析
            # score = F.sigmoid(score)
            # class_1 = score[dependency_mat == 1].cpu().tolist()
            # class_2 = score[dependency_mat == 2].cpu().tolist()
            # set_key_value_list("class_1", class_1)
            # set_key_value_list("class_2", class_2)

            def _count(i):
                predict_i = (predict == i).sum().item()  # 1 是相关
                target_i = (target == i).sum().item()
                correct_i = ((predict == i) & (target == i)).sum().item()
                return predict_i, target_i, correct_i

            def _perturb(label):
                predict_i, target_i, correct_i = _count(label)
                co_score = score + (target == label).long()
                co_score = co_score.view(-1)
                _, target_rank = co_score.sort(descending=True)
                mask = target_rank.new_zeros(target_rank.size()).bool().fill_(False)
                mask_length = target_i - correct_i
                mask[target_rank[:mask_length]] = True
                mask = mask.view(l, l)
                return mask

            name = ["pad", "positive", "negative", "same"]
            if perturb != 0.0:
                _label = target.clone()
                score = _label.clone().float().uniform_()
                _label.masked_fill_(_perturb(1), 2)
                _label.masked_fill_(_perturb(2), 1)
                predict = _label

            for i in [1, 2]:  # 0 pad 1 相关 2 不相关 3 相似
                predict_i, target_i, correct_i = _count(i)
                set_key_value("predict_" + name[i], predict_i)
                set_key_value("target_" + name[i], target_i)
                set_key_value("correct_" + name[i], correct_i)

        return _label, dep_hidden

    @staticmethod
    def add_args(parser):
        parser.add_argument('--dep-loss-factor', default=1.0, type=float)
        parser.add_argument('--encoder-dep-input', type=str, default="transformer")  # lstm
        parser.add_argument('--glat-classifier', type=str, default="none")  # all、input
        parser.add_argument('--increase-p', action="store_true")
        parser.add_argument('--distill-method', type=str, default="none")  # kl、mse
        parser.add_argument('--imitation', action="store_true")
        parser.add_argument('--no-mlp', action="store_true")

        """ 二分类器相关"""
        parser.add_argument('--positive-class-factor', type=float, default=1.0)
        parser.add_argument('--classifier-mutual-method', type=str, default="none", choices=['none', 'bias', 'logit'])
        parser.add_argument('--dep-focal-loss', action="store_true")
        parser.add_argument('--relax-dep-mat', action="store_true")
        parser.add_argument('--gumbel-softmax-mat', action="store_true")
        parser.add_argument('--threshold', type=float, default=0.556)
        parser.add_argument('--classifier-method', type=str, default="concat")  # biaffine

        """ 依存树相关"""
        parser.add_argument('--add-position-method', type=str, default="cat")  # 如何使用position信息 add、none、cat
        parser.add_argument('--use-MST', action="store_true")

        parser.add_argument('--share-with-ref', type=str, default="none")  # encoder、W、encoderW

    def inference_accuracy(self, compute_loss, eval_accuracy=False, target_tokens=None, **kwargs):
        loss, all, correct = 0, 0, 0
        score, predict, head_ref, train_mat, reference_mask, dep_hidden, other = self.predict(
            target_tokens=target_tokens,
            **kwargs)
        if compute_loss:
            ref_embedding = kwargs.get("ref_embedding", None)
            kwargs['hidden_state'] = ref_embedding
            decoder_padding_mask = target_tokens.eq(1)
            teacher_score, teacher_hidden, teacher_other = self.get_teacher_hidden(
                decoder_padding_mask=decoder_padding_mask, target_tokens=target_tokens,
                teacher=True, **kwargs)
            if teacher_score is not None:
                teacher_score = teacher_score[reference_mask]
            loss = self.compute_loss(predict, head_ref, dep_hidden=dep_hidden, teacher_hidden=teacher_hidden,
                                     teacher_score=teacher_score, reference_mask=reference_mask, other=other,
                                     teacher_other=teacher_other)
        if eval_accuracy:
            all, correct = self.compute_accuracy(predict, head_ref, score, target_tokens, mask=reference_mask)

        return loss, all, correct, train_mat, dep_hidden

    def get_teacher_hidden(self, **kwargs):
        if self.share_with_ref in ["encoderW", "encoder"]:  # 共享encoder和W矩阵
            teacher_score, teacher_hidden, teacher_other = self._forward_classifier(**kwargs)
            return teacher_score, teacher_hidden, teacher_other

        return None, None, None

    def predict(self, sample, target_tokens, result_mat, **kwargs):

        decoder_padding_mask = target_tokens.eq(1)
        score, loss_mask, dep_hidden, other = self.forward_classifier(sample=sample, target_tokens=target_tokens,
                                                                      decoder_padding_mask=decoder_padding_mask,
                                                                      **kwargs)

        sample_ids = sample['id'].cpu().tolist()
        reference = self.get_reference(sample_ids, target_tokens)

        reference_mask = reference != self.padding_idx
        if loss_mask is not None:
            if len(loss_mask.shape) != len(reference_mask.shape):
                loss_mask = loss_mask.unsqueeze(-1)
            reference_mask.masked_fill_(~loss_mask, False)
        mat = None
        if result_mat:
            mat = self.get_label(score, target_tokens, reference_mask)

        predict = score[reference_mask]
        dependency_mat = reference[reference_mask]

        return score, predict, dependency_mat, mat, reference_mask, dep_hidden, other

    def add_position(self, hidden_state, position):
        # 编码之后调用
        if self._add_position == "none":
            return hidden_state

        if self._add_position == "cat":
            return torch.cat((hidden_state, position), dim=-1)

        if self._add_position == "add":
            return hidden_state + position


class DepHeadClassifier(DepClassifier):
    def __init__(self, args, relative_dep_mat=None, use_two_class=False, head_tree=None, dep_file="", **kwargs):
        super().__init__(args, relative_dep_mat, use_two_class, **kwargs)

        self.padding_idx = -1
        self.use_MST = getattr(self.args, "use_MST", False)
        if self.use_MST:
            print("使用最小生成树，目前仅仅在生成矩阵（计算准确率+generate时）使用")

        if head_tree is None:
            head_tree = DepHeadTree(valid_subset=self.args.valid_subset, dep_file=dep_file)

        self.no_mlp = getattr(self.args, "no_mlp", False)
        self.biaffine_attention = BiaffineAttentionDependency(input_dim=self.mlp_input_dim, head_tree=head_tree)

        self.teacher_biaffine = None
        if self.share_with_ref == "encoder":
            self.teacher_biaffine = BiaffineAttentionDependency(input_dim=self.mlp_input_dim, head_tree=head_tree)

        self.imitation = getattr(self.args, "imitation", False)
        if self.imitation:
            self.imitation_module = ImitationModule(input_dim=self.mlp_input_dim, ffn_dim=self.mlp_input_dim * 2,
                                                    dropout=args.dropout)

    def compute_loss(self, outputs, targets, teacher_score=None, other=None, **kwargs):
        # 计算损失的肯定是依存树预测损失
        b_loss = self.biaffine_attention.compute_loss(outputs, targets) * self.dep_loss_factor
        loss = {"dep_classifier": {"loss": b_loss}}
        if teacher_score is not None:
            teacher_loss = self.biaffine_attention.compute_loss(teacher_score, targets)
            if teacher_loss.item() != 0:
                scale = max(min(b_loss.item() / teacher_loss.item(), 5), 1)
                teacher_loss = teacher_loss * scale
            loss.update({"teacher_dep": {"loss": teacher_loss}})
        return loss

    def _forward_classifier(self, hidden_state, position_embedding, decoder_padding_mask, encoder_out, teacher=False,
                            target_tokens=None, **kwargs):
        hidden_state = self.dropout(hidden_state)
        if self.encoder is not None:
            hidden_state = self.encoder(encoder_out, hidden_state, decoder_padding_mask)
        else:
            hidden_state = hidden_state.transpose(0, 1)
        _, _, embed_dim = hidden_state.size()
        hidden_state2 = self.add_position(hidden_state, position_embedding)
        hidden_state2 = self.dropout(hidden_state2)

        other = {}
        # if self.imitation:
        #     action, imitation_loss, output = self.imitation_module(hidden_state2, target_tokens)
        #     hidden_state2 = hidden_state2 + action
        #     other = {"action": action, "imitation_loss": imitation_loss, "output": output}
        #
        # if teacher and self.teacher_biaffine is not None:
        #     output, dep_hidden = self.teacher_biaffine.forward_classifier(hidden_state2,
        #                                                                   return_hidden=True)
        # else:
        output, dep_hidden = self.biaffine_attention.forward_classifier(hidden_state2,
                                                                        return_hidden=True)  # [b, tgt_len, tgt_len]
        return output, hidden_state2[:, :, :embed_dim], other

    def get_reference(self, sample_ids, target_tokens=None):
        head_ref = self.biaffine_attention.get_reference(sample_ids)
        return head_ref

    def get_head(self, score):
        return score.argmax(-1)

    def compute_accuracy(self, predict_head, head_ref, score, reference, mask=None):
        if self.use_MST:
            predict = []
            batch_size, tgt_len, tgt_len = score.size()
            base_mask = get_base_mask(reference)
            base_mask[:, 0] = True
            score = F.softmax(score, dim=-1)
            for b in range(batch_size):
                head = self.biaffine_attention.parser_tree(score[b].squeeze(0).cpu().numpy(), base_mask[b].sum().item(),
                                                           base_mask[b].int().cpu().numpy())
                predict.append(head)
            predict = head_ref.new_tensor(predict)
            base_mask[:, 0] = False
            if mask is not None:
                predict = predict[mask]
        else:
            predict = self.get_head(predict_head)
        target = head_ref
        all = len(predict)
        correct = (target == predict).sum().item()
        return all, correct

    def get_label(self, score, target_tokens=None, mask=None, use_MST=True, return_head=False):
        # score: [b, l, l]
        """ 只返回head时不会使用MST计算一下头"""
        dep_mat = score.new_zeros(score.size()).long()

        if not (self.use_MST and use_MST):
            all_head = self.get_head(score)
            if return_head:
                return all_head

        batch_size, tgt_len = target_tokens.shape
        score = F.softmax(score, dim=-1)
        for i in range(batch_size):
            ref_len = (target_tokens[i] != 1).sum().item()
            base_mask = get_base_mask(target_tokens[i])
            base_mask[0] = True
            if self.use_MST and use_MST:
                head = self.biaffine_attention.parser_tree(score.squeeze(0).detach().cpu().numpy(),
                                                           base_mask.sum().item(),
                                                           base_mask.int().cpu().numpy())
            else:
                head = all_head[i]
            dep_mat[i][1:ref_len - 1, 1:ref_len - 1] = 2  # 不相关

            for j in range(1, ref_len - 1):
                dep_mat[i][j][j] = 1
                h = head[j]
                if h == 0:  # 0是根节点
                    continue

                dep_mat[i][h][j] = 1
                dep_mat[i][j][h] = 1

                grand = head[h]
                if grand == 0:
                    continue
                dep_mat[i][grand][j] = 1
                dep_mat[i][j][grand] = 1

            # parent_child = {}
            # for c, p in enumerate(head[1: ref_len - 1]):
            #     parent_child.setdefault(p.item(), [])
            #     parent_child[p.item()].append(c + 1)
            # same_word = []
            # for p, c in parent_child.items():
            #     r = []
            #     if p + 1 in c:
            #
            #         if p == 0:
            #             r = [p + 1]
            #         else:
            #             r = [p, p + 1]
            #         for t in range(2, 1000):
            #             if p + t in c:
            #                 r.append(p + t)
            #             else:
            #                 break
            #         same_word.append(r)
            # # print(same_word)
            # for same in same_word:
            #     if len(same) <= 1:
            #         continue
            #     r = dep_mat[i][same[0]]
            #     for s in same:
            #         r.masked_fill_(dep_mat[i][s] == 1, 1)
            #     for s in same:
            #         dep_mat[i][s] = r

        return dep_mat

    def get_mask_num(self, label, reference, reference_mask, update_nums):
        ratio = (1 - update_nums / 300000) * 0.2 + 0.3
        diff = ((label != reference) & reference_mask).sum(-1).detach()
        mask_length = (diff * ratio).round()
        return mask_length


class DoubleClassifier(DepClassifier):
    def __init__(self, args, relative_dep_mat=None, use_two_class=False, **kwargs):
        super().__init__(args, relative_dep_mat, use_two_class, **kwargs)
        self.student_encoder = None
        self.teacher_encoder = None

    def get_imitation(self, teacher_output, student_output):
        # 优化交叉熵等效于优化相对熵
        _ce_loss = F.kl_div(student_output.log(), teacher_output, reduction="none").sum(-1).mean()

        return _ce_loss

    def get_label(self, score, target_tokens=None, mask=None, use_MST=True, return_head=False):
        return self.student_encoder.get_label(score, target_tokens, mask, use_MST, return_head)

    def _forward_classifier(self, hidden_state, position_embedding, decoder_padding_mask, encoder_out, **kwargs):
        return self.student_encoder._forward_classifier(hidden_state, position_embedding, decoder_padding_mask,
                                                        encoder_out, **kwargs)

    def get_mask_num(self, label, reference, reference_mask):
        return self.student_encoder.get_mask_num(label, reference, reference_mask)

    def compute_loss(self, outputs, targets, dep_hidden=None, teacher_hidden=None, teacher_score=None, other=None,
                     reference_mask=None, teacher_other=None):
        loss = self.student_encoder.compute_loss(outputs, targets, None)

        if teacher_score is not None:
            teacher_loss = self.teacher_encoder.compute_loss(teacher_score, targets)
            loss.update({"teacher_dep": teacher_loss.get("dep_classifier")})

        def _dep_hidden(index):
            return dep_hidden[index][reference_mask]

        def _teacher_hidden(index):
            return teacher_hidden[index][reference_mask].detach()

        if self.distill_method == "mse":
            if teacher_hidden is not None and dep_hidden is not None:
                if isinstance(dep_hidden, tuple):
                    l1 = F.mse_loss(_dep_hidden(0), _teacher_hidden(0), reduction="none").sum(-1).mean()
                    l2 = F.mse_loss(_dep_hidden(1), _teacher_hidden(1), reduction="none").sum(-1).mean()
                    mse_loss = l1 + l2
                else:
                    mse_loss = F.mse_loss(dep_hidden[reference_mask], teacher_hidden[reference_mask].detach(),
                                          reduction="none").sum(-1).mean()

                loss.update({"distill_loss": {"loss": mse_loss * 0.001}})

        if self.distill_method == "kl":
            if teacher_score is not None and outputs is not None:
                l = F.kl_div(outputs.log_softmax(-1), teacher_score.softmax(-1).detach(), reduction='none')
                l = l.sum(-1).mean()
                loss.update({"distill_kl_loss": {"loss": l}})

        if self.distill_method == "cos":
            if teacher_hidden is not None and dep_hidden is not None:
                if isinstance(dep_hidden, tuple):
                    l1 = cos_loss(_dep_hidden(0), _teacher_hidden(0))
                    l2 = cos_loss(_dep_hidden(1), _teacher_hidden(1))
                    _cos_loss = l1 + l2
                else:
                    _cos_loss = cos_loss(dep_hidden[reference_mask], teacher_hidden[reference_mask].detach())
                loss.update({"distill_cos_loss": {"loss": _cos_loss}})

        if other is not None and len(other) > 0:
            teacher_imitation = teacher_other.get('imitation_loss', None)
            teacher_action = teacher_other.get("output", None)[reference_mask]
            student_action = other.get("output", None)[reference_mask]
            imitation = self.get_imitation(teacher_action, student_action)
            loss.update({
                "teacher_imitation": {"loss": teacher_imitation},
                "imitation_loss": {"loss": imitation * 10}
            })

        return loss

    def get_reference(self, sample_ids, target_tokens=None):
        return self.student_encoder.get_reference(sample_ids, target_tokens)

    def get_head(self, score):
        return self.student_encoder.get_head(score)

    def compute_accuracy(self, predict_head, head_ref, score, reference, mask=None):
        return self.student_encoder.compute_accuracy(predict_head, head_ref, score, reference, mask)

    def get_teacher_hidden(self, hidden_state, position_embedding, decoder_padding_mask, encoder_out, **kwargs):
        return self.teacher_encoder._forward_classifier(hidden_state, position_embedding, decoder_padding_mask,
                                                        encoder_out, **kwargs)


class DHeadClassifier(DoubleClassifier):

    def __init__(self, args, relative_dep_mat=None, use_two_class=False, dep_file='', **kwargs):
        super().__init__(args, relative_dep_mat, use_two_class, **kwargs)
        head_tree = DepHeadTree(valid_subset=self.args.valid_subset, dep_file=dep_file)
        print("--------------------student model : ----------------------")
        self.student_encoder = DepHeadClassifier(args, relative_dep_mat, use_two_class, head_tree, **kwargs)

        print("--------------------teacher model : ----------------------")
        args.glat_classifier = False
        self.teacher_encoder = DepHeadClassifier(args, relative_dep_mat, use_two_class, head_tree, **kwargs)

        self.padding_idx = self.student_encoder.padding_idx


#
class DepCoarseClassifier(DepClassifier):
    pass
#     def __init__(self, args, relative_dep_mat=None, use_two_class=False, **kwargs):
#         super().__init__(args, relative_dep_mat, use_two_class)
#
#         self.padding_idx = 0
#         self.dep_focal_loss = getattr(self.args, "dep_focal_loss", False)
#         if self.dep_focal_loss:
#             print("use focal loss: ", self.dep_focal_loss)
#
#         self.gumbel_softmax_mat = getattr(self.args, "gumbel_softmax_mat", False)
#         if self.gumbel_softmax_mat:
#             print("使用gumbel-softmax获得依存矩阵")
#
#         self.classifier_mutual_method = getattr(self.args, "classifier_mutual_method", "none")
#         if self.classifier_mutual_method != "none":
#             print("是否使用互信息来解决类别不均衡问题: ", self.classifier_mutual_method)
#
#             if self.classifier_mutual_method in ["logit", "bias"]:
#                 self.log_prior = self.mutual_logits()
#
#         self.threshold = getattr(self.args, "threshold", 0.556)
#         print("threshold: ", self.threshold)
#
#         self.mlp_dim = 400
#         # self.mlp_dim = self.mlp_input_dim
#         self.class_num = 3
#         if use_two_class and not self.gumbel_softmax_mat:
#             self.class_num = 1
#             self.loss = nn.BCEWithLogitsLoss(reduction='none')
#
#         if self.gumbel_softmax_mat:
#             self.class_num = 2
#
#         self.classifier_method = getattr(self.args, "classifier_method", "concat")
#         if self.classifier_method == "biaffine":
#             self.mlp_output = nn.Sequential(
#                 nn.Linear(self.mlp_input_dim, self.mlp_dim),
#                 nn.LeakyReLU(),
#                 nn.Dropout(0.33))
#             self.W_arc = nn.Parameter(orthogonal_(
#                 torch.empty(self.mlp_dim + 1, self.mlp_dim).cuda()
#             ), requires_grad=True)
#         else:
#             self.mlp_output = nn.Sequential(
#                 nn.Linear(self.mlp_input_dim * 2, self.mlp_dim),
#                 nn.Tanh(),
#                 nn.Dropout(0.3),
#                 nn.Linear(self.mlp_dim, self.class_num),
#             )
#         self.positive_class_factor = getattr(self.args, "positive_class_factor", 1.0)
#
#     def get_reference(self, sample_ids, target_tokens):
#         dependency_mat = self.relative_dep_mat.get_dependency_mat(sample_ids, target_tokens, training=self.training)
#         return dependency_mat
#
#     def mutual_logits(self):
#         prior = [0.143, 0.857]
#         log_prior = np.log(prior)
#         tau = 1.0
#         delta = (log_prior[1] - log_prior[0]) * tau
#         return delta
#
#     def apply_init_bias(self):
#         self.mlp_output[-1].bias.data = torch.FloatTensor([self.log_prior])
#
#     def get_mask_num(self, label, reference, reference_mask):
#         diff = ((label != reference) & reference_mask).sum(-1).sum(-1).detach()
#         mask_length = (diff.pow(1 / 2) * 0.5).round()
#         return mask_length
#
#     def compute_loss(self, outputs, targets, **kwargs):
#
#         targets = targets - 1  # 去除pad类别
#         # 多分类
#         if self.class_num != 1:
#             logits = F.log_softmax(outputs, dim=-1)
#             losses = F.nll_loss(logits, targets.to(logits.device), reduction='none')
#         else:
#             losses = self.loss(outputs.squeeze(-1), targets.float())
#
#         if self.positive_class_factor != 1.0:
#             weights = losses.new_ones(size=losses.size()).float()
#             weights.masked_fill_(targets == 0, self.positive_class_factor)
#             losses = losses * weights
#
#         if self.dep_focal_loss:
#             p_logits = F.sigmoid(outputs)
#             neg_logits = 1 - p_logits
#             p_logits = p_logits.masked_fill(targets == 1, 1).detach()
#             p_logits = p_logits * p_logits
#             neg_logits = neg_logits.masked_fill(targets == 0, 1).detach()
#             neg_logits = neg_logits * neg_logits
#             losses = losses * p_logits * neg_logits
#
#         loss = mean_ds(losses)
#
#         loss = loss * self.dep_loss_factor
#         loss = {"dep_classifier": {"loss": loss}}
#         return loss
#
#     def _forward_classifier(self, hidden_state, position_embedding, decoder_padding_mask, encoder_out, **kwargs):
#
#         hidden_state = self.dropout(hidden_state)
#         if self.encoder is not None:
#             hidden_state = self.encoder(encoder_out, hidden_state, decoder_padding_mask)
#         else:
#             hidden_state = hidden_state.transpose(0, 1)
#
#         hidden_state = self.add_position(hidden_state, position_embedding)
#         hidden_state = self.dropout(hidden_state)
#
#         if self.classifier_method == "biaffine":
#             feature = self.mlp_output(hidden_state)
#             batch_size, seq_len, hidden_dim = hidden_state.size()
#             arc_dep = torch.cat((feature, torch.ones(batch_size, seq_len, 1).cuda()),
#                                 dim=-1)  # batch * trg_len * (dim+1)
#
#             score = arc_dep.matmul(self.W_arc).matmul(feature.transpose(1, 2))
#             dep_hidden = feature
#         else:
#             batch_size, seq_len, hidden_dim = hidden_state.size()
#             a = hidden_state.unsqueeze(1).repeat(1, seq_len, 1, 1)  # [b, l, l, d]
#             b = hidden_state.unsqueeze(2).repeat(1, 1, seq_len, 1)  # [b, l, l, d]
#             feature = torch.cat((a, b), dim=-1)
#             score = self.mlp_output(feature)  # [b, l, l, 3]
#             dep_hidden = None
#
#         if self.classifier_mutual_method == "logit":
#             assert self.class_num == 1, "只支持二分类打分"
#             score = score + self.log_prior
#         return score.squeeze(-1), dep_hidden
#
#     def sample_mat(self, score):
#         mat = F.gumbel_softmax(score, hard=True, tau=1)
#         if len(mat.size()) == 4:
#             result = mat[:, :, :, 1] + 1  # 相关
#         if len(mat.size()) == 3:
#             result = mat[:, :, 1] + 1
#         if len(mat.size()) == 2:
#             result = mat[:, 1] + 1
#         return result.long()
#
#     def get_label(self, score, target_tokens=None, reference_mask=None, **kwargs):
#         """预测->依存矩阵"""
#         if self.gumbel_softmax_mat:
#             assert self.class_num == 2, "gumbel softmax只支持分类呀~"
#             mat = self.sample_mat(score)
#             return mat
#         if self.class_num == 1:
#             mat = (F.sigmoid(score) > self.threshold).long() + 1
#         else:
#             _score, _label = F.log_softmax(score, dim=-1).max(-1)
#             mat = _label + 1
#
#         if reference_mask is not None:
#             mat = mat.masked_fill(~reference_mask, self.padding_idx)
#
#         return mat
#
#     def compute_accuracy(self, predict, target, score, target_token, **kwargs):
#         _label = self.get_label(predict)
#
#         all = len(_label)
#         correct = (target == _label).sum().item()
#         return all, correct
