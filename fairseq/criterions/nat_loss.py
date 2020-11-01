# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
from torch import Tensor

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("nat_loss")
class LabelSmoothedDualImitationCriterion(FairseqCriterion):

    def __init__(self, task, label_smoothing):
        super().__init__(task)
        self.label_smoothing = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            '--label-smoothing',
            default=0.,
            type=float,
            metavar='D',
            help='epsilon for label smoothing, 0 means no label smoothing',
        )

    def _compute_loss(
            self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0,
            weights=None
    ):
        """
            outputs: batch x len x d_model
            targets: batch x len
            masks:   batch x len

            policy_logprob: if there is some policy
                depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction='none')

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction='none')
                losses = losses.sum(-1)

            if weights is not None:
                token_weight = None
                if "dependency_weights" in weights and weights['dependency_weights'] is not None:
                    token_weight = weights['dependency_weights'][masks]

                if "freq_weights" in weights and weights['freq_weights'] is not None:
                    freq_weight = weights['freq_weights'][targets]
                    if token_weight is not None:
                        token_weight = (token_weight + freq_weight) / 2
                    else:
                        token_weight = freq_weight

                if token_weight is not None:
                    losses.mul_(token_weight)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = nll_loss * (
                        1 - label_smoothing) - mean_ds(logits) * label_smoothing
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, reduce=True, **kwargs):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens, sample=sample, **kwargs)
        losses, nll_loss = [], []

        # 仅仅预测依存树的下一层节点, 所以模型会主动传递mask
        if sample.get('word_ins_mask', None) is not None:
            outputs["word_ins"]["mask"] = sample['word_ins_mask']

        # 取出需要回传的
        train_need = None
        if "train_need" in outputs:
            train_need = outputs.pop('train_need')

        for obj in outputs:

            if outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + '-loss',
                    factor=outputs[obj].get("factor", 1.0),
                    weights=outputs[obj].get('weights', None)
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + '-loss',
                    factor=outputs[obj].get("factor", 1.0)
                )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 \
            else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "train_need": train_need,
            "need_print": train_need.get('print', {}) if train_need is not None else {}
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

        # add some need print
        if logging_outputs[0].get('need_print', None) is not None:
            if "all_predict_head" in logging_outputs[0]['need_print']:
                all_predict_head = sum(log['need_print'].get("all_predict_head") for log in logging_outputs)
                correct_predict_head = sum(log['need_print'].get("correct_predict_head") for log in logging_outputs)
                # print(correct_predict_head / all_predict_head)
                metrics.log_scalar('dep_accuracy', correct_predict_head / all_predict_head, weight=all_predict_head,
                                   round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
