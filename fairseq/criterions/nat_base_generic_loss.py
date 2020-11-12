import math

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.nat_loss import LabelSmoothedDualImitationCriterion


class Accuracy(object):
    def __init__(self, counts=0, totals=0):
        self.counts = counts
        self.totals = totals

    @property
    def score(self):
        if self.totals <= 0:
            return 0.0
        else:
            return 100.0 * self.counts / self.totals

    def format(self, width=2):
        precisions = "/".join(["{:.1f}".format(p) for p in [self.counts, self.totals]])
        return 'Accuracy = {score:.{width}f} {precisions}'.format(
            score=self.score,
            width=width,
            precisions=precisions,
        )

    def __iter__(self):
        return self

    def __add__(self, other):
        if isinstance(other, int):
            return self
        return Accuracy(
            counts=self.counts + other.counts,
            totals=self.totals + other.totals,
        )

    def __radd__(self, other):
        return self + other

    @classmethod
    def compute_acc(cls, tgt, out=None, mask=None, pred=None):
        # assert out is not None and tgt is not None, "output and target should not be empty"
        if out is None or tgt is None:
            return Accuracy()
        tgt = tgt.view(-1)
        if pred is None and out is not None:
            out = out.view(-1, out.size(-1))
            pred = out.max(-1)[1]

        indicate = pred.eq(tgt)

        if mask is not None:
            mask = mask.view(-1)
            indicate = indicate * mask
            totals = mask.sum().item()
        else:
            totals = tgt.size(0)
        counts = indicate.sum().item()

        return Accuracy(counts=counts, totals=totals)


@register_criterion("generic_loss")
class GenericLMCriterion(LabelSmoothedDualImitationCriterion):
    """
    return the loss and accuracy
    """

    def _compute_acc(self, acc, name="acc"):
        return {"name": name, "acc": acc}

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
        accuracies = []

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
                    factor=outputs[obj].get("factor", 1.0)
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + '-loss',
                    factor=outputs[obj].get("factor", 1.0)
                )
            if not model.args.no_accuracy and not outputs[obj].get("no-acc", False):
                if not model.training and "tgt" in outputs[obj]:  # compute only for evaluation
                    acc = self._compute_acc(
                        acc=Accuracy.compute_acc(
                            pred=outputs[obj].get("pred", None),
                            out=outputs[obj].get("out", None),
                            tgt=outputs[obj].get("tgt", None),
                            mask=outputs[obj].get("mask", None),
                        ),
                        name=obj + '-acc'
                    )
                    accuracies += [acc]
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
        for l in accuracies:
            logging_output[l["name"]] = l["acc"]

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
            elif "acc" in key:
                val = sum([log.get(key, Accuracy()) for log in logging_outputs], Accuracy())
                metrics.log_scalar(key, val.score, sample_size, round=2)

        if logging_outputs[0].get('need_print', None) is not None:
            if "all_predict_head" in logging_outputs[0]['need_print']:
                all_predict_head = sum(log['need_print'].get("all_predict_head") for log in logging_outputs)
                correct_predict_head = sum(log['need_print'].get("correct_predict_head") for log in logging_outputs)
                # print(correct_predict_head / all_predict_head)
                metrics.log_scalar('dep_accuracy', correct_predict_head / all_predict_head, weight=all_predict_head,
                                   round=3)
            if "mask_ratio" in logging_outputs[0]['need_print']:
                mask_ratio = utils.item(sum(log['need_print']['mask_ratio'] for log in logging_outputs))
                metrics.log_scalar('mask_ratio', mask_ratio, weight=0, priority=300)
