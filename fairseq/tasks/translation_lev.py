# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq import utils
from fairseq.data import LanguagePairDataset
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset, EVAL_BLEU_ORDER
from fairseq.util2 import new_arange, get_base_mask


@register_task('translation_lev')
class TranslationLevenshteinTask(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument(
            '--noise',
            default='random_delete',
            choices=['random_delete', 'random_mask', 'no_noise', 'full_mask'])

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            prepend_bos=True,
        )

    def inject_noise(self, target_tokens, noise_probability=None):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0)
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True)

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = 2 + ((target_length - 2) * target_score.new_zeros(
                target_score.size(0), 1).uniform_()).long()
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = target_tokens.gather(
                1, target_rank).masked_fill_(target_cutoff, pad).gather(
                1,
                target_rank.masked_fill_(target_cutoff,
                                         max_len).sort(1)[1])
            prev_target_tokens = prev_target_tokens[:, :prev_target_tokens.
                ne(pad).sum(1).max()]

            return prev_target_tokens

        def _random_mask(target_tokens, noise_probability=None):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = target_tokens.ne(pad) & \
                           target_tokens.ne(bos) & \
                           target_tokens.ne(eos)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            if noise_probability is None:
                # sample from [0,1]
                target_length = target_length * target_length.clone().uniform_()  # 要mask的长度
            else:
                target_length = target_length * noise_probability
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk)
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = target_tokens.eq(bos) | target_tokens.eq(
                eos) | target_tokens.eq(pad)
            return target_tokens.masked_fill(~target_mask, unk)

        if self.args.noise == 'random_delete':
            return _random_delete(target_tokens)
        elif self.args.noise == 'random_mask':
            return _random_mask(target_tokens, noise_probability)
        elif self.args.noise == 'full_mask':
            return _full_mask(target_tokens)
        elif self.args.noise == 'no_noise':
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args):
        # add models input to match the API for SequenceGenerator
        from fairseq.iterative_refinement_generator import IterativeRefinementGenerator
        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
            max_iter=getattr(args, 'iter_decode_max_iter', 10),
            beam_size=getattr(args, 'iter_decode_with_beam', 1),
            reranking=getattr(args, 'iter_decode_with_external_reranker', False),
            decoding_format=getattr(args, 'decoding_format', None),
            adaptive=not getattr(args, 'iter_decode_force_max_iter', False),
            retain_history=getattr(args, 'retain_iter_history', False),
            args=args)

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary, append_bos=True
        )

    def noise_probability_schedule(self, update_number):

        if self.args.noise_probability == 'none':
            return None
        if update_number < 20000 or update_number > 250000:
            return None
        p = update_number / 250000

        return p

    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False):
        model.train()
        noise_probability = self.noise_probability_schedule(update_num)
        sample['prev_target'] = self.inject_noise(sample['target'], noise_probability)

        special_input = model.get_special_input(sample, update_num=update_num)
        loss, sample_size, logging_output = criterion(model, sample, update_nums=update_num, **special_input)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)

        if "train_need" in logging_output:
            logging_output.pop("train_need")

        if "need_print" in logging_output and len(logging_output['need_print']) == 0:
            logging_output.pop("need_print")

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        # 不用在乎梯度，没有梯度问题
        model.eval()

        _logging_output = None

        # 这个会不会和其余的地方冲突，但是因为是valid，没有loss一说
        if self.args.eval_accuracy:
            with torch.no_grad():
                target_mask = get_base_mask(sample['target'])
                sample['prev_target'] = sample['target'].masked_fill(target_mask, self.tgt_dict.unk())

                _, _, _logging_output = criterion(model, sample, eval_accuracy=True)

        with torch.no_grad():
            sample['prev_target'] = self.inject_noise(sample['target'], None)
            special_input = model.get_special_input(sample, update_num=None)
            loss, sample_size, logging_output = criterion(model, sample, **special_input)

            if _logging_output is not None:
                logging_output.setdefault('need_print', {})
                # 因为之前的办法计算的hidden是mask掉部分target token计算的
                logging_output['need_print'].update(_logging_output['need_print'])

        if getattr(self.args, 'eval_bleu', False):
            sample['special_input'] = special_input
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]

        if "train_need" in logging_output:
            logging_output.pop("train_need")

        if "need_print" in logging_output and len(logging_output['need_print']) == 0:
            logging_output.pop("need_print")

        return loss, sample_size, logging_output
