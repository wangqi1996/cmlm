# encoding=utf-8

from fairseq.tasks import register_task
from fairseq.tasks.translation_lev import TranslationLevenshteinTask
from fairseq.util2 import get_reference_mask, get_dependency_mask, merge_mask, get_base_mask
from fairseq.util2 import load_dependency_tree


@register_task('translation_nat')
class TranslationNATTask(TranslationLevenshteinTask):

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):

        super().load_dataset(split, epoch=epoch, combine=combine, **kwargs)
        if split == "train" and self.args.dependency_tree_path:
            self.dependency_dataset = load_dependency_tree(self.args.dependency_tree_path)
        else:
            self.dependency_dataset = None

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

    def get_iterative_epoch(self, sample):
        if self.args.use_dynamic_iterative_epoch:
            _, len_y = sample['target'].shape
            iterative_training_epoch = min(4, int(len_y / 10) + 1)
        else:
            iterative_training_epoch = self.args.iterative_training_epoch

        return iterative_training_epoch

    def end_iterative(self, epoch_num, iterative_training_epoch, sample, mask):
        if epoch_num >= iterative_training_epoch:
            # 是否根据模型运行的结果来选择某个sample是否停止
            if not self.args.use_iterative_epoch_with_time:
                return True

            all_tokens = self.get_base_mask(sample['target']).sum().item()
            required = mask.sum().item()
            if required < all_tokens * 0.15 or epoch_num >= 10:  # 最大迭代轮数
                return True

        return False

    def get_mask_target(self, sample, predicted_output, dependency_list, epoch_num, unk):
        mask = None
        # 训练节点的mask策略
        if self.args.use_reference_mask:
            reference_mask = get_reference_mask(target_tokens=sample['target'], predicted=predicted_output)
            mask = merge_mask(mask, reference_mask)

        if self.dependency_dataset is not None:
            dependency_mask = get_dependency_mask(sample['target'], dependency_list, step=epoch_num - 1,
                                                  modify=False)
            mask = merge_mask(mask, dependency_mask)

        # 是否需要训练
        required = mask.sum().item()
        if required == 0:
            return None, None

        return sample['target'].masked_fill(mask, unk), mask

    def add_loss(self, sum_loss, count_loss, loss):
        if loss.item() == 0:
            return sum_loss, count_loss
        if sum_loss is None:
            sum_loss = loss
            count_loss = 1
        else:
            sum_loss += loss
            count_loss += 1
        return sum_loss, count_loss

    def baseline_train(self,
                       sample,
                       model,
                       criterion,
                       optimizer,
                       update_num,
                       ignore_grad=False):
        noise_probability = self.noise_probability_schedule(update_num)
        sample['prev_target'] = self.inject_noise(sample['target'], noise_probability)
        loss, sample_size, logging_output = criterion(model, sample, need_sample=False)

        return loss, sample_size, logging_output

    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False):
        model.train()

        # 前XXX轮仅仅训练NAT
        # 紧接着，开始训练后验
        # 最后，开始训练先验
        if update_num < -1:
            # 这样的话，分段阶段不好弄
            loss, sample_size, logging_output = self.baseline_train(sample, model, criterion, optimizer, update_num,
                                                                    ignore_grad)
        else:
            # if update_num == 14000:
            #     model.froze_prior(froze=True)
            # if update_num >= 20000:
            #     if update_num % 4000 == 0:
            #         model.froze_posterior(True)
            #         model.froze_prior(False)
            #     elif update_num % 2000 == 0:
            #         model.froze_posterior(False)
            #         model.froze_prior(True)

            epoch_numbers = self.get_iterative_epoch(sample)
            reference = sample['target']
            posterior_sample = None
            predict_word = None
            sum_loss = None
            count_loss = 0

            for i in range(epoch_numbers):

                if i == epoch_numbers - 1:
                    need_sample = False
                else:
                    need_sample = True

                if i == 0:
                    mask = get_base_mask(reference)
                    prev_target = reference.masked_fill(mask, self.tgt_dict.unk())
                else:
                    # 不可导！
                    # prev_target = predict_word.masked_fill(posterior_sample[:, :, 0].bool() & get_base_mask(reference),
                    #                                        self.tgt_dict.unk())  # dim=0表示是否mask
                    # predict_unk = predict_word.new_empty(predict_word.shape, requires_grad=False).fill_(
                    #     self.tgt_dict.unk())
                    # predict_word_unk = torch.cat((predict_word.unsqueeze(-1), predict_unk.unsqueeze(-1)), dim=-1)
                    #
                    # prev_target = (predict_word_unk * posterior_sample).sum(-1)
                    # prev_target_mask = None
                    prev_target = None
                    sample['posterior_sample'] = posterior_sample

                sample['prev_target'] = prev_target
                sample['need_sample'] = need_sample

                loss, sample_size, logging_output = criterion(model, sample, need_sample=need_sample,
                                                              posterior_sample=posterior_sample,
                                                              predict_word=predict_word)

                if i != epoch_numbers - 1:
                    posterior_sample = logging_output['train_need']['posterior_sample']
                    predict_word = logging_output['train_need']['predict_word']

                sum_loss, count_loss = self.add_loss(sum_loss, count_loss, loss)

            loss = sum_loss / count_loss

        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)

        if "train_need" in logging_output:
            logging_output.pop("train_need")
        return loss, sample_size, logging_output
    #
    # def train_step(self,
    #                sample,
    #                model,
    #                criterion,
    #                optimizer,
    #                update_num,
    #                ignore_grad=False):
    #     model.train()
    #
    #     if self.args.use_baseline_train:
    #         noise_probability = self.noise_probability_schedule(update_num)
    #         sample['prev_target'] = self.inject_noise(sample['target'], noise_probability)
    #         loss, sample_size, logging_output = criterion(model, sample)
    #         if ignore_grad:
    #             loss *= 0
    #         optimizer.backward(loss)
    #
    #     else:
    #         predicted_output = None
    #         sample_size = 0
    #         unk = self.tgt_dict.unk()
    #         sum_loss = None
    #         count_loss = 0
    #
    #         if self.dependency_dataset is not None:
    #             dependency_list = [self.dependency_dataset[id] for id in sample["id"].cpu().tolist()]
    #
    #         iterative_training_epoch = self.get_iterative_epoch(sample)
    #         epoch_num = 0
    #         while True:
    #
    #             prev_target, mask = self.get_mask_target(sample, predicted_output, dependency_list, epoch_num, unk)
    #
    #             if prev_target is None:
    #                 break
    #             sample['prev_target'] = prev_target
    #
    #             # 仅仅学习预测dependency这种依赖关系
    #             if self.args.use_dependency_loss:
    #                 word_ins_mask = get_dependency_mask(sample['target'], dependency_list, step=epoch_num, modify=False)
    #                 sample['word_ins_mask'] = word_ins_mask
    #
    #             loss, sample_size, logging_output = criterion(model, sample)
    #             predicted_output = logging_output['predicted_output']
    #             prev_mask = sample['prev_target'].ne(self.tgt_dict.unk())
    #             predicted_output.masked_scatter_(prev_mask, sample['prev_target'][prev_mask])
    #
    #             if ignore_grad:
    #                 loss *= 0
    #             sum_loss, count_loss = self.add_loss(sum_loss, count_loss, loss)
    #
    #             epoch_num += 1
    #             if self.end_iterative(epoch_num, iterative_training_epoch, sample, mask):
    #                 break
    #
    #         optimizer.backward(sum_loss / count_loss)
    #
    #     return loss, sample_size, logging_output
