# encoding=utf-8
import numpy as np
import torch
import torch.nn.functional as F


def get_base_mask(target_tokens):
    """ mask=True 表示不是特殊字符"""
    pad = 0
    bos = 1
    eos = 2

    target_masks = target_tokens.ne(pad) & \
                   target_tokens.ne(bos) & \
                   target_tokens.ne(eos)

    return target_masks


def merge_mask(mask1, mask2):
    if mask1 is None:
        return mask2

    if mask2 is None:
        return mask1
    return mask1 & mask2


# def get_comma_mask():
#     if use_comma_mask:
#         assert comma_mask_list is not None
#         dependency_mask = output_tokens.new_full((output_tokens.shape), True).bool()
#         # dependency_mask.masked_fill_(~output_masks, False)
#         _, len_y = dependency_mask.shape
#         for index, dependency in enumerate(comma_mask_list):
#             if step < len(dependency):
#                 for d in dependency[step]:
#                     dependency_mask[index][d + 1] = False
#         if skeptical_mask is not None:
#             skeptical_mask.masked_fill_(~dependency_mask, False)
#         else:
#             skeptical_mask = dependency_mask

# def use_reference_probability():
#     if use_reference_probability != 0.0:
#         batch_size, target_len, vocab_size = logits.shape
#
#         flatten_logits = logits.view(-1, vocab_size)
#         flatten_reference = references.reshape(-1, 1)
#         flatten_reference_score = flatten_logits.gather(-1, flatten_reference)
#
#         references_score = flatten_reference_score.view(batch_size, -1)  # 越大越好
#         probability_mask = (references_score < use_reference_probability)  # True 表示需要mask
#
#         if skeptical_mask is not None:
#             # diff = ((skeptical_mask == True) & (probability_mask == False)).long().sum().item()
#             # print(diff)
#             # set_diff_tokens(diff)
#             # all_tokens = (skeptical_mask == True).long().sum().item()
#             # set_all_token(all_tokens)
#             skeptical_mask.masked_fill_(~probability_mask, False)
#         else:
#             skeptical_mask = probability_mask

#
# def use_block():
#     # use block  未处理<pad>和<s></s>
#     if use_block_mask_size != 0 and use_block_mask_method != 'none':
#         batch_size, target_len, vocab_size = logits.shape
#
#         need_padded = target_len % use_block_mask_size
#         if need_padded != 0:
#             need_padded = use_block_mask_size - need_padded
#             padded = skeptical_mask.new_full((batch_size, need_padded), fill_value=1,
#                                              requires_grad=False).bool()
#             padded_skeptical_mask = torch.cat((skeptical_mask, padded), dim=-1)
#         else:
#             padded_skeptical_mask = skeptical_mask
#
#         need_block_select = (~padded_skeptical_mask).reshape(batch_size, -1, use_block_mask_size).sum(-1) == 0
#
#         if need_block_select.sum() != 0:
#             if use_block_mask_method == "baseline":
#                 block_score = output_scores
#             elif use_block_mask_method == "probability":
#                 flatten_logits = logits.view(-1, vocab_size)
#                 flatten_reference = references.reshape(-1, 1)
#                 flatten_reference_score = flatten_logits.gather(-1, flatten_reference)
#
#                 block_score = flatten_reference_score.view(batch_size, -1)  # 越大越好
#             else:
#                 raise Exception(u"unknown methodd!!!!")
#
#             if need_padded != 0:
#                 padded = block_score.new_full((batch_size, need_padded), fill_value=float('-inf'),
#                                               requires_grad=False)
#                 padded_output_score = torch.cat((block_score, padded), dim=-1)
#             else:
#                 padded_output_score = block_score
#             max_index = padded_output_score.reshape(batch_size, -1, use_block_mask_size).max(-1)[1].reshape(
#                 batch_size, -1, 1)
#             # default: True
#             gram_mask = padded_skeptical_mask.new_full((batch_size, need_padded + target_len), fill_value=1,
#                                                        requires_grad=False).bool().reshape(batch_size, -1,
#                                                                                            use_block_mask_size)
#
#             gram_mask.scatter_(-1, max_index, False)
#             need_block_select = need_block_select.reshape(batch_size, -1, 1)
#             gram_mask.masked_fill_(~need_block_select, True)
#
#             gram_mask = gram_mask.reshape(batch_size, -1)[:, :target_len]
#             skeptical_mask.masked_fill_(~gram_mask, False)

def get_reference_mask(reference, predicted=None):
    """ mask=True ===> 不是特殊字符且和reference不一致"""
    target_masks = get_base_mask(reference)

    if predicted is None:
        mask = target_masks
    else:
        correct = predicted != reference
        mask = correct & target_masks  # mask掉那些 不是特殊字符但是预测错误的
    return mask


def get_dependency_mask(target_tokens, dependency_list, step, modify=False, reference=None, random_modify_token=True):
    """
    将dependency的step步进行mask。
    """
    target_masks = get_base_mask(target_tokens)

    if step != 0:
        modify = False

    import numpy as np
    if step >= 0:
        for index, dependency in enumerate(dependency_list):
            if step < len(dependency):
                for d in dependency[step]:
                    target_masks[index][d + 1] = False  # 开头的bos,所以+1
                    if modify:
                        if random_modify_token:
                            if target_tokens[index][d + 1] != reference[index][d + 1]:
                                #  随机选择一个错误的token改正确
                                non_eq = (target_tokens[index] != reference[index]) & get_base_mask(reference[index])
                                all_index = []
                                for idx, value in enumerate(non_eq.cpu().tolist()):
                                    if value == True:
                                        all_index.append(idx)

                                idx = all_index[np.random.randint(0, len(all_index))]
                                target_tokens[index][idx] = reference[index][idx]
                        else:
                            target_tokens[index][d + 1] = reference[index][d + 1]

    return target_masks


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def _random_mask(target_tokens, noise_probability=None):
    unk = 3

    target_masks = get_base_mask(target_tokens)
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


def load_dependency_tree(dependency_tree_path, convert=False, add_one=False, scale=2):
    """ convert表示是否弄成损失样式  """
    dependency_list = []

    dependency_index = []
    with open(dependency_tree_path, "r") as f:
        for line in f:
            layers = line.strip('\n').split('\t')
            dependency_list.append([])
            max_value = 0
            for layer in layers:

                if layer == "":
                    dependency_list[-1].append([])
                    continue
                if add_one:
                    c = [int(i) + 1 for i in layer.split(',')]
                else:
                    c = [int(i) for i in layer.split(',')]

                dependency_list[-1].append(c)
                if convert:
                    max_value = max(max_value, max(c))

            if convert:
                dependency_index.append(convert_tree_to_index(dependency_list[-1], max_value, scale))

    if convert:
        return dependency_index

    else:
        return dependency_list


def load_dependency_head_tree(dependency_tree_path, convert=False, add_one=False, scale=2):
    """ convert表示是否弄成损失样式  """
    dependency_list = []

    with open(dependency_tree_path, "r") as f:
        for line in f:
            heads = line.strip().split(',')
            if add_one:
                c = [int(i) + 1 for i in heads]  # add one
            else:
                c = [int(i) for i in heads]

            dependency_list.append(c)

    return dependency_list


def convert_tree_to_index(sentence, max_value, scale):
    result = [0 for _ in range(max_value + 1)]

    result[0] = 1  # add_one
    step = 1 / len(sentence)
    for layer_id, layer in enumerate(sentence):
        for l in layer:
            result[l] = scale - step * layer_id * scale + 1

    return result


def init_global_count_tokens():
    global DIFF_TOKENS
    DIFF_TOKENS = 0
    global ALL_TOKENS
    ALL_TOKENS = 0
    global VALUE1
    VALUE1 = 0
    global VALUE2
    VALUE2 = 0
    global VALUE3
    VALUE3 = 0
    global STEP_CORRECT_TOKENS
    STEP_CORRECT_TOKENS = {}


def set_diff_tokens(value):
    global DIFF_TOKENS
    DIFF_TOKENS += value


def set_value1(value):
    global VALUE1
    VALUE1 += value


def set_value2(value):
    global VALUE2
    VALUE2 += value


def set_value3(value):
    global VALUE3
    VALUE3 += value


def set_all_token(value):
    global ALL_TOKENS
    ALL_TOKENS += value


def set_step_value(step, value):
    global STEP_CORRECT_TOKENS

    # ratio = int(value * 10)
    ratio = value
    pre = STEP_CORRECT_TOKENS.get(step, {}).get(ratio, 0)
    STEP_CORRECT_TOKENS.setdefault(step, {})
    STEP_CORRECT_TOKENS[step].setdefault(ratio, 0)
    STEP_CORRECT_TOKENS[step][ratio] = pre + 1


def get_step_value():
    return STEP_CORRECT_TOKENS


def get_diff_tokens():
    return DIFF_TOKENS


def get_value1():
    return VALUE1


def get_value2():
    return VALUE2


def get_value3():
    return VALUE3


def get_all_tokens():
    return ALL_TOKENS


def get_probability(x, log_x=False):
    """ 计算词频"""
    if isinstance(x, list):
        np_x = np.array(x)
        _sum = np_x.sum()
        np_x = np_x / _sum
    return np_x


def get_phrase_probability(x, log_x=False):
    """ 计算词频"""
    if isinstance(x, list):
        np_x = np.array(x)
        _sum = np_x.sum()
        np_x = np_x / _sum
    return np_x


def compute_kl(l1, l2):
    """
    KL(p|q) p一般是真实分布，q是近似的分布，希望q的分布接近于p的分布，
    loss(x=log(q), y=p)
    """
    l1 = l1.log()
    return F.kl_div(l1, l2, reduction='batchmean')
