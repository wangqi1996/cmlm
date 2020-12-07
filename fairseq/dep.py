# encoding=utf-8
import torch


def load_relative_tree(dependency_tree_path):
    """ 不需要add_one """
    relative_dependency_mat = []

    with open(dependency_tree_path, "r") as f:
        for index, line in enumerate(f):
            relation = line.strip('\n').split('\t')
            relation_list = []
            for r in relation:  # 目前有两种关系
                if r.strip() == "":
                    relation_list.append([])
                    continue
                tuple = r.strip().split(',')
                result = []

                for t in tuple:
                    result.append([int(i) for i in t.strip().split('-')])
                relation_list.append(result)
            relative_dependency_mat.append(relation_list)
    return relative_dependency_mat


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


class DepTree():

    def get_file_dir(self, dep_file):
        if dep_file == "iwslt16":
            return "/home/data_ti5_c/wangdq/data/distill/iwslt16_en_de/"
        if dep_file == "wmt16":
            return "/home/data_ti5_c/wangdq/data/test/wmt16_en_ro/dependency/"

    def __init__(self, valid_subset="valid", use_tree=True, only_valid=False, **kwargs):
        if use_tree:
            self.train_tree, self.valid_tree = self.get_dep_tree(valid_subset, only_valid, **kwargs)
        else:
            self.train_tree, self.valid_tree = None, None

    def get_dep_tree(self, valid_subset, only_valid=False, **kwargs):
        raise Exception("怎么是这个类呢？？？？？")

    def get_one_sentence(self, index, training):
        if training:
            return self.train_tree[index]
        else:
            return self.valid_tree[index]

    def get_sentences(self, index_list, training):
        tree = self.train_tree if training else self.valid_tree
        return [tree[id] for id in index_list]


class DepChildTree(DepTree):
    def get_dep_tree(self, valid_subset="valid", only_valid=False, dep_file="", **kwargs):

        dir_name = self.get_file_dir(dep_file)
        if not only_valid:
            train_dependency_tree_child = load_dependency_tree(
                dir_name + "dependency_child.train.log",
                add_one=True
            )  # 节点index index从0开始计数，孩子节点编号从1开始计数。
        else:
            train_dependency_tree_child = None

        valid_dependency_tree_child = load_dependency_tree(
            dir_name + "dependency_child." + str(
                valid_subset) + ".log",
            add_one=True
        )
        return train_dependency_tree_child, valid_dependency_tree_child


class DepLayerTree(DepTree):
    def get_dep_tree(self, valid_subset="valid", only_valid=False, dep_file="", **kwargs):

        dir_name = self.get_file_dir(dep_file)

        if not only_valid:
            train_dependency_tree_child = load_dependency_tree(
                dir_name + "dependency.train.log",
                add_one=True
            )  # 节点index index从0开始计数，孩子节点编号从1开始计数。
        else:
            train_dependency_tree_child = None

        valid_dependency_tree_child = load_dependency_tree(
            dir_name + "dependency." + str(
                valid_subset) + ".log",
            add_one=True
        )
        return train_dependency_tree_child, valid_dependency_tree_child


class DepHeadTree(DepTree):

    def get_dep_tree(self, valid_subset="valid", only_valid=False, dep_file="", **kwargs):

        dir_name = self.get_file_dir(dep_file)

        if not only_valid:
            train_dependency_tree_head = load_dependency_head_tree(
                dependency_tree_path=dir_name + "dependency_head_2.train.log",
                add_one=True)  # head[i]=j， 节点i的父节点是节点j，i从0开始（下标） j从1开始。
        else:
            train_dependency_tree_head = None

        valid_dependency_tree_head = load_dependency_head_tree(
            dependency_tree_path=dir_name + "dependency_head_2." + str(
                valid_subset) + ".log",
            add_one=True)

        return train_dependency_tree_head, valid_dependency_tree_head


class RelativeDepMat(DepTree):
    def get_dep_tree(self, valid_subset="valid", only_valid=False, args=None, dep_file="iwslt16", **kwargs):

        # mat_type = getattr(args, "use_dependency_mat_type", False)
        prefix = "relative_dependency_mat_grandparent"
        # if mat_type != "grandparent":
        #     prefix ="relative_dependency_mat"
        # print("dep_relative_mat: ", prefix)

        dir_name = self.get_file_dir(dep_file)

        if valid_subset == "test":
            only_valid = True

        self.use_two_class = kwargs.get("use_two_class", False)

        if not only_valid:
            train_relative_dependency_mat = load_relative_tree(
                dependency_tree_path=dir_name + prefix + ".train.log")
        else:
            train_relative_dependency_mat = None

        valid_relative_dependency_mat = load_relative_tree(
            dependency_tree_path=dir_name + prefix + "." + str(
                valid_subset) + ".log")

        train_relative_dependency_mat = self.process_mat(train_relative_dependency_mat)
        valid_relative_dependency_mat = self.process_mat(valid_relative_dependency_mat)

        return train_relative_dependency_mat, valid_relative_dependency_mat

        # return None, None

    def process_mat(self, samples):
        if samples is None:
            return None

        result = []
        for sample in samples:
            sample_len = sample[0][0][0]
            mat = torch.LongTensor(sample_len + 2, sample_len + 2).fill_(0)
            mat[1:sample_len + 1, 1:sample_len + 1] = 2  # 不相关

            flag = 3
            if self.use_two_class:
                flag = 1
            same_word_relation = sample[1]
            if len(same_word_relation) > 0:
                for (start_pos, end_pos) in same_word_relation:
                    mat[start_pos: end_pos, start_pos:end_pos] = flag

            # 相关节点=1
            related_relation = sample[2]
            if len(related_relation) > 0:
                for (start1, end1, start2, end2) in related_relation:
                    mat[start1:end1, start2:end2] = 1
                    mat[start2:end2, start1:end1] = 1

            result.append(mat)
        return result

    def get_dependency_mat(self, sample_ids, reference, training=True, perturb=0.0, **kwargs):

        batch_size, seq_len = reference.size()
        dep_tensor = reference.new_zeros(size=reference.size()).unsqueeze(-1).repeat(1, 1, seq_len)  # pad=0

        def _perturb(label, mask_length):
            co_score = score + (dep_tensor[index][:length][:length] == label).long()
            co_score = co_score.view(-1)
            _, target_rank = co_score.sort(descending=True)
            mask = target_rank.new_zeros(target_rank.size()).bool().fill_(False)
            mask[target_rank[:mask_length]] = True
            return mask.view(length, length)

        for index, id in enumerate(sample_ids):
            relative_dep_postion = self.get_one_sentence(id, training)
            length, _ = relative_dep_postion.size()
            dep_tensor[index][:length, :length] = relative_dep_postion

            # oracle = dep_tensor[index].clone()
            if perturb > 0.0:
                correlation_length = (dep_tensor[index] == 1).sum().item()
                uncorrelation_length = (dep_tensor[index] == 2).sum().item()
                mask_length = round(correlation_length * perturb)
                score = dep_tensor[index][:length][:length].clone().float().uniform_()

                if correlation_length > 0:
                    _mask_length = min(mask_length, correlation_length)
                    _perturb_1 = _perturb(1, _mask_length)  # 1->2
                if uncorrelation_length > 0:
                    _mask_length = min(mask_length, uncorrelation_length)
                    _perturb_2 = _perturb(2, _mask_length)  # 2->1

                if correlation_length > 0:
                    dep_tensor[index].masked_fill_(_perturb_1, 2)
                if uncorrelation_length > 0:
                    dep_tensor[index].masked_fill_(_perturb_2, 1)

            # l, _ = dep_tensor[index].size()
            #
            # name = ["pad", "positive", "negative", "same"]
            # for i in [1, 2]:  # 0 pad 1 相关 2 不相关 3 相似
            #     predict_i = (dep_tensor[index] == i).sum().item()  # 1 是相关
            #     target_i = (oracle == i).sum().item()
            #     correct_i = ((dep_tensor[index] == i) & (oracle == i)).sum().item()
            #     set_key_value("predict_" + name[i], predict_i)
            #     set_key_value("target_" + name[i], target_i)
            #     set_key_value("correct_" + name[i], correct_i)

        return dep_tensor

    def get_random_mat(self, sample_ids, reference, training=True, **kwargs):
        batch_size, seq_len = reference.size()
        dep_tensor = reference.new_zeros(size=reference.size()).unsqueeze(-1).repeat(1, 1, seq_len)  # pad=0
        # print("sample_ids: ", sample_ids)
        for index, id in enumerate(sample_ids):
            ref_len = (reference[index] != 1).sum().item()
            dep_tensor[index][:ref_len, :ref_len] = 2
            # 随机扰动14% = 1
            mask_len = int(ref_len * ref_len * 0.14)
            target_score = dep_tensor[index][:ref_len, :ref_len].clone().float().uniform_()
            target_score = target_score.view(-1)

            _, target_rank = target_score.sort()
            target_cutoff = (target_rank < mask_len).view(ref_len, ref_len)
            dep_tensor[index] = dep_tensor[index].masked_fill(target_cutoff, 1)

        return dep_tensor
