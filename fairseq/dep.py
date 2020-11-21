# encoding=utf-8


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
    def get_dep_tree(self, valid_subset="valid", only_valid=False, **kwargs):

        if not only_valid:
            train_dependency_tree_child = load_dependency_tree(
                "/home/data_ti5_c/wangdq/data/distill/iwslt16_en_de/dependency_child.train.log",
                add_one=True
            )  # 节点index index从0开始计数，孩子节点编号从1开始计数。
        else:
            train_dependency_tree_child = None

        valid_dependency_tree_child = load_dependency_tree(
            "/home/data_ti5_c/wangdq/data/distill/iwslt16_en_de/dependency_child." + str(
                valid_subset) + ".log",
            add_one=True
        )
        return train_dependency_tree_child, valid_dependency_tree_child


class DepLayerTree(DepTree):
    def get_dep_tree(self, valid_subset="valid", only_valid=False, **kwargs):

        if not only_valid:
            train_dependency_tree_child = load_dependency_tree(
                "/home/data_ti5_c/wangdq/data/distill/iwslt16_en_de/dependency.train.log",
                add_one=True
            )  # 节点index index从0开始计数，孩子节点编号从1开始计数。
        else:
            train_dependency_tree_child = None

        valid_dependency_tree_child = load_dependency_tree(
            "/home/data_ti5_c/wangdq/data/distill/iwslt16_en_de/dependency." + str(
                valid_subset) + ".log",
            add_one=True
        )
        return train_dependency_tree_child, valid_dependency_tree_child


class DepHeadTree(DepTree):

    def get_dep_tree(self, valid_subset="valid", only_valid=False, **kwargs):

        if not only_valid:
            train_dependency_tree_head = load_dependency_head_tree(
                dependency_tree_path="/home/data_ti5_c/wangdq/data/distill/iwslt16_en_de/dependency_head.train.log",
                add_one=True)  # head[i]=j， 节点i的父节点是节点j，i从0开始（下标） j从1开始。
        else:
            train_dependency_tree_head = None

        valid_dependency_tree_head = load_dependency_head_tree(
            dependency_tree_path="/home/data_ti5_c/wangdq/data/distill/iwslt16_en_de/dependency_head." + str(
                valid_subset) + ".log",
            add_one=True)

        return train_dependency_tree_head, valid_dependency_tree_head


class RelativeDepMat(DepTree):
    def get_dep_tree(self, valid_subset="valid", only_valid=False, **kwargs):

        prefix = "relative_dependency_mat"
        mat_type = kwargs.get('use_dependency_mat_type', 'parent')
        if mat_type == "grandparent":
            prefix = "relative_dependency_mat_grandparent"
        print(prefix)

        self.use_two_class = kwargs.get("use_two_class", False)
        if not only_valid:
            train_relative_dependency_mat = load_relative_tree(
                dependency_tree_path="/home/data_ti5_c/wangdq/data/distill/iwslt16_en_de/" + prefix + ".train.log")
        else:
            train_relative_dependency_mat = None

        valid_relative_dependency_mat = load_relative_tree(
            dependency_tree_path="/home/data_ti5_c/wangdq/data/distill/iwslt16_en_de/" + prefix + "." + str(
                valid_subset) + ".log")

        return train_relative_dependency_mat, valid_relative_dependency_mat

    def get_dependency_mat(self, sample_ids, reference, training=True, contain_eos=True):

        batch_size, seq_len = reference.size()
        dep_tensor = reference.new_zeros(size=reference.size()).unsqueeze(-1).repeat(1, 1, seq_len)  # pad=0

        start = 1
        for index, id in enumerate(sample_ids):
            relative_dep_postion = self.get_one_sentence(id, training)

            # 不相关
            length = reference[index].ne(1).sum(-1).item()  # 比head child多一个eos的位置。  pad=1
            if contain_eos:
                length -= 2
            else:
                length -= 1
            dep_tensor[index][start:start + length, start:start + length].fill_(2)

            # 同一个token = 3
            flag = 3
            if self.use_two_class:
                flag = 1
            same_word_relation = relative_dep_postion[0]
            if len(same_word_relation) > 0:
                for (start_pos, end_pos) in same_word_relation:
                    dep_tensor[index][start_pos: end_pos, start_pos:end_pos] = flag

            # 相关节点=1
            related_relation = relative_dep_postion[1]
            if len(related_relation) > 0:
                for (start1, end1, start2, end2) in related_relation:
                    dep_tensor[index][start1:end1, start2:end2] = 1
                    dep_tensor[index][start2:end2, start1:end1] = 1

        return dep_tensor
