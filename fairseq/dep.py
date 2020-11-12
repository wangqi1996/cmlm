# encoding=utf-8
import torch


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

    def __init__(self, valid_subset="valid", use_tree=True, only_valid=False):
        if use_tree:
            self.train_tree, self.valid_tree = self.get_dep_tree(valid_subset, only_valid)
        else:
            self.train_tree, self.valid_tree = None, None

    def get_dep_tree(self, valid_subset, only_valid=False):
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
    def get_dep_tree(self, valid_subset="valid", only_valid=False):

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
    def get_dep_tree(self, valid_subset="valid", only_valid=False):

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

    def get_dep_tree(self, valid_subset="valid", only_valid=False):

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


def get_dependency_mat(head_tree: DepHeadTree, child_tree: DepChildTree, sample_ids, training,
                       token_tensor: torch.Tensor, contain_eos=False):
    """token_tensor,需要区分AT和NAT,AT移除了bos
    以后可以做成文件流的方式
    head_tree: 多级跳转，寻找parent、grandfather、deep ==> 1,2,3
    child_tree: 多级跳转，寻找child、grandson、deep ==> 4,5,6
    初始化大的batch: 0
    初始化每个句子：length other=1
    output: 和token_tensor同样维度的数据
    """
    root_index = 0
    start = 1

    batch_size, seq_len = token_tensor.size()
    dep_tensor = token_tensor.new_zeros(size=token_tensor.size()).unsqueeze(-1).repeat(1, 1, seq_len) # pad=0

    for index, id in enumerate(sample_ids):
        head = head_tree.get_one_sentence(id, training)
        child = child_tree.get_one_sentence(id, training)

        length = token_tensor[index].ne(1).sum(-1).item()  # 比head child多一个eos的位置。
        if contain_eos:
            length -= 2
        else:
            length -= 1

        dep_tensor[index][start:start + length, start:start + length].fill_(7)  # other节点 去除最后的eos节点

        for i in range(start, length + 1):  # 对应的token_tensor中的节点index
            # 自身节点
            dep_tensor[index][i][i] = 8
            # 父亲节点
            head_index = head[i - 1]
            if head_index != root_index:
                dep_tensor[index][i][head_index] = 1  # 父亲节点

                grandparent_index = head[head_index - 1]  # add one 造成的
                if grandparent_index != root_index:
                    dep_tensor[index][i][grandparent_index] = 2  # 祖父节点

                    # 更深的祖父节点
                    grandparent_index = head[grandparent_index - 1]
                    while grandparent_index != root_index:
                        dep_tensor[index][i][grandparent_index] = 3  # deep节点
                        grandparent_index = head[grandparent_index - 1]

                # 孩子节点
            child_index = child[i - 1]
            for j in child_index:
                dep_tensor[index][i][j] = 4  # 孩子节点
                grandson_index = child[j - 1]
                queue = []
                for k in grandson_index:
                    dep_tensor[index][i][k] = 5  # 孙子节点
                    if len(child[k - 1]) > 0:
                        queue.extend(child[k - 1])
                while len(queue) > 0:
                    grandson_index = queue.pop()
                    dep_tensor[index][i][grandson_index] = 6
                    if len(child[grandson_index - 1]) > 0:
                        queue.extend(child[grandson_index - 1])

    return dep_tensor


def get_coarse_dependency_mat(head_tree: DepHeadTree, child_tree: DepChildTree, sample_ids, training,
                              token_tensor: torch.Tensor, contain_eos=False):
    """token_tensor,需要区分AT和NAT,AT移除了bos
       以后可以做成文件流的方式
       二分类任务：相关1 vs 不相关2  相关只考虑父节点、祖父节点、孩子节点、孙子节点
       初始化大的batch: 0
       初始化每个句子：length other=1
       output: 和token_tensor同样维度的数据
       """
    root_index = 0
    start = 1

    batch_size, seq_len = token_tensor.size()
    dep_tensor = token_tensor.new_zeros(size=token_tensor.size()).unsqueeze(-1).repeat(1, 1, seq_len)  # pad=0

    for index, id in enumerate(sample_ids):
        head = head_tree.get_one_sentence(id, training)
        child = child_tree.get_one_sentence(id, training)

        length = token_tensor[index].ne(1).sum(-1).item()  # 比head child多一个eos的位置。  pad=1
        if contain_eos:
            length -= 2
        else:
            length -= 1

        dep_tensor[index][start:start + length, start:start + length].fill_(2)  # 不想关 2

        for i in range(start, length + 1):  # 对应的token_tensor中的节点index
            # 自身节点
            dep_tensor[index][i][i] = 1
            # 父亲节点
            head_index = head[i - 1]
            if head_index != root_index:
                dep_tensor[index][i][head_index] = 1  # 父亲节点

                grandparent_index = head[head_index - 1]  # add one 造成的
                if grandparent_index != root_index:
                    dep_tensor[index][i][grandparent_index] = 1  # 祖父节点

                # 孩子节点
            child_index = child[i - 1]
            for j in child_index:
                dep_tensor[index][i][j] = 1  # 孩子节点
                grandson_index = child[j - 1]
                for k in grandson_index:
                    dep_tensor[index][i][k] = 1  # 孙子节点

    return dep_tensor

# def get_random_dependency_mat(head_tree: DepHeadTree, child_tree: DepChildTree, sample_ids, training,
#                               token_tensor: torch.Tensor, contain_eos=False):
#     """token_tensor,需要区分AT和NAT,AT移除了bos
#        以后可以做成文件流的方式
#        二分类任务：相关1 vs 不相关2  相关只考虑父节点、祖父节点、孩子节点、孙子节点
#        初始化大的batch: 0
#        初始化每个句子：length other=1
#        output: 和token_tensor同样维度的数据
#        """
#     root_index = 0
#     start = 1
#
#     batch_size, seq_len = token_tensor.size()
#     dep_tensor = token_tensor.new_zeros(size=token_tensor.size()).unsqueeze(-1).repeat(1, 1, seq_len)
#
#     for index, id in enumerate(sample_ids):
#         head = head_tree.get_one_sentence(id, training)
#         child = child_tree.get_one_sentence(id, training)
#
#         length = token_tensor[index].ne(1).sum(-1).item()  # 比head child多一个eos的位置。
#         if contain_eos:
#             length -= 2
#         else:
#             length -= 1
#
#         dep_tensor[index][start:start + length, start:start + length].fill_(2)  # 不想关 2
#
#         for i in range(start, length + 1):  # 对应的token_tensor中的节点index
#             # 自身节点
#             dep_tensor[index][i][i] = 1
#             # 父亲节点
#             head_index = head[i - 1]
#             if head_index != root_index:
#                 dep_tensor[index][i][head_index] = 1  # 父亲节点
#
#                 grandparent_index = head[head_index - 1]  # add one 造成的
#                 if grandparent_index != root_index:
#                     dep_tensor[index][i][grandparent_index] = 1  # 祖父节点
#
#                 # 孩子节点
#             child_index = child[i - 1]
#             for j in child_index:
#                 dep_tensor[index][i][j] = 1  # 孩子节点
#                 grandson_index = child[j - 1]
#                 for k in grandson_index:
#                     dep_tensor[index][i][k] = 1  # 孙子节点
#
#     return dep_tensor
