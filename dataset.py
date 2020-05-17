# -*- coding:utf-8 -*-
# author:Chris Young
# datetime:2020/5/2 16:36
# project: PyCharm

__author__ = 'Chris Young'

import copy
import random
from preprocess import Vocab
import numpy

import configx as cfg
from utils import load_flile

SOS_token = 0
EOS_token = 1


# 数据集类
# 初始化参数说明：
#       input_lang:输入语言的字典对象
#       output_lang:输出语言的字典对象
#       pairs:语句对（tokenized）
#       pairs_raw:语句对原文
#       ids:pairs中所有语句对在原pairs中的下标
#       max_len:序列最大长度
#       dtype:数据类型
class Dataset(object):
    def __init__(self, vocab, pairs, pairs_raw, ids, max_len=30, vocab_size=10000, dtype=None):
        self.__vocab = vocab
        self.__dtype = dtype
        self.__max_len = max_len
        if self.__vocab.vocab_size < vocab_size:
            self.__vocab_size = self.__vocab.vocab_size
        else:
            self.__vocab_size = vocab_size
        self.__ids = []
        self.__xs = []  # 输入数据集
        self.__ys = []  # 输出数据集
        self.__xraws = []  # 输入数据原文
        self.__yraws = []  # 输出数据原文
        self.__xls = []  # 每个输入序列的真实长度
        self.__yls = []  # 每个输出序列的真实长度
        assert len(pairs) == len(ids)
        # 生成输入、输出数据集
        for p, r, i in zip(pairs, pairs_raw, ids):
            x = p[0]
            y = p[1]
            xraw = r[0]
            yraw = r[1]
            self.__ids.append(i)
            # 如果输入序列长度大于最大长度，则截取
            if len(x) > self.__max_len:
                x = x[:self.__max_len]
                self.__xls.append(self.__max_len)
            else:
                self.__xls.append(len(x))
            # 将处理后的输入序列加入输入数据集
            self.__xs.append([])
            for t in x:
                if t >= self.__vocab_size:
                    self.__xs[-1].append(self.__vocab.word2idx['UNK'])
                else:
                    self.__xs[-1].append(t)

            # 如果输入序列的长度小于最大长度，则做padding，用pad符补齐
            while len(self.__xs[-1]) < self.__max_len:
                self.__xs[-1].append(self.__vocab.word2idx["PAD"])

            if len(y) > self.__max_len:
                self.__yls.append(self.__max_len)
                y = y[:self.__max_len]
            else:
                self.__yls.append(len(y))
            self.__ys.append([])
            for t in y:
                if t >= self.__vocab_size:
                    self.__ys[-1].append(self.__vocab.word2idx['UNK'])
                else:
                    self.__ys[-1].append(t)

            while len(self.__ys[-1]) < self.__max_len:
                self.__ys[-1].append(self.__vocab.word2idx["PAD"])
            # 获取输入和输出的原文数据
            self.__xraws.append(xraw)
            self.__yraws.append(yraw)
        # 将输入、输出等转换为numpy数组
        self.__xs = numpy.asarray(self.__xs, dtype=self.__dtype['int'])
        self.__ys = numpy.asarray(self.__ys, dtype=self.__dtype['int'])
        self.__ids = numpy.asarray(self.__ids, dtype=self.__dtype['int'])
        self.__size = len(self.__xs)  # 获取数据集大小

        # 如果所有数据集长度相同，则进行最后一步初始化工作
        assert self.__size == len(self.__xs) and len(self.__xs) == len(self.__ys) and len(self.__ys) == len(
            self.__yls) and len(self.__yls) == len(self.__xls) and len(self.__xls) == len(self.__ids)

        self.__epoch = None
        self.reset_epoch()

    # 为训练服务，重置未被使用的数据的下标列表
    def reset_epoch(self):
        self.__epoch = random.sample(range(self.__size), self.__size)  # 随机生成该数据集未被使用的数据的下标idx列表

    # 获取下一批数据
    def next_batch(self, batch_size):
        # batch字典用于存储每一批数据
        batch = {'x': [], 'y': [], 'xl': [], 'yl': [], 'id': [], 'new_epoch': False}
        assert batch_size <= self.__size
        # 如果未被使用的数据量小于一批数据的大小，则开始新的一轮
        if len(self.__epoch) < batch_size:
            batch['new_epoch'] = True
            self.reset_epoch()
        # 从未被使用的数据的下标列表中顺序截取batch_size个数据
        idx = self.__epoch[:batch_size]
        # 未被使用的数据的下标列表中减少batch_size个数据
        self.__epoch = self.__epoch[batch_size:]
        # 根据idx使用numpy直接获取相应数据信息，构成这批数据
        batch['x'] = numpy.take(self.__xs, indices=idx, axis=0)
        batch['y'] = numpy.take(self.__ys, indices=idx, axis=0)
        batch['xl'] = numpy.take(self.__xls, indices=idx, axis=0)
        batch['yl'] = numpy.take(self.__yls, indices=idx, axis=0)
        batch['xraw'] = numpy.take(self.__xraws, indices=idx, axis=0)
        batch['yraw'] = numpy.take(self.__yraws, indices=idx, axis=0)
        batch['id'] = numpy.take(self.__ids, indices=idx, axis=0)
        return batch

    def get_size(self):
        return self.__size

    def get_raws(self):
        return copy.deepcopy(self.__xraws), copy.deepcopy(self.__yraws)


# 数据集类
# 对预处理后的数据进行划分，生成训练集、验证集和测试集
# 初始化参数说明：
#       path：经过预处理以后的数据文件路径
#       max_len:序列的最大长度
#       v_ratio:验证数据集所占比例
#       t_ratio:测试数据集所占比例
#       dtype:数据类型
#       train_model:训练标识。若为True，表示需要生成数据集；若为False，则表示数据集对象只包含字典，用于使用模型进行预测
#       vocab_size:字典大小
class DuiLian(object):
    def __init__(self, path=cfg.data_save_path + cfg.data_file_name, max_len=20, v_ratio=0.2, t_ratio=0.2, dtype='32',
                 train_mode=True, vocab_size=10000):
        self.__dtypes = self.__dtype(dtype)
        self.__max_len = max_len
        data = load_flile(path, 'rb')  # 加载预处理后的数据
        self.__vocab = data['vocab']  # 获取输出语言的字典
        self.__vocab_size = vocab_size

        # 如果为训练模式，准备训练集、测试集和验证集
        if train_mode:
            # 获取数据集中所有的语句对
            self.__pairs = data['pairs']
            self.__pairs_raw = data['pairs_raw']
            assert len(self.__pairs) == len(self.__pairs_raw)
            total_num = len(self.__pairs)  # 获取语句对的数量
            idxs = random.sample(range(total_num), total_num)  # 随机生成语句对下标列表，用于后序生成数据集
            # 根据比例计算各数据集的数据量
            n_valid = int(total_num * v_ratio)
            n_test = int(total_num * t_ratio)
            n_train = total_num - n_valid - n_test

            # 根据训练集、测试集、验证集的数据量生成数据集
            # 生成训练集
            pairs, ids, raws = [], [], []
            for i in idxs[:n_train]:
                pairs.append(self.__pairs[i])
                ids.append(i)
                raws.append(self.__pairs_raw[i])
            self.train = Dataset(self.__vocab,
                                 pairs,
                                 raws,
                                 ids,
                                 self.__max_len,
                                 self.__vocab_size,
                                 self.__dtypes)
            # 生成验证集
            pairs, ids, raws = [], [], []
            for i in idxs[n_train:n_train + n_valid]:
                pairs.append(self.__pairs[i])
                ids.append(i)
                raws.append(self.__pairs_raw[i])
            self.valid = Dataset(self.__vocab,
                                 pairs,
                                 raws,
                                 ids,
                                 self.__max_len,
                                 self.__vocab_size,
                                 self.__dtypes)
            # 生成测试集
            pairs, ids, raws = [], [], []
            for i in idxs[n_train + n_valid:]:
                pairs.append(self.__pairs[i])
                ids.append(i)
                raws.append(self.__pairs_raw[i])
            self.test = Dataset(self.__vocab,
                                pairs,
                                raws,
                                ids,
                                self.__max_len,
                                self.__vocab_size,
                                self.__dtypes)

    def __dtype(self, dtype='32'):

        assert dtype in ['16', '32', '64']
        if dtype == '16':
            return {'fp': numpy.float16, 'int': numpy.int16}
        elif dtype == '32':
            return {'fp': numpy.float32, 'int': numpy.int32}
        elif dtype == '64':
            return {'fp': numpy.float64, 'int': numpy.int64}

    def get_vocab(self):
        return copy.deepcopy(self.__vocab)

    # 获取输出语言的字典大小
    def get_vocab_size(self):
        return self.__vocab_size

    # 使用字典根据id转换成word
    def idx2words(self, idxs):
        return [self.__vocab.idx2word[idx] for idx in idxs]

    # 使用字典根据word转换成id
    def words2idxs(self, words):
        idxs = []

        for word in words:
            if word not in self.__vocab.word2idx:
                idxs.append(self.__vocab.word2idx['UNK'])
            else:
                idxs.append(self.__vocab.word2idx[word])
        idxs.append(self.__vocab.word2idx['EOS'])

        return idxs


if __name__ == "__main__":
    data = DuiLian()
    train = data.train
    test = data.test
    valid = data.valid
    print("train set size is:{}".format(train.get_size()))
    print("valid set size is:{}".format(valid.get_size()))
    print("test set size is:{}".format(test.get_size()))
    for i in range(6):
        b = train.next_batch(1)
        if b['new_epoch']:
            print("new epoch")
