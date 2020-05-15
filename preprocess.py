# -*- coding:utf-8 -*-
# author:Chris Young
# datetime:2020/5/2 14:52
# project: PyCharm

__author__ = 'Chris Young'

import random
from io import open

import configx as config
from utils import *

SOS_token = 0
EOS_token = 1


class Vocab():
    def __init__(self):
        self.word2idx = {'SOS': 0, 'EOS': 1, 'UNK': 2, 'PAD': 3}
        self.word2count = {}
        self.idx2word = {0: "SOS", 1: 'EOS', 2: 'UNK', 3: 'PAD'}
        self.vocab_size = 4

    def build_vocab(self, sentence):
        for word in sentence.split():
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.word2count[word] = 1
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
            else:
                self.word2count[word] += 1


def read_data():
    print("Reading lines...")
    pairs = []
    # 读数据并分行
    lines = open(config.data_path, encoding='utf-8').read().strip().split('\n')
    for l in lines:
        temp_l = l.split(' ')
        pair = [normalizeCNString(temp_l[0]), normalizeCNString(temp_l[1])]  # 去掉标点符号
        pairs.append(pair)
        vocab = Vocab()
    return vocab, pairs


# 预处理数据
def prepareData():
    pairs = []
    vocab, pairs_raw = read_data()
    print("Read %s sentencce parirs" % len(pairs_raw))
    print("Trimmed to %s sectence pairs" % len(pairs_raw))
    print("Counting words.....")
    # 创建字典
    for pair in pairs_raw:
        vocab.build_vocab(pair[0])
        vocab.build_vocab(pair[1])
    print("Counted words:")
    print(vocab.vocab_size)
    for pair in pairs_raw:
        up_tokens = tokenizer(vocab, pair[0])  # 上联tokens
        down_tokens = tokenizer(vocab, pair[1])  # 下联tokens
        pairs.append([up_tokens, down_tokens])
    data = {"vocab": vocab,
            "pairs_raw": pairs_raw,
            "pairs": pairs}
    save_file(config.data_save_path, config.data_file_name, data, 'wb')


if __name__ == "__main__":
    prepareData()
    data = load_flile(config.data_save_path + config.data_file_name, 'rb')
    print(random.choice(data['pairs']))
