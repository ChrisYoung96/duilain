# -*- coding:utf-8 -*-
# author:Chris Young
# datetime:2020/5/2 15:14
# project: PyCharm

__author__ = 'Chris Young'

import jieba
import torch
import unicodedata
import re
import time
import math
import os
import pickle
import configx

'''
工具模型快
'''

# 去除中文中的标点符号
def normalizeCNString(s):
    s = s.lower().strip()
    s = re.sub(r"([，。！？])", "", s)
    return s

# 时间工具
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# 计时工具
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# 检查存储路径，若路径不存在创建路径
def check_save_path(save_path):
    is_exist = os.path.exists(save_path)
    if not is_exist:
        os.makedirs(save_path)
        print("Save path created!")
    else:
        print("Save path is existed")

# 保存文件
def save_file(save_path, file_name, data, write_way='w'):
    check_save_path(save_path)
    with open(save_path + file_name, write_way) as f:
        pickle.dump(data, f)
    print("File is Saved at " + save_path + file_name)

# 读取文件
def load_flile(file_path, read_way='r', encoding=''):
    if encoding == '':
        with open(file_path, read_way) as f:
            data = pickle.load(f)
    else:
        with open(file_path, read_way, encoding=encoding) as f:
            data = pickle.load(f)
    return data

# 将句子离散化，即将句子中的字（词）转换为字典中的下标
def tokenizer(vocab, sentence):
    idxs = []
    for word in sentence.split(' '):
        if word not in vocab.word2idx:
            idxs.append(vocab.word2idx['UNK'])
        else:
            idxs.append(vocab.word2idx[word])
    idxs.append(vocab.word2idx['EOS'])
    return idxs

# 获取模型的输入
def get_intput(batch):
    xs = batch['x']
    ys = batch['y']
    input = torch.tensor(xs, dtype=torch.long).cuda()
    output = torch.tensor(ys, dtype=torch.long).cuda()
    return input, output

# 输出中文
def print_cn(target_words):
    words = ""
    for word in target_words:
        if word == "EOS":
            break
        elif word != "PAD":
            words += word

    print(words)
