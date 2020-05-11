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


def unicodeToAscii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeEnString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", "", s)
    return s


def normalizeCNString(s):
    s = s.lower().strip()
    s = re.sub(r"([。！？])", "", s)
    return s


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def check_save_path(save_path):
    is_exist = os.path.exists(save_path)
    if not is_exist:
        os.makedirs(save_path)
        print("Save path created!")
    else:
        print("Save path is existed")


def save_file(save_path, file_name, data, write_way='w'):
    check_save_path(save_path)
    with open(save_path + file_name, write_way) as f:
        pickle.dump(data, f)
    print("File is Saved at " + save_path + file_name)


def load_flile(file_path, read_way='r', encoding=''):
    if encoding == '':
        with open(file_path, read_way) as f:
            data = pickle.load(f)
    else:
        with open(file_path, read_way, encoding=encoding) as f:
            data = pickle.load(f)
    return data


def idxFromSentence(lang, sentence):
    idxs = []
    for word in sentence:
        if word not in lang.word2idx:
            idxs.append(lang.word2idx['UNK'])
        else:
            idxs.append(lang.word2idx[word])
    idxs.append(lang.word2idx['EOS'])
    return idxs


def get_intput(batch):
    xs = batch['x']
    ys = batch['y']
    input = torch.tensor(xs, dtype=torch.long).cuda()
    output = torch.tensor(ys, dtype=torch.long).cuda()
    return input, output


def print_cn(target_words):
    words = ""
    for word in target_words:
        if word == "EOS":
            break
        elif word != "PAD":
            words += word

    print(words)
