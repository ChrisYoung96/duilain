# -*- coding:utf-8 -*-
# author:Chris Young
# datetime:2020/5/10 14:31
# project: PyCharm

__author__ = 'Chris Young'

from model import *
from utils import *
from preprocess import *
from dataset import DuiLian
import configx


def predict(sentence):
    en_cn = DuiLian(train_mode=False)
    s = normalizeEnString(sentence)
    sentence = idxFromSentence(en_cn.get_input_lang(), s)
    while len(sentence) < config.max_len:
        sentence.append(en_cn.get_input_lang().word2idx["PAD"])

    encoder = Encoder(en_cn.get_input_features(), configx.hidden_size, configx.hidden_size).to(configx.device)
    attn_decoder = AttnDecoderRNN("dot", configx.hidden_size, configx.hidden_size, en_cn.get_vocab_size(),
                                  dropout_p=0.1).to(
        configx.device)
    seq2seq = Seq2Seq(encoder, attn_decoder, configx.device).to(configx.device)
    seq2seq.load_state_dict(torch.load(configx.model_save_path + configx.model_save_name))
    input_tensor = torch.tensor(sentence, dtype=torch.long).cuda()
    input_tensor = input_tensor.unsqueeze(0)
    decoder_input = Variable(torch.tensor([[SOS_token]])).to(configx.device)
    target_idxs, _ = seq2seq.predict(input_tensor, decoder_input)
    target_words = en_cn.idx2words(configx.target_name, [idx for idx in target_idxs])
    print_cn(target_words)


if __name__ == "__main__":
    sentence = input()
    predict(sentence)
