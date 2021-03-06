# -*- coding:utf-8 -*-
# author:Chris Young
# datetime:2020/5/10 14:31
# project: PyCharm

__author__ = 'Chris Young'

from model import *
from plot import show_attention
from utils import *
from preprocess import *
from dataset import DuiLian
import configx


def predict(s):
    duilian = DuiLian(train_mode=False)
    slen = len(s.split(' '))
    s = normalizeCNString(s)
    s = tokenizer(duilian.get_vocab(), s)
    while len(s) < config.max_len:
        s.append(duilian.get_vocab().word2idx["PAD"])

    encoder = Encoder(duilian.get_vocab_size(), configx.hidden_dim, configx.hidden_dim).to(configx.device)
    attn_decoder = AttnDecoderRNN("dot", configx.hidden_dim, configx.hidden_dim, duilian.get_vocab_size(),
                                  dropout_p=0.1).to(configx.device)
    seq2seq = Seq2Seq(encoder, attn_decoder, configx.device, max_length=config.max_len).to(configx.device)
    seq2seq.load_state_dict(torch.load(configx.model_save_path + configx.model_save_name))
    input_tensor = torch.tensor(s, dtype=torch.long).cuda()
    input_tensor = input_tensor.unsqueeze(0)
    decoder_input = Variable(torch.tensor([[SOS_token]])).to(configx.device)

    target_idxs, attentions = seq2seq.predict(input_tensor, decoder_input, configx.max_len)

    input_words = [duilian.get_vocab().idx2word[idx] for idx in input_tensor.cpu().numpy()[0]]
    target_words = duilian.idx2words([idx for idx in target_idxs])

    print_cn(target_words)
    show_attention(input_words[:slen], target_words[:slen], attentions.view(-1, configx.max_len)[:slen, :slen], 1)


if __name__ == "__main__":
    sentence = input()
    predict(sentence)
