# -*- coding:utf-8 -*-
# author:Chris Young
# datetime:2020/5/2 14:52
# project: PyCharm

__author__ = 'Chris Young'

import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

'''
维度表示符号：
B:Batch_Size
S:Sequence_length
H:Hidden_dim
E:Embedding_dim
L:N_layers
I:Input_size
O:Output_size
'''
SOS_token = 0
EOS_token = 1


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hiddeng_dim, n_layers=1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.n_layers = n_layers
        self.hidden_dim = hiddeng_dim

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.GRU(hiddeng_dim, hiddeng_dim, n_layers)  # 输入形式（S,B,H),输出形式（S,B,biderction*H)

    # inputs(B,S),hidden(L,B,H)
    def forward(self, inputs, hidden):
        seq_len = len(inputs[0])
        embedded = self.embedding(inputs)  # (B,S,H)
        embedded = embedded.view(seq_len, 1, -1)  # (S,B,H) 将tensor转置，满足gru输入
        output, hidden = self.gru(embedded, hidden)  # (S,B,H)
        return output, hidden

    # 初始化0时刻的hidden_state
    def initHidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_dim, device=torch.device("cuda")))


# attention层
# 用于计算Encoder的每一个hidden_state的权重
# 参数说明:
#   method:评分方法：dot，general，concat
#   hidden_dim:隐藏层维度
class Attn(nn.Module):
    def __init__(self, method, hidden_dim):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_dim = hidden_dim

        if self.method == 'general':  # 如果为general模式，需要过一层线性层，来乘参数w
            self.attn = nn.Linear(self.hidden_dim, self.hidden_dim)
        # 如果为concat模式，需要初始化参数v以及线性层，其中cancat模式将Encoder的hidden_state与decoder的hidden_state进行拼接，
        # 因此结构为（hidden_dim*2，hidden_dim)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_dim))

    # 前向传播
    # dicoder_hidden(1,H),encoder_outputs(S,B,H)
    def forward(self, decoder_hidden, encorder_ouputs):
        # 获取序列（语句）长度
        seq_len = len(encorder_ouputs)
        # 记录Encoder的每个hidden_state的attention权重
        attn_energies = Variable(torch.zeros(seq_len)).cuda()  # (S)
        # 对Encoder的每个hidden_state进行打分
        for i in range(seq_len):
            attn_energies[i] = self.score(decoder_hidden[0], encorder_ouputs[i].squeeze(0))
        return F.softmax(attn_energies, dim=0).unsqueeze(0).unsqueeze(0)  # (1,1,S)

    # attention机制中的打分函数
    # 输入为t时刻的decoder的hidden_state和Encoder的一个hidden_state
    def score(self, decoder_hidden, encoder_output):
        attn_score = 0
        # 如果打分方式为dot，则直接点乘
        if self.method == 'dot':
            attn_score = decoder_hidden.dot(encoder_output)
        # 如果打分方式为general，则将Encoder的hidden_state过线性层来乘权重w，
        # 再和decoder的hidden_state点乘
        elif self.method == 'general':
            attn_score = self.attn(encoder_output)
            attn_score = decoder_hidden.dot(attn_score)
        # 如果打分方式为concat
        # 则先将Encoder和Decoder的hidden_state拼接在一起，然后过线性层乘以权重w
        # 再和参数v进行点乘
        elif self.method == 'concat':
            attn_score = torch.tanh(self.attn(torch.cat((decoder_hidden, encoder_output), 1)))
            attn_score = self.v.dot(attn_score)
        return attn_score


# 带Attention的Decoder
# 其结构包括Embedding层、gru层、attention层、线性层
# 初始化参数说明：
#       attn_model:attention层
#       embedding_dmi:embedding层的维度
#       hidden_dim:隐藏层的维度
#       output_size:输出语言字典大小
#       n_layers:gru层数
#       drop_out:丢失率
#       max_len:序列最大长度
class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding_dim, hidden_dim, output_size, n_layers=1, dropout_p=0.1, max_length=20):
        super(AttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.embedding_size = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        # GRU接收的输入是embedding和t-1时刻的contex_vecoter的拼接，因此形状为
        # (hidden_dim+embedding_size,hidden_dim)
        self.gru = nn.GRU(self.hidden_dim + self.embedding_size, self.hidden_dim, n_layers)
        self.dropout = nn.Dropout(self.dropout_p)
        # 因在每个时间步t将通过attention得到的context_vector与decoder的hidden_state连接
        # 所以线性层的形状为（hidden_dim*2,output_size)
        self.out = nn.Linear(self.hidden_dim * 2, self.output_size)
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_dim)

    # 前向传播，输入参数形状为
    # d_input(1),last_hidden(B,1,H),last_context(1,H),encoder_outputs(S,B,H)
    # Decoder的前向传播过程为：
    #   输入目标语言（输出语言）的起始符号
    #   经过Embedding层，获取Embedding并转换为形状
    #   将Embedding和上一时刻的context_vector进行拼接,输入GRU层,得到hidden_state
    #   将hidden_state和Encoder的所有hidden_states输入attention层计算注意力权重attention_weights
    #   将得到的attention_weights与Encoder的hidden_states相乘，便得到了context_vecoter
    #   将context_vector与decoder在t时刻的hidden_state串起来，过线性层，在经过softmax函数便可的到Decoder的输出
    def forward(self, d_input, last_hidden, last_context, encorder_outputs):
        embedded = self.embedding(d_input).view(1, 1, -1)  # (1,B,H)
        decoder_input = torch.cat((embedded, last_context.unsqueeze(0)), 2)  # (B,1,H+E)
        output, hidden = self.gru(decoder_input, last_hidden)  # (B,1,H)
        output = self.dropout(output)  # (B,1,H)
        attn_weights = self.attn(output.squeeze(1), encorder_outputs)  # (B,1,S)
        context = attn_weights.bmm(encorder_outputs.transpose(0, 1))  # (B,1,H)

        output = output.squeeze(1)  # (B,H)
        context = context.squeeze(1)  # (B,H)
        output = F.log_softmax(self.out(torch.cat((output, context), 1)), dim=1)  # (B,O)
        return output, context, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=torch.device("cuda"))


# seq2seq模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, max_length=20, teacher_forcing=0.3):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing

        self.encoder = encoder
        self.decoder = decoder

    # 前向传播过程
    #       初始化0时刻Encoder的hidden_state
    #       获取要输出的语句的长度
    #       将输入input输入到Encoder中，获取Encoder的hidden_states
    #       decoder在0时刻的输入包括目标语言的第一个字符、
    #       Encoder最后一个时间步t的hidden_state以及context_vecoter
    #       根据输出的语句长度，将decoder_input,context_vector和Encoder的hidden_states输入Decoder，得到输出
    #       需要注意的是，我们设置的teacher_forcing，即decoder在t时刻的输入可以为t-1时刻得到的结果或者目标输出在t-1时刻的字符
    def forward(self, input_tensors, target_tensors, decoder_input):
        encoder_hidden = self.encoder.initHidden()
        target_length = target_tensors.size(1)

        encoder_outputs, encoder_hidden = self.encoder(input_tensors, encoder_hidden)
        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_dim)).to(self.device)
        decoder_hidden = encoder_hidden
        # 随机决定是否使用t时刻的目标语句的字符作为Decoder的输入
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        outputs = torch.zeros(self.max_length, 1, self.decoder.output_size).to(self.device) # 存储decoder所有时间步的输出
        decoder_contexts = torch.zeros(self.max_length, 1, self.decoder.hidden_dim).to(self.device)
        decoder_hiddens = torch.zeros(self.max_length, 1, self.decoder.hidden_dim).to(self.device)
        attn_weights = torch.zeros(self.max_length, 1, 1, self.max_length).to(self.device)
        # 根据目标语句长度，对每个位置的词进行预测
        for t in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(
                decoder_input,
                decoder_hidden,
                decoder_context,
                encoder_outputs
            )
            outputs[t] = decoder_output
            decoder_contexts[t] = decoder_context
            decoder_hiddens[t] = decoder_hidden
            attn_weights[t] = decoder_attention

            # 如果使用，则t+1时刻的输入为目标语句在t时刻的词
            if use_teacher_forcing:
                decoder_input = target_tensors[0][t].view(1)
            # 如果不适用，则t+1时刻的输入为t时刻预测的词
            else:
                topi = decoder_output.max(1)[1]
                decoder_input = topi
                if topi == EOS_token:
                    break
        return outputs # (S,1,O)

    # 预测，根据输入语句，目标语句的第一个字符，以及最长序列长度预测翻译的语句
    def predict(self, input_tensors, decoder_input, max_length=20):
        encoder_hidden = self.encoder.initHidden()
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.encoder(input_tensors, encoder_hidden)
            decoder_context = Variable(torch.zeros(1, self.decoder.hidden_dim)).to(self.device)
            decoder_hidden = encoder_hidden
            outputs = torch.zeros(self.max_length, 1, self.decoder.output_size).to(self.device)
            attn_weights = torch.zeros(self.max_length, 1, 1, self.max_length).to(self.device)
            tar_idxs = []
            for t in range(max_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input,
                    decoder_hidden,
                    decoder_context,
                    encoder_outputs
                )
                outputs[t] = decoder_output
                attn_weights[t] = decoder_attention
                tar_idx = decoder_output.max(1)[1]
                tar_idxs.append(tar_idx.cpu().numpy()[0])
                decoder_input = tar_idx
                if tar_idx == EOS_token:
                    break

        return tar_idxs, attn_weights
