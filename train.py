# -*- coding:utf-8 -*-
# author:Chris Young
# datetime:2020/5/2 14:53
# project: PyCharm

__author__ = 'Chris Young'

import argparse

from nltk.translate.bleu_score import corpus_bleu

from dataset import *
from model import *
from plot import *
from preprocess import *
import configx as cfg
import blue


# input_tensor(B,L) target_tensor(B,L)
def train(input_tensor, target_tensor, seq2seq, encorder_optimizer, decoder_optimizer, criterion,
          ):
    encorder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    decoder_input = Variable(torch.tensor([[SOS_token]])).to(device)
    # last encoder hidden

    outputs = seq2seq(input_tensor, target_tensor, decoder_input)  # (L,B,O)
    loss = criterion(outputs.view(-1, outputs.shape[2]), target_tensor.view(-1))
    loss.backward()
    encorder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def trainIters(encoder, decoder, seq2seq, train_set, valid_set, epoch, batch_size, print_every=1000, plot_every=50,
               lr=0.01):
    seq2seq.train()
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    n_epoch = 0
    count = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr)

    criterion = nn.CrossEntropyLoss()

    print("start training epoch {}".format(n_epoch + 1))
    while n_epoch != epoch:
        b = train_set.next_batch(batch_size)
        if b['new_epoch']:
            showPlot(plot_losses,n_epoch)
            evaluate(valid_set, seq2seq)
            seq2seq.train()
            n_epoch += 1
            coutn = 0
            check_save_path(configx.model_save_path)
            torch.save(seq2seq.state_dict(), configx.model_save_path + configx.model_save_name)
            print("start training epoch {}".format(n_epoch + 1))

        input_tensor, target_tensor = get_intput(b)

        loss = train(input_tensor, target_tensor, seq2seq, encoder_optimizer, decoder_optimizer,
                     criterion)
        # print(loss)
        print_loss_total += loss
        plot_loss_total += loss

        count += batch_size
        if count % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, (n_epoch + 1) / epoch),
                                         count, (n_epoch + 1) / epoch * 100, print_loss_avg))

        if count % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    evaluate(test_set, seq2seq, plot=True)


def evaluate(valid_set, seq2seq, max_length=20, plot=False):
    seq2seq.eval()
    score = 0
    total_num = valid_set.get_size()
    while True:
        b = valid_set.next_batch(1)
        if b['new_epoch']:
            break
        input_tensor, target_tensor = get_intput(b)
        decoder_input = Variable(torch.tensor([[SOS_token]])).to(device)
        target_idxs, attentions = seq2seq.predict(input_tensor, decoder_input, max_length)
        pre_words = []
        for i in range(len(target_idxs)):
            idx = target_idxs[i]
            word = duilian.get_vocab().idx2word[idx]
            pre_words.append(word)
        tar_words = [duilian.get_vocab().idx2word[idx] for idx in target_tensor.cpu().numpy()[0]]
        input_words = [duilian.get_vocab().idx2word[idx] for idx in input_tensor.cpu().numpy()[0]]
        score_t, _, _, _, _, _ = blue.compute_bleu([b['y'].tolist()], [target_idxs], 2)
        score += score_t
        if plot:
            show_attention(input_words[:b['xl'][0]], pre_words[:b['yl'][0]], attentions.view(-1, max_length),0)
    print("AVG BLEU Score is : %.4f" % (score / total_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', required=True)
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    duilian = DuiLian(max_len=cfg.max_len, v_ratio=cfg.v_ratio, t_ratio=cfg.t_ratio, vocab_size=cfg.vocab_size)
    train_set = duilian.train
    valid_set = duilian.valid
    test_set = duilian.test
    encoder = Encoder(duilian.get_vocab_size(), cfg.embedding_dim, cfg.hidden_dim).to(device)
    attn_decoder = AttnDecoderRNN("dot", cfg.embedding_dim, cfg.hidden_dim, duilian.get_vocab_size(),
                                  dropout_p=cfg.dropout_p, max_length=cfg.max_len).to(
        device)
    seq2seq = Seq2Seq(encoder, attn_decoder, device, max_length=cfg.max_len).to(device)

    trainIters(encoder, attn_decoder, seq2seq, train_set, valid_set, cfg.epoch, cfg.batch_size,
               print_every=cfg.print_every, plot_every=cfg.plot_every,
               lr=cfg.learning_rate)
