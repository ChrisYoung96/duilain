# -*- coding:utf-8 -*-
# author:Chris Young
# datetime:2020/5/2 17:21
# project: PyCharm

__author__ = 'Chris Young'

'''
绘图模块
'''

import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from utils import *
import  configx


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()
    check_save_path(configx.img_path + configx.loss_img_path)
    plt.savefig(configx.img_path + configx.loss_img_path + "loss.png")


def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['EOS'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    check_save_path(configx.img_path+configx.attn_img_path)
    plt.savefig(configx.img_path+configx.attn_img_path+"attn.png")
    plt.close()
