# -*- coding:utf-8 -*-
# author:Chris Young
# datetime:2020/5/9 9:47
# project: PyCharm

__author__ = 'Chris Young'

import torch

# 存储相关
model_save_path = "./model/"
model_save_name = "train_model.pt"
data_path = "./data/duilian.txt"
data_save_path = "./data/"
data_file_name = "lang_data.pkl"
img_path = "./img/"
loss_img_path = "./loss_img/"
attn_img_path = "./attn_img/"

# 配置相关
input_name = "en"
target_name = "cn"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
v_ratio=0.2
t_ratio=0.2

# 训练相关
hidden_size = 256
epoch = 20
learning_rate = 0.001
batch_size = 1
max_len = 20
