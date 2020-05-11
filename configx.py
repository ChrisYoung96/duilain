# -*- coding:utf-8 -*-
# author:Chris Young
# datetime:2020/5/9 9:47
# project: PyCharm

__author__ = 'Chris Young'

import torch

# 存储相关
model_save_path = "./model/"  # 模型保存路径
model_save_name = "train_model.pt"  # 模型文件名
data_path = "./data/duilian.txt"  # 数据路径
data_save_path = "./data/"  # 预处理后的数据保存路径
data_file_name = "lang_data.pkl"  # 预处理后的数据名称
img_path = "./img/"  # 图像输出路径
loss_img_path = "./loss_img/"  # 损失曲线图像输出路径
attn_img_path = "./attn_img/"  # 注意力图输出路径

# 配置相关
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用cpu还是gpu

# 数据集相关
v_ratio = 0.2  # 验证集比重
t_ratio = 0.2  # 测试集比重

# 模型相关
max_len = 20  # 序列（句子）最大长度
hidden_dim = 256  # 隐藏层维度
embedding_dim = 256  # 词嵌入层维度
dropout_p = 0.1  # dropout概率

# 训练相关
epoch = 20  # 训练轮数
learning_rate = 0.001  # 学习率
batch_size = 1  # 一批数据大小
print_every = 1000  # 打印数据间隔，用于保存需要打印的数据
plot_every = 500  # 绘图数据间隔，用于保存需要绘图的数据
