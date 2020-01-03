import jieba
import pandas as pd
import numpy as np
from gensim.models import FastText, Word2Vec
import warnings
import torch
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from torch import nn

warnings.filterwarnings('ignore')


def Data_Process(traindata, trainlabel):
    """处理所用的数据"""

    # 删除对应的nan行
    dellist = np.where(traindata['title'].isna())[0].tolist()
    data = traindata.drop(dellist)
    data = pd.merge(data, trainlabel, on='id')  # 数据库的连接部分
    title = data['title']

    # 收集词袋，创建对应的w2id表
    maxlength = 0
    wordofbag = set()
    for row in title:
        # 采用jieba分词，就保留
        splitrow = jieba.lcut(row)
        maxlength = max(len(splitrow), maxlength)  # 计算最长的句子长度,便于之后创建使用
        for word in splitrow:
            wordofbag.add(word)

    wordofbag = list(wordofbag)
    id = [i for i in range(len(wordofbag))]
    w2id = dict(zip(wordofbag, id))
    id2w = dict(zip(id, wordofbag))
    return w2id, id2w, wordofbag, data, maxlength


def Transfer_Word(title, w2id, row, column):
    # 将文本转换成对应的数字
    transtitle = np.zeros((row, column))
    for i in range(len(title)):
        splitrow = jieba.lcut(title[i])
        transrow = np.zeros(column)
        for j in range(len(splitrow)):
            transrow[j] = w2id[splitrow[j]]
        transtitle[i] = transrow
    return transtitle


class TextCNN(nn.Module):
    # 待处理和测试
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        label_num = args['label']  # 最后输出的维度
        filter_num = args['num']  # 核的数量
        filter_sizes = args['filter_sizes']  # 核的第二个维度
        vocab_size = args['vocab_size']  # 初始给定维度
        embedding_dim = args['embedding_dim']  # 嵌入后的维度
        seq_len = args['seq_len']

        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 对数字进行嵌入

        # 多个一维的卷积层
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim, out_channels=filter_num, kernel_size=size),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=seq_len - size + 1),  # 在后面前馈的过程再进行池化-->实际不可
            ) for size in filter_sizes])
        # 防止过拟合的dropout层
        self.dropout = nn.Dropout()

        # 池化后的拼接的向量的维度,因为是最大池化，每一个分类器只提供一个数据,这里我们只用一个线性层即可.
        self.relu = nn.ReLU()
        self.Linear = nn.Linear(in_features=filter_num * len(filter_sizes), out_features=label_num)
        self.softmax = nn.LogSoftmax()  # 输出概率向量，但是损失函数需要使用NLLLoss()

    def forward(self, x):
        # 前馈过程
        out = self.embedding(x)
        out = torch.transpose(out, 1, 2)
        out = [conv(out) for conv in self.conv]

        # 池化后进行连接.
        out = torch.cat(out, dim=1)
        out = torch.squeeze(out, 2)  # 从n*6*1转换为二维
        out = self.relu(out)
        out = self.Linear(out)
        # 转换成概率
        out = self.softmax(out)
        return out


if __name__ == '__main__':
    traindata = pd.read_csv("/Users/XYJ/Downloads/Competitions/互联网新闻情感分析/OriginalDataSet/Train_DataSet.csv")
    trainlabel = pd.read_csv("/Users/XYJ/Downloads/Competitions/互联网新闻情感分析/OriginalDataSet/Train_DataSet_Label.csv")
    w2id, id2w, wordofbag, data, maxlength = Data_Process(traindata, trainlabel)

    # 将词语转换成对应的数字表示.
    transtitle = Transfer_Word(data['title'], w2id, len(data), maxlength)

    vocab_size = len(wordofbag)
    args = {
        'label': 3,
        'num': 2,
        'filter_sizes': [3, 4, 5],  # 一共6层
        'vocab_size': vocab_size,
        'embedding_dim': 300,
        'static': True,
        'fine_tuning': True,
        'seq_len': maxlength
    }
    model = TextCNN(args)
    Loss = nn.NLLLoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    label = np.array(data['label'])
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)  # 五折交叉验证
    for train, valid in kfold.split(transtitle, label):
        for epoch in range(10):  # 不等长数据行会产生错误.处理的时候记得整理成同一种
            #训练部分
            traintitle = torch.LongTensor(transtitle[train])  # 传入的应该是Longtensor类型
            trainlabel = torch.LongTensor(label[train])
            prediction = model(traintitle)
            loss = Loss(prediction, trainlabel)
            optimizer.zero_grad()  # 自动求导
            loss.backward()
            optimizer.step()

            #验证部分
            correct = 0
            vailddata = torch.LongTensor(transtitle[valid])
            vaildlabel = torch.LongTensor(label[valid])
            vaildprediction = model(vailddata)
            # 对应的argmax中的1是指第二个维度
            correct = np.mean((torch.argmax(vaildprediction, 1) == vaildlabel).sum().numpy())
            loss = Loss(vaildprediction, vaildlabel)
            print("valid_loss:", loss.data.item(), "ACC:", correct/len(valid))

    # 用我的FastText玩一玩
    # model_ted = FastText(wordofbag, size=100, window=5, min_count=5, workers=4, sg=1)
    # model_1 = Word2Vec(wordofbag, size=100, window=5, min_count=5, workers=4, sg=1)
