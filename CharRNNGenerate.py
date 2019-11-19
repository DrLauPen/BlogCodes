import random
from _tkinter import _flatten

import jieba
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch import nn
import numpy as np
import warnings

from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings('ignore')


def LoadData(filepath, type):
    # 载入数据集
    data = []
    with open(filepath, type) as f:
        for line in f:
            line = line.replace(' ', '')
            line = line.replace('\n', '')
            line = line.replace('，', '')
            line = line.replace('。', '')
            data.append(line)
    return data

def DataProcess(data):
    """
    :param data:
    :return: 处理过后的词
    """
    SplitLine = []
    BagOfWords = []
    for line in data:#居然不需要精确划分词句.....
        # line = jieba.lcut(line)  # 精确模式分割诗句
        # line = " ".join(line).split(" ")
        line = list(line)
        SplitLine.append(line)
        BagOfWords += line
    BagOfWords = set(BagOfWords)

    # 将词语转换成对应的数值
    NumOfWord = [[_] for _ in range(len(BagOfWords))]
    ec2 = OneHotEncoder()
    NumOfVec = ec2.fit_transform(NumOfWord).toarray()
    NumOfVec = NumOfVec.astype(int).tolist()  # 数值转换成onehot向量
    word2num = dict(zip(BagOfWords, [_[0] for _ in NumOfWord]))
    num2word = dict(zip([_[0] for _ in NumOfWord], BagOfWords))

    SplitLineVec = []
    for line in SplitLine:
        for i, word in enumerate(line):  # 数字转换成对应的one_hot
            line[i] = word2num[word]
        SplitLineVec.append(line)
    return BagOfWords, word2num, num2word, SplitLineVec, NumOfVec

class Rnn(nn.Module):
    def __init__(self, Vocab_Size, num_layers=1, embed_size=256, hidden_size=128):
        super(Rnn, self).__init__()
        self.Vocab_Size = Vocab_Size
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.Vocab_Size, self.embed_size)  # 实现词嵌入
        self.rnn = nn.GRU(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        self.Dropout = nn.Dropout(0.5)
        self.out = nn.Linear(self.hidden_size, self.Vocab_Size)  # 输出多分类词数用于下一次输入

    def forward(self, x, h_state=None):
        batch = x.size(0)
        if h_state is None:
            h_state = torch.zeros(self.num_layers, batch, self.hidden_size)

        x = self.embedding(x)  # 首先实现词嵌入,按照词数目自动转换成对应的嵌入向量，不必再onehot自己编啦！
        x = x.permute(1, 0, 2)
        out, h_state = self.rnn(x, h_state)
        le, mb, hd = out.shape
        out = out.view(le * mb, hd)
        out = self.Dropout(out)
        out = self.out(out)
        return out, h_state

class TextDataset(object):
    """torch默认的数据类"""
    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, item):
        x = torch.LongTensor(self.arr[item])

        y = torch.zeros(x.shape, dtype=torch.int64)
        # 将输入的第一个字符作为最后一个输入的 label
        y[:-1], y[-1] = x[1:], x[0]
        return x, y

    def __len__(self):
        return self.arr.shape[0]


def Pick_Top_N(preds, top_n=5):
    """按概率随机选取前n个预测值作为最后的输出"""
    top_prob, top_label = torch.topk(preds, top_n, dim=1)
    top_prob += abs(min(top_prob))  #防止出现小数
    top_prob /= torch.sum(top_prob)
    top_prob = top_prob.squeeze(0).numpy()
    top_label = top_label.squeeze(0).numpy()
    return np.random.choice(top_label, size=1, p=top_prob)


def GeneratePoem(model, train_set):
    """预测生成部分:"""
    Line_Length = 20 #默认词的长度是20个词

    # 对当前的状态进行预热
    init_state = None
    init_X, _ = train_set.__getitem__(rand)
    for i in range(len(init_X) // 2):
        input = init_X[i].reshape(1, 1)
        _, init_state = model(input, init_state)

    #获取预热后的最后一个元素
    last_x = init_X[len(init_X)//2 - 1]
    result = []  #生成的结果
    for j in range(Line_Length):
        last_x = last_x.reshape(1, 1)
        out, init_state = model(last_x, init_state)
        pre = Pick_Top_N(out.data, 1)
        result.append(pre)

    return result


if __name__ == '__main__':
    data = LoadData("../poetry.txt", "r")

    BagOfWords, word2num, num2word, SplitLineVec, NumOfVec = DataProcess(data[:1500])

    model = Rnn(BagOfWords.__len__())
    print(BagOfWords.__len__())

    # 定义损失函数等
    loss_func = nn.CrossEntropyLoss()  # 交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    train_set = TextDataset(np.array(SplitLineVec))

    EPOCHES = 12  # 迭代次数
    #BATCH 固定了为1，即一次只输入一行语句，采用的SGD的方法训练

    """训练部分"""
    for epoch in range(EPOCHES):
        train_loss = 0
        for length in range(SplitLineVec.__len__()):
            batch_x, batch_y = train_set.__getitem__(length)
            #调整数据喂入的维度 wordnum * batchsize
            batch_x = batch_x.reshape(batch_x.size(0), 1)

            prediction, _ = model(batch_x)

            loss = loss_func(prediction, batch_y)
            optimizer.zero_grad()  # 清空原有梯度 否则梯度会累加
            loss.backward()  # 自动调用BP
            nn.utils.clip_grad_norm(model.parameters(), 5)  # 梯度裁剪
            optimizer.step()  # 更新参数
            train_loss += loss.data.numpy()
        print("\nEpoch: {}".format(epoch + 1), " Loss: {:.3f}".format(np.exp(train_loss / SplitLineVec.__len__())))

    #用训练好的模型生成数据
    rand = random.randint(0, SplitLineVec.__len__())

    result = GeneratePoem(model, train_set, NumOfVec)
    result = [num2word[int(_)] for _ in result]
    Gen_Poem = []
    for i in range(0, 16, 5):
        Gen_Poem.append(''.join(result[i:i+5]))
    print(result)
    print(Gen_Poem)

    with open("MyGeneratePoems.txt", 'a+') as f:
        f.write("EPOCHES: {}\n".format(EPOCHES))
        for line in Gen_Poem:
            f.write(line+" ")
        f.write("\n")

