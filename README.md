# BlogCodes

这个仓库主要保存的对应网页下的CSDN的代码,其中包括对应的原理和源码.

## MyWord2Vec:

用全联接网络将“语料”训练,最后将对应的权重保存到相对路径下的“w.txt”文件下,还没有对保存后的权重
矩阵做处理

(博客地址: https://blog.csdn.net/Laugh_xiaoao/article/details/102944177)

## MyTF-IDF:

用全联接网络将“语料”训练,最后将对应的权重保存到相对路径下的“w.txt”文件下,还没有对保存后的权重
矩阵做处理.

(博客地址: https://blog.csdn.net/Laugh_xiaoao/article/details/102944177)

## Pytorch_RNN:

使用了CharRNN对于语句序列进行学习, 通过将诗句划分成一个个字(没有采用分词的原因是对于不等长序列的处理还不够纯熟,有待改进.)
网络模型架构是一层Embedding层+GRU层+Relu层+输出层, 由于是采用的CharRNN所以, 每次输入RNN的batchsize都只能是1, seq_length 就是一句话中包含的字数, 

除此之外, SGD的方法训练的所以网络的Loss波动很大
总结下来, 这个小实验对于自己了解RNN,LSTM,和GRU都还是挺有用的...至少明白了在pytorch框架下,RNN的需求输入和输出. 能比较好的转换了...

(博客地址: https://blog.csdn.net/Laugh_xiaoao/article/details/103137601)

## FiveChess:

用java实现的五子棋系统,algorithm包是基本的算法,除了level3的采用了决策树,另外两个没有使用,纯粹是为了当时课设使用的所需.
考虑优化的代码的话,有两方面:

1.提升下棋速度,可以从择点入手,当初我选点的时候比较粗暴,直接添加到Vector里面了,可能存在重复的点被重复计算,可以考虑像DP一样采用一个对应的表来加速计算或避免重复点加入.

2.提升AI智能的话,可以加深层次,修改MaxDepth可以,也可以增加选点的范围,主要是改变对应的range变量.评估函数上的选择也可以在精细一点,因为目前只能匹配出连起来的四个点这种,但其实也可以有’***0*‘这样的棋型.权重也可以修改的更仔细.期待大家可以得到更好的AI.

(博客地址: https://blog.csdn.net/Laugh_xiaoao/article/details/103761624)

## Pytorch_TextCNN:

用TextCNN实现的情感分类的小实验代码.

(博客地址:https://blog.csdn.net/Laugh_xiaoao/article/details/103820784)



## Pytorch_NER:

该py.文件主要实现了大致的NER中模型包括HMM,CRF,CRF+LSTM,LSTM等,LSTM是借助于Pytorch实现的.

博客我分成了两份来写,具体的地址如下:

(博客地址:https://blog.csdn.net/Laugh_xiaoao/article/details/104592504,

博客地址:https://blog.csdn.net/Laugh_xiaoao/article/details/104805813)

