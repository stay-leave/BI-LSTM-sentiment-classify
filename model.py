import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer,BertModel
from config import ModelConfig
from dataset import Data_convert#引入数据集

#使用GPU
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

class BI_lstm(nn.Module):
    def __init__(self, vocab_size,vocab_to_index,n_layers,hidden_dim,embed,output_size,dropout):
        super(BI_lstm, self).__init__()
        self.n_layers = n_layers # LSTM的层数
        self.hidden_dim = hidden_dim# 隐状态的维度，即LSTM输出的隐状态的维度
        self.embedding_dim = embed # 将单词编码成多少维的向量
        self.dropout=dropout # dropout
        self.output_size=output_size
        
        # 定义embedding，随机将数字编码成向量。还没学会怎么使用预训练词向量
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim,padding_idx=vocab_to_index['PAD'])
        
        self.lstm = nn.LSTM(self.embedding_dim, # 输入的维度
                            hidden_dim, # LSTM输出的hidden_state的维度
                            n_layers, # LSTM的层数
                            dropout=self.dropout, 
                            batch_first=True, # 第一个维度是否是batch_size
                            bidirectional = True#双向
                           )
        # LSTM结束后的全连接线性层
        self.fc = nn.Linear(self.hidden_dim*2, self.output_size
                            ) # 由于情感分析只需要输出0或1，所以输出的维度是1# 将LSTM的输出作为线性层的输入
        self.sigmoid = nn.Sigmoid() # 线性层输出后，还需要过一下sigmoid
        self.tanh = torch.nn.Tanh()#激活函数
        #self.softmax=nn.Softmax()
        # 给最后的全连接层加一个Dropout
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, x, hidden):
        """
        x: 本次的输入，其size为(batch_size, 200)，200为句子长度
        hidden: 上一时刻的Hidden State和Cell State。类型为tuple: (h, c), 
        其中h和c的size都为(n_layers, batch_size, hidden_dim)
        """
        # 因为一次输入一组数据，所以第一个维度是batch的大小
        batch_size = x.size(0) 
        # 由于embedding只接受LongTensor类型，所以将x转换为LongTensor类型
        x = x.long() 
        # 对x进行编码，这里会将x的size由(batch_size, 200)转化为(batch_size, 200, embedding_dim)
        embeds = self.embedding(x)
        #embeds=self.relu(embeds)
        # 将编码后的向量和上一时刻的hidden_state传给LSTM，并获取本次的输出和隐状态（hidden_state, cell_state）
        # lstm_out的size为 (batch_size, 200, 128)，200是单词的数量，由于是一个单词一个单词送给LSTM的，所以会产生与单词数量相同的输出
        # hidden为tuple(hidden_state, cell_state)，它们俩的size都为(2, batch_size, 512), 2是由于lstm有两层。由于是所有单词都是共享隐状态的，所以并不会出现上面的那个200
        lstm_out, hidden = self.lstm(embeds, hidden)   
        # 接下来要过全连接层，所以size变为(batch_size * 200, hidden_dim)，
        # 之所以是batch_size * 200=40000，是因为每个单词的输出都要经过全连接层。
        # 换句话说，全连接层的batch_size为40000
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # 给全连接层加个Dropout
        out = self.dropout(lstm_out)
        # 将dropout后的数据送给全连接层
        # 全连接层输出的size为(40000, 1)
        out=torch.reshape(out,(-1,256))#改变形状
        out=self.tanh(out)#隐藏层激活函数
        out = self.fc(out)
        # 过一下sigmoid
        out = self.sigmoid(out)
        # 将最终的输出数据维度变为 (batch_size, 200)，即每个单词都对应一个输出
        out = out.view(batch_size, -1)
        # 只取最后一个单词的输出
        # 所以out的size会变为(200, 1)
        out = out[:,-1]
        # 将输出和本次的(h, c)返回
        return out,hidden
    
    def init_hidden(self, batch_size):
        """
        初始化隐状态：第一次送给LSTM时，没有隐状态，所以要初始化一个
        这里的初始化策略是全部赋0。
        这里之所以是tuple，是因为LSTM需要接受两个隐状态hidden state和cell state
        """
        hidden = (torch.zeros(self.n_layers*2, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.n_layers*2, batch_size, self.hidden_dim).to(device)
                 )
        
        return hidden

'''
#实例化
config=ModelConfig()
#训练集加载
data_train=Data_convert(config.input_path_test,config.seq_len,config.batch_size)#改变文件路径即可
vocab_train,sentence_train,label_train,sentences_train=data_train.count_s()#返回字典，分词句子，标签
train_loader=data_train.data_for_train_txt(sentence_train,vocab_train,label_train)
   
vocab_size = len(vocab_train)#字典大小

model=BI_lstm(vocab_size,vocab_train,config.n_layers,config.hidden_dim,config.embed,config.output_size,config.dropout)#模型实例化
print(model)
'''