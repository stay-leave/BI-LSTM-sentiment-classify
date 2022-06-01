#coding='utf-8'
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F 
import torch.nn as nn
from torch.utils.data import *
import os
import re 
import jieba
import xlrd
import collections
import numpy as np 
from config import ModelConfig#配置

class Data_convert():
    '''将xls数据转为tensor，供模型加载'''
    def __init__(self,inpath,seq_length,batch_size):
        #初始化
        self.inpath=inpath
        self.seq_length = seq_length#每句话截断长度max([len(i) for i in sentence])
        self.batch_size = batch_size

    def xls_file(self,inpath):
        """提取一个文件为一个大列表"""
        data = xlrd.open_workbook(self.inpath, encoding_override='utf-8')
        table = data.sheets()[0]#选定表
        nrows = table.nrows#获取行号
        ncols = table.ncols#获取列号
        numbers=[]
        for i in range(1, nrows):#第0行为表头
            alldata = table.row_values(i)#循环输出excel表中每一行，即所有数据
            numbers.append(alldata)
        return numbers

    def txt_file(self,inpath):
    #输入TXT，返回列表
        data = []
        fp = open(self.inpath,'r',encoding='utf-8')
        for line in fp:
            line=line.strip('\n')
            line=line.split('\t')
            data.append(line)
        data=data[1:]#去掉表头
        return data

    def tokenlize(self,sentence):
        #分词,只要/保留 中文/其他字符,单句
        #sentence = re.sub('[^\u4e00-\u9fa5]+','',sentence)
        URL_REGEX = re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',re.IGNORECASE)
        sentence= re.sub(URL_REGEX,'', sentence)# 去除网址
        sentence =jieba.cut(sentence.strip(),cut_all=False,use_paddle=10)#默认精确模式
        out=[] 
        for word in sentence:
            out.append(word)
        return out

    def splitt(self,data):
        #句子和标签的提取
        sentence=[]
        label=[]
        for i in data:
            sentence.append(self.tokenlize(i[1]))
            label.append(int(i[0]))#使用txt
            #label.append(int(i[2]))#使用xls
        sentence=tuple(sentence)
        label=tuple(label)
        return sentence,label

    def splitt_nos(self,data):
        #句子和id的提取，不分词，预测用
        sentence=[]
        id=[]
        for i in data:
            sentence.append(i[1])
            id.append(int(i[0]))#使用txt
            #id.append(int(i[2]))#使用xls
        sentence=tuple(sentence)
        id=tuple(id)
        return sentence,id

    def count_s(self):
        #统计词频，排序，建立词典（词和序号对）
        sentence,label=self.splitt(self.txt_file(self.inpath))#提取数据，分词,使用txt读取
        #sentence,label=self.splitt(self.xls_file(self.inpath))#提取数据，分词,使用xls读取
        count_dict = dict()#普通词典，词：词频
        sentences=[]#合并列表
        for i in sentence:
            sentences += i
        for item in sentences:
            if item in count_dict:
                count_dict[item] += 1
            else:
                count_dict[item] = 1
        #print(count_dict)
        #count_dict_s = sorted(count_dict.items(),key=lambda x: x[1], reverse=True)#以值来排序
        count_dict_s = collections.OrderedDict(sorted(count_dict.items(),key=lambda t:t[1], reverse=True))#降序
        #print('排序字典：')
        #print(count_dict_s)
        vocab=list(count_dict_s.keys())#转换成列表
        vocab_index=[i for i in range(1,len(vocab)+1)]#索引值
        vocab_to_index = dict(zip(vocab, vocab_index))#词汇索引
        vocab_to_index["PAD"] = 0#补全
        #vocab_to_index["UNK"] = 0#补零
        return vocab_to_index,sentence,label,sentences

    def seq_to_array(self,seq,vocab_to_index):
        #单个句子转换为数字序列，顺序输出标签,需要先将句子分词
        #inputs = []
        #for i in seq:#取单个句子
            seq_index=[]#单个句子的数字序列
            for word in seq:#取句子的词
                if word in vocab_to_index:#句子的字在字典中
                    seq_index.append(vocab_to_index[word])
                else:
                    seq_index.append(0)#未登录词的处理，为pad
            # 保持句子长度一致
            if len(seq_index) < self.seq_length:#若句子的数字序列短，补全为0
                seq_index = [0] * (self.seq_length-len(seq_index)) + seq_index
            elif len(seq_index) > self.seq_length:#若句子的数字序列长，截断
                seq_index = seq_index[:self.seq_length]
            else:
                seq_index=seq_index
            #inputs.append(seq_index)#所有句子的数字序列
            #targets = [i for i in label]#对应标签
            return seq_index

    def array_to_seq(self,indices):
        #数字序列转换为句子,一大堆
        vocab_to_index,sentence,label,sentences=self.count_s()
        seqs=[]#全部
        for i in indices:
            seq=[]#单句
            for j in i:
                for key, value in vocab_to_index.items():
                    if value==j:
                        seq.append(key)
            seqs.append(seq)
        return seqs
    
    def data_for_train_dev_test(self,sentence,vocab_to_index,label):
        #切分训练、测试集
        features=[self.seq_to_array(seq,vocab_to_index) for seq in sentence]#将所有分词好的句子转换为数字序列
        # 随机打乱索引
        random_order = list(range(len(features)))
        np.random.seed(2)   # 固定种子
        np.random.shuffle(random_order)#洗牌
        #划分训练集，80%
        features_train = np.array([features[i] for i in random_order[:int(len(features)*0.8)]])
        label_train = np.array([label[i] for i in random_order[:int(len(features) * 0.8)]])[:, np.newaxis]
        #print(features_train.shape,label_train.shape)#打印形状
        #验证集，20%
        #features_dev = np.array([features[i] for i in random_order[int(len(features) * 0.6):int(len(features)*0.8)]])
        #self.writes_2((features_dev))#将数组写入
        #label_dev = np.array([label[i] for i in random_order[int(len(features) * 0.6):int(len(features) * 0.8):]])[:, np.newaxis]
        #测试集，20%
        features_test = np.array([features[i] for i in random_order[int(len(features)*0.8):]])
        label_test = np.array([label[i] for i in random_order[int(len(features) * 0.8):]])[:, np.newaxis]
        #print(features_test.shape, label_test.shape)

        #加载到tensor
        train_data = TensorDataset(torch.LongTensor(features_train), 
                            torch.LongTensor(label_train))
        train_sampler = RandomSampler(train_data)  
        train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size, drop_last=True)
        '''
        dev_data = TensorDataset(torch.LongTensor(features_dev), 
                            torch.LongTensor(label_dev))
        dev_sampler = RandomSampler(dev_data)  
        dev_loader = DataLoader(dev_data, sampler=dev_sampler, batch_size=self.batch_size, drop_last=True)
        '''
        test_data = TensorDataset(torch.LongTensor(features_test), 
                          torch.LongTensor(label_test))
        test_sampler = SequentialSampler(test_data)
        test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size, drop_last=True)
        #return train_loader,dev_loader,test_loader
        return train_loader,test_loader

    def data_for_train_txt(self,sentence,vocab_to_index,label):
        #加载训练集
        features=[self.seq_to_array(seq,vocab_to_index) for seq in sentence]#将所有分词好的句子转换为数字序列
        # 随机打乱索引
        random_order = list(range(len(features)))
        np.random.seed(2)   # 固定种子
        np.random.shuffle(random_order)#洗牌
        #训练集to数组
        features_train = np.array([features[i] for i in random_order])
        label_train = np.array([label[i] for i in random_order])[:, np.newaxis]
        #print(features_train.shape,label_train.shape)#打印形状
        #加载到tensor
        train_data = TensorDataset(torch.LongTensor(features_train), 
                            torch.LongTensor(label_train))
        train_sampler = RandomSampler(train_data)  
        train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size, drop_last=True)

        return train_loader

    def data_for_test_txt(self,sentence,vocab_to_index,label):
        #加载测试集
        features=[self.seq_to_array(seq,vocab_to_index) for seq in sentence]#将所有分词好的句子转换为数字序列
        # 随机打乱索引
        random_order = list(range(len(features)))
        np.random.seed(2)   # 固定种子
        np.random.shuffle(random_order)#洗牌
        #训练集to数组
        features_test = np.array([features[i] for i in random_order])
        label_test = np.array([label[i] for i in random_order])[:, np.newaxis]
        #print(features_test.shape,label_test.shape)#打印形状
        #加载到tensor
        test_data = TensorDataset(torch.LongTensor(features_test), 
                            torch.LongTensor(label_test))
        test_sampler = RandomSampler(test_data)  
        test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size, drop_last=True)

        return test_loader

    def data_for_pred_txt(self,sentence,vocab_to_index,id):
        #加载测试集
        features=[self.seq_to_array(seq,vocab_to_index) for seq in sentence]#将所有分词好的句子转换为数字序列
        #训练集to数组
        features_pred = np.array(features)
        id_pred = np.array(id)#[:, np.newaxis]
        #加载到tensor
        pred_data = TensorDataset(torch.LongTensor(features_pred),
                                  torch.LongTensor(id_pred))
        pred_loader = DataLoader(pred_data,batch_size=self.batch_size,drop_last=True) #根据待预测的句子数确定batch_size

        return pred_loader

    def writes_1(self,data):
        #将列表写入txt
        f = open('词汇列表.txt','w')
        for i in data:
            f.write(str(i)+'\n')
        f.close()
    def writes_2(self,numpy_data):
        #将数组写入txt
        np.savetxt("训练集数组.txt", numpy_data)
    

#-------------------------------------------------数据处理完毕--------------------------------------------------------------#
#实例化
#vocab_size = len(vocab_to_index)# 词典大小
#seq_size = len(sentence)#句子数量
'''
config=ModelConfig()

data_train=Data_convert(config.input_path_train,config.seq_len,config.batch_size)#改变文件路径即可
vocab_train,sentence_train,label_train,sentences_train=data_train.count_s()#返回字典，分词句子，标签
train_loader=data_train.data_for_train_txt(sentence_train,vocab_train,label_train)

#保存字典
np.save(config.save_dict_path,vocab_train)
#读取字典
vocab_train = np.load(config.save_dict_path+'.npy',allow_pickle=True)
vocab_train= vocab_train.item()#读取

'''

'''
data_test=Data_convert(config.input_path_train,config.seq_len,config.batch_size)#改变文件路径即可
vocab_test,sentence_test,label_test,sentences_test=data_train.count_s()#返回字典，分词句子，标签，注意这里是测试集的字典，实际需要用到训练集字典，因为没有用到词向量嵌入
test_loader=data_test.data_for_test_txt(sentence_test,vocab_train,label_test)#使用训练集字典
'''

'''
#打印数据
# obtain one batch of training data
dataiter = iter(test_loader)
sample_x, sample_y = dataiter.next()
print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x[0,:])
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)
'''