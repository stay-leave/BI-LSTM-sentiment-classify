#coding='utf-8'
from dataset import Data_convert#引入数据集
from model import BI_lstm#模型
from config import ModelConfig#配置
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def train(config,model,train_loader):
    #模型训练
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)#
    criterion = nn.BCELoss()# 分类问题
    y_loss=[]#训练过程的所有loss
    for e in range(config.epochs):
        # initialize hidden state,初始化隐层状态
        h = model.init_hidden(config.batch_size)
        counter = 0
        train_losses=[]
        # 分批
        for inputs, labels in train_loader:
            counter += 1
            inputs, labels = inputs.cuda(), labels.cuda()#  GPU
            h = tuple([each.data for each in h])
            #model.zero_grad()#梯度清零
            output,h= model(inputs, h)
            output=output[:, np.newaxis]#加上新的维度
            #print(inputs)
            #print(output)
            #print(labels.float())
            train_loss = criterion(output, labels.float())
            train_losses.append(train_loss.item())
            optimizer.zero_grad()
            train_loss.backward()#反向传播
            optimizer.step()#更新权重

            
            # loss 训练集信息
            if counter % config.print_every == 0:#打印间隔
                print("Epoch: {}/{}, ".format(e+1, config.epochs),
                        "Step: {}, ".format(counter),
                        "Loss: {:.6f}, ".format(train_loss.item()),
                        "Val Loss: {:.6f}".format(np.mean(train_losses)))
            y_loss.append(train_loss.item())#写入
    # 训练完画图
    x = [i for i in range(len(y_loss))]
    fig = plt.figure()
    plt.plot(x, y_loss)
    plt.show()
    #保存完整的预训练模型
    torch.save(model,config.save_model_path)
            
def test(config, model, test_loader):
    #模型验证，计算损失和准确率
    criterion = nn.BCELoss()# 分类问题
    h = model.init_hidden(config.batch_size)
    with torch.no_grad():#不计算梯度，不进行反向传播，节省资源
        count = 0  # 预测的和实际的label相同的样本个数
        total = 0  # 累计validation样本个数
        loss=0#损失
        l=0#损失的计数
        for input_test, target_test in test_loader:
            h = tuple([each.data for each in h])
            input_test = input_test.type(torch.LongTensor)#long
            target_test = target_test.type(torch.LongTensor)
            target_test = target_test.squeeze(1)
            input_test = input_test.cuda()#GPU
            target_test = target_test.cuda()
            output_test,h = model(input_test,h)#output_test为输出结果,(0,1)
            pred=output_test.cpu().numpy().tolist()#输出值列表
            target=target_test.cpu().numpy().tolist()#目标值列表
            for i,j in zip(pred,target):
                if round(i)==j:
                    count=count+1#正确个数
            total += target_test.size(0)#测试样本总数
            #损失计算
            loss = criterion(output_test, target_test.float())
            loss+=loss#自增
            l=l+1#计数
        acc=100 * count/ total#测试集准确率
        test_loss=loss/l#测试集平均损失
        print("test mean loss: {:.3f}".format(test_loss))
        print("test accuracy : {:.3f}".format(acc))

        
if __name__ == '__main__':

    config=ModelConfig()#实例化配置

    #训练集加载
    data_train=Data_convert(config.input_path_train,config.seq_len,config.batch_size)#改变文件路径即可
    vocab_train,sentence_train,label_train,sentences_train=data_train.count_s()#返回字典，分词句子，标签
    train_loader=data_train.data_for_train_txt(sentence_train,vocab_train,label_train)
   
    vocab_size = len(vocab_train)#字典大小
    #保存字典
    np.save(config.save_dict_path,vocab_train)
    print('字典已保存')

    #测试集加载
    data_test=Data_convert(config.input_path_test,config.seq_len,config.batch_size)#改变文件路径即可
    vocab_test,sentence_test,label_test,sentences_test=data_test.count_s()#返回字典，分词句子，标签，注意这里是测试集的字典，实际需要用到训练集字典，因为没有用到词向量嵌入
    test_loader=data_test.data_for_test_txt(sentence_test,vocab_train,label_test)#使用训练集字典

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")#GPU

    model=BI_lstm(vocab_size,vocab_train,config.n_layers,config.hidden_dim,config.embed,config.output_size,config.dropout)#模型实例化
    model.to(device)
    #训练并保存模型
    train(config,model,train_loader)
    #测试评估模型
    test(config,model,test_loader)



    '''
    训练和测试在一个文件
    data=Data_convert(config.input_path_all,config.seq_len,config.batch_size)
    vocab,sentence,label,sentences=data.count_s()#返回字典，分词句子，标签
    vocab_size = len(vocab)# 词典大小
    #train_loader,dev_loader,test_loader=data.data_for_train_dev_test(sentence,vocab,label)
    train_loader,test_loader=data.data_for_train_dev_test(sentence,vocab,label)
    #保存字典
    np.save(config.save_dict_path,vocab)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    model=BI_lstm(vocab_size,vocab,config.n_layers,config.hidden_dim,config.embed,config.output_size,config.dropout)
    model.to(device)
    #训练并保存模型
    train(config, model, train_loader)
    #测试评估模型
    test(config, model, test_loader)


    
    '''

    


