#!/usr/bin/env python
# _*_coding:utf_8_*_
# @Time:2020/12/2413:25
# Authon:lingquan Liu
# @File:jiqi.py

# 导入必要包
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import scipy.sparse as sp

# 设置模型参数
epochs = 50//50
batch_size = 4096
lr = 0.005
lamda = 0.001       #SGD参数
factor_num=32
num_ng = 2          #负样本数目
test_num_ng = 99    #测试时负样本数目
top_k = 10
out = True
model_path='E:\\学习\\机器学习\\beautyDataset-master\\temp'

#读入数据集
data=pd.read_csv('E:\\学习\\机器学习\\beautyDataset-master\\Beauty.txt',sep=' ',header=None,names=['user','item'])
data["count"]=data.groupby("user")["item"].transform("count")
data=data[data["count"]>=5].copy() #去除了记录次数小于x的用户
data.drop(columns=['count'],inplace = True)

print('user_count:{}\t item_count:{}'.format(len(data['user'].unique()),len(data['item'].unique())))

#leave-one-out切分
test = data.groupby('user').tail(1).reset_index()
data.drop(index=test['index'], inplace=True)
test.drop(columns=['index'],inplace=True)
train = data


# 负采样
def get_neg_items(data):
    cands = set(list(train.item.unique())) - set(data)
    return np.random.choice(list(cands), 99)


neg_sample = train.groupby('user')['item'].apply(get_neg_items).reset_index()
test.rename(columns={'item': 'true_item'}, inplace=True)
neg_sample.rename(columns={'item': 'negative_item'}, inplace=True)

test_negative = pd.merge(test, neg_sample)

List = []
for index, row in test_negative.iterrows():
    u = row['user']
    ti = row['true_item']
    nis = row['negative_item']
    List.append([u, ti])
    for ni in nis:
        List.append([u, ni])

test_negative = pd.DataFrame(List)
test_negative.rename(columns={0: 'user', 1: 'item'}, inplace=True)

data.to_csv('E:\\学习\\机器学习\\beautyDataset-master\\temp\\train5.csv',encoding='gbk',index=False)
test.to_csv('E:\\学习\\机器学习\\beautyDataset-master\\temp\\test5.csv',encoding='gbk',index=False)
test_negative.to_csv('E:\\学习\\机器学习\\beautyDataset-master\\temp\\test_negative5.csv',encoding='gbk',index=False)

train_data=pd.read_csv('E:\\学习\\机器学习\\beautyDataset-master\\temp\\train5.csv')
test_data=pd.read_csv('E:\\学习\\机器学习\\beautyDataset-master\\temp\\test_negative5.csv')
user_num = train_data['user'].max()+1
item_num = max(train_data['item'].max(),test_data['item'].max())+1

train_data = train_data.values.tolist()
test_data = test_data.values.tolist()

# load ratings as a dok matrix
train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
for x in train_data:
    train_mat[x[0], x[1]] = 1.0

##模型数据加载类
class BPRData(Data.Dataset):
    def __init__(self,features,
                 num_item,train_mat=None,num_ng=0,is_training=None):
        super(BPRData,self).__init__()

        self.features=features
        self.num_item=num_item
        self.train_mat=train_mat
        self.num_ng=num_ng
        self.is_training=is_training

    #将数据集正例样本添加负例
    def ng_sample(self):
        assert self.is_training,'no need sampling when testing'


        self.features_fill=[]
        for x in self.features:
            u,i=x[0],x[1]
            for t in range(self.num_ng):
                j=np.random.randint(self.num_item)
                while(u,j) in self.train_mat:
                    j=np.random.randint(self.num_item)
                self.features_fill.append([u,i,j])

    def __len__(self):
        return self.num_ng * len(self.features) if \
            self.is_training else len(self.features)

    def __getitem__(self,idx):
        features=self.features_fill if \
            self.is_training else self.features

        user=features[idx][0]
        item_i=features[idx][1]
        item_j=features[idx][2] if \
            self.is_training else features[idx][1]
        return user,item_i,item_j


# 实例化dataset、dataloader
train_dataset = BPRData(train_data, item_num, train_mat, num_ng, True)
test_dataset = BPRData(test_data, item_num, train_mat, 0, False)
train_loader = Data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = Data.DataLoader(test_dataset,batch_size=test_num_ng+1, shuffle=False, num_workers=0)


#定义BPR模型
class BPR(nn.Module):
	def __init__(self, user_num, item_num, factor_num):
		super(BPR, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
		"""
		self.embed_user = nn.Embedding(user_num, factor_num)  #定义user的embedding
		self.embed_item = nn.Embedding(item_num, factor_num)  #定义item的embedding

		nn.init.normal_(self.embed_user.weight, std=0.01)
		nn.init.normal_(self.embed_item.weight, std=0.01)
    #前向传播
	def forward(self, user, item_i, item_j):
		user = self.embed_user(user)     #根据用户获取用户的embedding
		item_i = self.embed_item(item_i) #根据商品获取商品的embedding
		item_j = self.embed_item(item_j)

		prediction_i = (user * item_i).sum(dim=-1) #得到商品i的预测得分
		prediction_j = (user * item_j).sum(dim=-1) #得到商品j的预测得分
		return prediction_i, prediction_j


#实例化模型
model = BPR(user_num, item_num, factor_num)
# model.cuda()

#模型设置
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=lamda)


# 计算hit指标
def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


# 计算ndcg指标
def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def metrics(model, test_loader, top_k):
    HR, NDCG = [], []
    i = 0
    result = []
    for user, item_i, item_j in test_loader:
        user = user  # .cuda()
        item_i = item_i  # .cuda()
        item_j = item_j  # .cuda() # not useful when testing

        prediction_i, prediction_j = model(user, item_i, item_j)
        _, indices = torch.topk(prediction_i, top_k)  # 选出得分最高的前K的商品
        recommends = torch.take(
            item_i, indices).cpu().numpy().tolist()

        gt_item = item_i[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

        if i < 25:
            result.append([gt_item, recommends])
            i += 1
    return np.mean(HR), np.mean(NDCG), result

count, best_hr = 0, 0
loss_epoch = []
hit_epoch = []
ndcg_epoch = []
time_epoch = []
for epoch in range(epochs):
    model.train() #调用模型训练方法
    start_time = time.time() #记录训练起始时间
    train_loader.dataset.ng_sample() #取负例
    sum_loss_epoch = 0
    for user, item_i, item_j in train_loader:
        count = 0
        user = user#.cuda()
        item_i = item_i#.cuda()
        item_j = item_j#.cuda()

        model.zero_grad()
        prediction_i, prediction_j = model(user, item_i, item_j)     #得到商品预测得分
        loss = - (prediction_i - prediction_j).sigmoid().log().sum() #计算loss
        loss.backward()
        optimizer.step()
        # writer.add_scalar('data/loss', loss.item(), count)
        count += 1
        sum_loss_epoch += loss.item() / len(prediction_i)
    sum_loss_epoch += sum_loss_epoch / count
    print("loss:", sum_loss_epoch)
    loss_epoch.append(sum_loss_epoch)
    model.eval()
    HR, NDCG, result = metrics(model, test_loader, top_k)
    hit_epoch.append(HR)
    ndcg_epoch.append(NDCG)

    elapsed_time = time.time() - start_time  # 计算训练总用时
    time_epoch.append(elapsed_time)
    print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
          str(elapsed_time))
    print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

    if HR > best_hr:
        best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
        if out:
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            torch.save(model, '{}BPR.pt'.format(model_path))  # 保存模型

x1 = range(0,epochs)
y1 = loss_epoch
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.xlabel('epoches')
plt.ylabel('Train loss')
plt.show()

print("End. Best epoch {:03d}: HR = {:.3f}, \ NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))

load_model = torch.load('{}BPR.pt'.format(model_path))
HR, NDCG,result= metrics(load_model, test_loader, top_k)
result





