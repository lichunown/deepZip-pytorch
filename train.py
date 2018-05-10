from model import Model
from util import getBatch,one_hots,ModelManage
import torch
import torch.optim as optim

import torch.nn.functional as F

num_classes = 10
num_layers = 2
hidden_size = 128
lr = 0.1
epoch_num = 10
batch_size = 32
max_length = 10
vocablen = 2

modelpath = 'models/'

def accuracy():
    succ = 0
    allnum = 0
    for x,label in getBatch('data/e.txt', batchsize = 32):
        x = one_hots(x, 2)
        label = torch.LongTensor(label).view(-1)
        succ += torch.sum(torch.max(model(x), 1)[1] == label).data.numpy()
        allnum += 32
        del x,label
    return succ/allnum



manage = ModelManage()

def train(model = None,e = 0):
    if model is None:
        model = Model(batch_size = batch_size, input_size = vocablen, 
                         max_length = max_length, hidden_size = 128, num_layers = 2,
                         dropout = 0.1)
        
    model.train(True)
    optimizer = optim.Adam(model.parameters())
    i = 0
    for x,label in getBatch('data/input2.txt', batchsize = 32):
        x = one_hots(x, 2)
        x.requires_grad = True
        label = one_hots(label, 2).view(-1,2).long()
#            label = torch.LongTensor(label)
        optimizer.zero_grad()
        loss = F.cross_entropy(x, label)
        loss.backward()
        optimizer.step()
        del x, label
        i += 1
        if i % 100 == 0:
            manage.save(model, {'hdsize':hidden_size})
            print('[epoch {}:{}]    loss:{}  acc:{} '.format(e, i, float(loss), accuracy()))
        del loss

try:
    __IPYTHON__
except Exception:
    if __name__=='__main__':
        traintype = {'hdsize':hidden_size}
        for e in range(epoch_num):
            model = manage.load_type(traintype)
            train(model, e)