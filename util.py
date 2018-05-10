import numpy as np
import torch
import os

vocab = 'ab'
vocablen = 2

def one_hot(inputs, nb_digits = vocablen):
    n = inputs.shape[0]
    y_onehot = torch.zeros([n, nb_digits])
    y_onehot.scatter_(1, inputs.view(-1,1).long(), 1)
    return y_onehot

def one_hots(inputs, nb_digits = vocablen):
    assert len(inputs.shape)==2
    if isinstance(inputs, np.ndarray):
        inputs = torch.Tensor(inputs)
    batch_size, n = inputs.shape
    y = torch.zeros([batch_size, n, nb_digits])
    for i in range(batch_size):
        y[i,:,:] = one_hot(inputs[i], nb_digits)
    return y


def vocab_encode(text):
    return [vocab.index(x) for x in text if x in vocab]

def vocab_decode(array):
    return ''.join([vocab[x] for x in array])

def readData(filename='input.txt',max_length=10):
    window = max_length
    overlap = 1
    for text in open(filename):
        text = vocab_encode(text)
        for start in range(0, len(text) - window - 1, overlap):
            chunk = text[start: start + window + 1]
            chunk += [0] * (window+1 - len(chunk))
            yield chunk
            
def getBatch(filename='input.txt', max_length=10, batchsize = 32):
    stream = readData(max_length = max_length)
    exiti = False
    while not exiti:
        x = np.zeros([batchsize, max_length])
        label = np.zeros([batchsize, max_length])
        for i in range(batchsize):
            try:
                d = next(stream)
            except Exception:
                exiti = True
                break
            else:
                x[i,:] = d[:-1]
                label[i,:] = d[1:]
        yield x, label[:,-1].reshape(-1,1)
    
class ModelManage(object):
    
    def __init__(self, modelpath = './models/'):
        self.modelpath = modelpath
        if not os.path.exists(self.modelpath):
            os.makedirs(self.modelpath)
            
    @property
    def list(self):
        return self.getModelList()
    
    def getModelList(self):
        modelnames = []
        for name in os.listdir(self.modelpath):
            if os.path.isfile(os.path.join(self.modelpath, name)):
                modelnames.append(name)
        return modelnames
    
    def typename(self,types):
        return '_'.join(['{}-{}'.format(dname, dvalue) for dname,dvalue in types.items()])
    
    def maxid(self, typename):
        filenamelist = self.list
        iid = -1
        for name in filenamelist:
            if name.split('.')[-2].split('__')[0] == typename:
                try:
                    tid = int(name.split('.')[-2].split('__')[-1])
                except Exception:
                    print('error: tid:`{}`'.format(name.split('.')[-2].split('_')[-1]))
                    continue
                else:
                    iid = tid if tid > iid else iid
        return iid
        
    def save(self, model, types = {}, ids = None):
        typename = self.typename(types)
        if ids is None:
            ids = self.maxid(typename) + 1
        filename = '{}__{}.model'.format(typename, ids)
        print('Save model: `{}`'.format(filename))
        with open(os.path.join(self.modelpath, filename), 'wb') as f:
            torch.save(model, f)
    
    def load(self,name):
        print('Load: `{}`'.format(name))
        with open(os.path.join(self.modelpath, name)) as f:
            return torch.load(f)
        
    def load_type(self, types, ids = None):
        typename = self.typename(types)
        if ids is None:
            ids = self.maxid(typename)
        name = '{}__{}.model'.format(typename, ids)
        print('Load model: `{}`'.format(name))
        with open(os.path.join(self.modelpath, name), 'rb') as f:
            return torch.load(f)
        
        
        
        
        
        
        
        
        
        
        
        