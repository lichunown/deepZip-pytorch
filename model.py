import torch
import torch.nn as nn   




num_classes = 10
num_layers = 2
hidden_size = 128
lr = 0.001
epoch_num = 10
batch_size = 32
max_length = 10
vocablen = 2


class Model(nn.Module):
    def __init__(self, batch_size = batch_size, input_size = vocablen, 
                 max_length = max_length, hidden_size = 128, num_layers = 2,
                 dropout = 0.1):
        
        super(Model, self).__init__()
        self.rnn = nn.GRU(input_size = input_size, hidden_size = hidden_size,
                          num_layers = 2, dropout = 0.1)
        self.state = torch.randn(num_layers, max_length, hidden_size)
        self.linear = nn.Linear(hidden_size, input_size)
#        self.softmax = nn.Softmax(dim = 1)#
        
    def forward(self, inputs):
        x, self.state = self.rnn(inputs, self.state)
        x = x[:,-1,:]
        x = self.linear(x)
#        x = self.softmax(x)
        return x
        

    
    
    
    
    
    
    
    
    
    