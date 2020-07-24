
import torch
import torch.nn as nn
import torch.legacy.nn as lnn

from functools import reduce
from torch.autograd import Variable

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))


/home/javaprog/Data/Broad/Basset/Model/ampt2d_cnn_900_best_cpu_th = nn.Sequential( # Sequential,
	nn.Conv2d(4,400,(21, 1),(1, 1),(10, 0)),
	nn.BatchNorm2d(400),
	nn.ReLU(),
	nn.MaxPool2d((3, 1),(3, 1),(0, 0),ceil_mode=True),
	nn.Conv2d(400,300,(11, 1),(1, 1),(5, 0)),
	nn.BatchNorm2d(300),
	nn.ReLU(),
	nn.MaxPool2d((4, 1),(4, 1),(0, 0),ceil_mode=True),
	nn.Conv2d(300,300,(7, 1),(1, 1),(3, 0)),
	nn.BatchNorm2d(300),
	nn.ReLU(),
	nn.MaxPool2d((4, 1),(4, 1),(0, 0),ceil_mode=True),
	nn.Conv2d(300,300,(5, 1),(1, 1),(2, 0)),
	nn.BatchNorm2d(300),
	nn.ReLU(),
	nn.MaxPool2d((4, 1),(4, 1),(0, 0),ceil_mode=True),
	Lambda(lambda x: x.view(x.size(0),-1)), # Reshape,
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(1500,1024)), # Linear,
	nn.BatchNorm1d(1024,1e-05,0.1,True),#BatchNorm1d,
	nn.ReLU(),
	nn.Dropout(0.3),
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(1024,512)), # Linear,
	nn.BatchNorm1d(512,1e-05,0.1,True),#BatchNorm1d,
	nn.ReLU(),
	nn.Dropout(0.3),
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(512,167)), # Linear,
	nn.Sigmoid(),
)