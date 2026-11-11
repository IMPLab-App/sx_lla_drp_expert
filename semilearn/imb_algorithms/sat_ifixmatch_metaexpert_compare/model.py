from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock1D(nn.Module):
    def __init__(self,in_ch,out_ch,stride=1):
        super().__init__()
        self.conv1=nn.Conv1d(in_ch,out_ch,3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm1d(out_ch)
        self.conv2=nn.Conv1d(out_ch,out_ch,3,padding=1,bias=False)
        self.bn2=nn.BatchNorm1d(out_ch)
        self.shortcut=nn.Identity() if (stride==1 and in_ch==out_ch) else nn.Sequential(nn.Conv1d(in_ch,out_ch,1,stride=stride,bias=False),nn.BatchNorm1d(out_ch))
    def forward(self,x):
        y=F.relu(self.bn1(self.conv1(x)),inplace=True)
        y=self.bn2(self.conv2(y))
        return F.relu(y+self.shortcut(x),inplace=True)

class ResNet1D(nn.Module):
    def __init__(self,num_classes,input_channels=1,base_channels=64,layers=(2,2,2,2),dropout=0.0):
        super().__init__()
        self.in_ch=base_channels
        self.stem=nn.Sequential(nn.Conv1d(input_channels,base_channels,64,stride=8,padding=28,bias=False),nn.BatchNorm1d(base_channels),nn.ReLU(inplace=True),nn.MaxPool1d(2,2))
        self.layer1=self._make_layer(base_channels,layers[0],1)
        self.layer2=self._make_layer(base_channels*2,layers[1],2)
        self.layer3=self._make_layer(base_channels*4,layers[2],2)
        self.layer4=self._make_layer(base_channels*8,layers[3],2)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.drop=nn.Dropout(dropout)
        self.fc=nn.Linear(base_channels*8,num_classes)
    def _make_layer(self,out_ch,n,stride):
        layers=[BasicBlock1D(self.in_ch,out_ch,stride)]
        self.in_ch=out_ch
        for _ in range(1,n): layers.append(BasicBlock1D(self.in_ch,out_ch,1))
        return nn.Sequential(*layers)
    def forward_features(self,x):
        x=self.stem(x); x=self.layer1(x); x=self.layer2(x); x=self.layer3(x); x=self.layer4(x)
        return self.pool(x).squeeze(-1)
    def forward(self,x,return_features=False):
        f=self.forward_features(x)
        logits=self.fc(self.drop(f))
        return (logits,f) if return_features else logits
