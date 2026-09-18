from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import torch
from torch import nn
LABELS=['breath','cough','crying','laugh','screaming','sneeze','yawn']
class SmallLogMelCNN(nn.Module):
    def __init__(self,num_classes=7):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),nn.MaxPool2d(2),nn.Dropout2d(.10),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),nn.MaxPool2d(2),nn.Dropout2d(.15),
            nn.Conv2d(64,96,3,padding=1),nn.BatchNorm2d(96),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(96,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d((1,1)))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Dropout(.25),nn.Linear(128,num_classes))
    def forward(self,x): return self.classifier(self.features(x))
