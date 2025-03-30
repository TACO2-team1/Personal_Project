import torch
import torch.nn as nn


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.pre_layer=nn.Sequential(
            # 12*12*3
            nn.Conv2d(3, 10, kernel_size=3, stride=1), # 10*10*10
            nn.BatchNorm2d(10),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 5*5*10

            nn.Conv2d(10, 16, kernel_size=3, stride=1), # 3*3*16
            nn.BatchNorm2d(16),
            nn.PReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=1), # 1*1*32
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.cls_layer=nn.Conv2d(32, 1, kernel_size=1, stride=1) # 1*1*1
        self.bbox_layer=nn.Conv2d(32, 4, kernel_size=1, stride=1) # 1*1*4
        self.landmark_layer=nn.Conv2d(32, 10, kernel_size=1, stride=1) # 1*1*10
    
    def forward(self, x):
        x=self.pre_layer(x)
        cls=torch.sigmoid(self.cls_layer(x))
        bbox=self.bbox_layer(x)
        landmark=self.landmark_layer(x)
        return cls, bbox, landmark


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.pre_layer = nn.Sequential(
            # 24*24*3
            nn.Conv2d(3, 28, kernel_size=3, stride=1), # 22*22*28
            nn.BatchNorm2d(28),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 11*11*28

            nn.Conv2d(28, 48, kernel_size=3, stride=1), # 9*9*48
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 4*4*48

            nn.Conv2d(48, 64, kernel_size=2, stride=1), # 3*3*64
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.fc=nn.Linear(64*3*3, 128)
        self.prelu=nn.PReLU()

        self.cls_layer=nn.Linear(128, 1)
        self.bbox_layer=nn.Linear(128, 4)
        self.landmark_layer=nn.Linear(128, 10)
    
    def forward(self, x):
        x=self.pre_layer(x)
        x=x.view(x.size(0), -1)
        x=self.prelu(self.fc(x))
        cls=torch.sigmoid(self.cls_layer(x))
        bbox=self.bbox_layer(x)
        landmark=self.landmark_layer(x)
        return cls, bbox, landmark


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.pre_layer = nn.Sequential(
            # 48*48*3
            nn.Conv2d(3, 32, kernel_size=3, stride=1), # 46*46*32
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 23*23*32

            nn.Conv2d(32, 64, kernel_size=3, stride=1), # 21*21*64
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 10*10*64
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 8*8*64
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4*4*64

            nn.Conv2d(64, 128, kernel_size=2, stride=1), # 3*3*128
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )
        self.fc=nn.Linear(3*3*128, 256)
        self.prelu=nn.PReLU()
        self.dropout=nn.Dropout(0.25)
        self.cls_layer=nn.Linear(256,1)
        self.bbox_layer=nn.Linear(256,4)
        self.landmark_layer=nn.Linear(256,10)
    
    def forward(self, x):
        x=self.pre_layer(x)
        x=x.view(x.size(0), -1)
        x=self.prelu(self.fc(x))
        cls=torch.sigmoid(self.cls_layer(x))
        bbox=self.bbox_layer(x)
        landmark=self.landmark_layer(x)
        return cls, bbox, landmark

if __name__=='__main__':
    print("PNET")
    net = PNet()
    x = torch.randn(16,3,12,12)
    y,z,w = net(x)
    print(y.shape)
    print(z.shape)
    print(w.shape)

    print("RNET")
    net = RNet()
    x = torch.randn(16,3,24,24)
    y,z,w = net(x)
    print(y.shape)
    print(z.shape)
    print(w.shape)

    print("ONET")
    net = ONet()
    x = torch.randn(16,3,48,48)
    y,z,w = net(x)
    print(y.shape)
    print(z.shape)
    print(w.shape)