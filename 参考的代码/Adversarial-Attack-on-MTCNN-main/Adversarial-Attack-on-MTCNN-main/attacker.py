import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import utils
import pronets
from torchvision import transforms
import time
import os
import sys
import math
from detector import Detector

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=0.5):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class Attacker:
    def __init__(self, pnet_param="./param/pnet/epoch17.pth", rnet_param="./param/rnet/epoch45.pth",
                onet_param="./param/onet/epoch44.pth"):
        self.epsilon = 0.03
        self.loss_func1 = nn.BCELoss()
        self.loss_func2 = TVLoss()
        self.inner_iteration = 50
        self.step_size = 0.001*(2**2.25)

        self.detector = Detector()

        self.device=utils.try_gpu()

        self.pnet = pronets.PNet().to(self.device)
        self.rnet = pronets.RNet().to(self.device)
        self.onet = pronets.ONet().to(self.device)

        if torch.cuda.is_available():
            self.pnet.load_state_dict(torch.load(pnet_param))
            self.rnet.load_state_dict(torch.load(rnet_param))
            self.onet.load_state_dict(torch.load(onet_param))
        else:
            self.pnet.load_state_dict(torch.load(pnet_param,map_location='cpu'))
            self.rnet.load_state_dict(torch.load(rnet_param,map_location='cpu'))
            self.onet.load_state_dict(torch.load(onet_param,map_location='cpu'))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.__image_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.__tensor_transform = transforms.Compose([
            transforms.ToPILImage()
        ])
        
    def __find_scale(self, image):
        img = image.copy()
        boxes = self.detector.detect(img)
        if boxes.shape[0]==0:
            return [-1]

        len = (boxes[0][2]-boxes[0][0]+boxes[0][3]-boxes[0][1]) / 2.0
        pos = round(math.log(12.0/len)/math.log(0.709))
        print(pos)
        if pos == 0:
            return [0.709**pos,0.709**(pos+1),0.709**(pos+2),0.709**(pos+3),0.709**(pos+4)]
        elif pos==1:
            return [0.709**(pos-1),0.709**pos,0.709**(pos+1),0.709**(pos+2),0.709**(pos+3)]
        else:
            return [0.709**(pos-2),0.709**(pos-1),0.709**pos,0.709**(pos+1),0.709**(pos+2)]

    def attack_img(self,img):
        data = self.__image_transform(img)
        data = data.to(self.device)
        data.unsqueeze_(0)

        X = data + torch.Tensor(np.random.uniform(-self.epsilon, self.epsilon, data.shape)).type_as(data)
        for _ in range(self.inner_iteration):
            X = X.to(self.device)
            X.requires_grad = True
            output, _offset, _landmark = self.pnet(X)
            loss = self.loss_func1(output, torch.zeros_like(output)) + self.loss_func2(output)

            self.pnet.zero_grad()
            loss.backward()
            grad_data = self.step_size*(X.grad.data.sign())

            eta = torch.clamp(X-grad_data-data, -self.epsilon, self.epsilon)
            X = torch.clamp(data+eta,0,1).detach()
        return X-data

    def pyramid(self,image):
        scales= self.__find_scale(image)

        w1,h1=image.size
        _w1 = int(w1*scales[0])
        _h1 = int(h1*scales[0])
        im = image.resize((_w1,_h1))

        for scale in scales:
            w,h=image.size
            _w = int(w*scale)
            _h = int(h*scale)
            if _w==_w1 and _h==_h1:
                continue
            if _w<12 or _h<12:
                continue
            img = image.resize((_w,_h))
            im.paste(img,(_w1-_w,_h1-_h))
        return im

    def attack(self, image):
        start_time = time.time()
        scales= self.__find_scale(image)
        if scales[0]==-1:
            return image

        image_t=self.__image_transform(image)
        w,h=image.size
        for scale in scales:
            _w = int(w*scale)
            _h = int(h*scale)
            if _w<12 or _h<12:
                continue
            img = image.resize((_w,_h))

            delta = self.attack_img(img).unsqueeze(0)

            # delta = F.interpolate(delta, size=(3,h,w), mode='nearest')
            delta = F.interpolate(delta, size=(3,h,w), mode='trilinear', align_corners=True)
            image_t += delta.squeeze(0).squeeze(0).cpu()
            image_t = torch.clamp(image_t,0,1)
            
        return self.__tensor_transform(image_t)

# if __name__ == '__main__':
#     eps = float(sys.argv[1])
#     print(eps)
#     detector = Detector()
#     file = open("eps_for_pgd{}.txt".format(eps),"w")

#     start_time = time.time()
#     attacker = Attacker(eps)
#     save_path = "Dataset/Attacked/eps_{}".format(eps)
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     count = 0
#     for i in range(1000):
#         image_file = "Dataset/Attack/{}.jpg".format(i+172600)
#         with Image.open(image_file) as im:
#             img = attacker.attack(im)
#             if img==im:
#                 count += 1
#                 continue

#             boxes = detector.detect(img)
#             if boxes.shape[0]==0:
#                 count += 1
#                 img_path = os.path.join(save_path, "{}.jpg".format(i+172600))
#                 img.save(img_path)
#                 img.show()

#         if i%10==0:
#             print("{:04d}/1000\tEps:{:.5f}\tAcc: {:.2f}%\t Time: {:.2f}".format(
#                 i+1,eps,100.0*count/(i+1),time.time() - start_time))
#     file.write("{} {} {}\n".format(eps,count,1.0*count/1000))
#     file.flush()

if __name__ == '__main__':
    attacker=Attacker()
    image_file = "img_folder/img0.jpeg"
    with Image.open(image_file) as im:
        img = attacker.attack(im)
        img.save("attacked_img/img0.jpg")
