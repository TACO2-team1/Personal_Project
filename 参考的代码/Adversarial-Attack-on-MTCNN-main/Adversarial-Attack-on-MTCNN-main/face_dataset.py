from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import PIL.Image as Image
import numpy as np
import torch
import os

class FaceDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(path, "positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "landmark.txt")).readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split(" ")
        img_path = os.path.join(self.path, strs[0])
        
        cls = torch.tensor([int(strs[1])],dtype=torch.float32)
        offset = torch.tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])],dtype=torch.float32)
        landmark = torch.tensor([float(strs[6]), float(strs[7]), float(strs[8]), float(strs[9]), float(strs[10]),
            float(strs[11]), float(strs[12]), float(strs[13]), float(strs[14]), float(strs[15])],dtype=torch.float32)
        img_data = torch.tensor(np.array(Image.open(img_path)) / 255.0 - 0.5,dtype=torch.float32)
        img_data = img_data.permute(2,0,1)

        return img_data, cls, offset, landmark

if __name__ == '__main__':
    dataset = FaceDataset(r"./Dataset/12")
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)
    dataloader = DataLoader(dataset,5,shuffle=True,num_workers=4)
    for i ,(img,cls,offset,landmark) in enumerate(dataloader):
        print(img.shape)
        print(cls.shape)
        print(cls)
        print(offset.shape)
        print(landmark.shape)
        pass