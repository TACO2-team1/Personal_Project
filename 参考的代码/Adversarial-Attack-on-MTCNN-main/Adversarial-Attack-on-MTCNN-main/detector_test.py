import os
import torch.cuda
import PIL.Image as Image
from detector import Detector

if __name__ == '__main__':
    detector = Detector()
    count=0
    for i in range(1000):
        image_file = "Dataset/Attack/{}.jpg".format(i+172600)
        with Image.open(image_file) as img:
            boxes = detector.detect(img)
            if boxes.shape[0]==0:
                print("{} {}".format(i,count))
                count += 1
                torch.cuda.empty_cache()
    print(count)

# 1000 53