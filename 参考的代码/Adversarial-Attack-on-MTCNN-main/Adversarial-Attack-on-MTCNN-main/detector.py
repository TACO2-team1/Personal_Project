import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw
import utils
import pronets
from torchvision import transforms
import sys


class Detector:
    def __init__(self, pnet_param="./param/pnet/epoch17.pth", rnet_param="./param/rnet/epoch45.pth",
                onet_param="./param/onet/epoch44.pth"):
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

    def detect(self, image):
        with torch.no_grad():
            pnet_boxes = self.__pnet_detect(image)
            if pnet_boxes.shape[0] == 0:
                return np.array([])

            rnet_boxes = (self.__rnet_detect(image, pnet_boxes))
            if rnet_boxes.shape[0] == 0:
                return np.array([])

            onet_boxes = self.__onet_detect(image, rnet_boxes)
            if onet_boxes.shape[0] == 0:
                return np.array([])

            return onet_boxes

    def __pnet_detect(self, image):
        boxes = []
        img = image
        w, h = img.size
        min_side_len = min(w, h)

        scale = 1

        while min_side_len > 12:
            img_data = self.__image_transform(img)
            img_data = img_data.to(self.device)
            img_data.unsqueeze_(0)
            _cls, _offest, _landmark = self.pnet(img_data)
            cls, offest = _cls[0][0].cpu().data, _offest[0].cpu().data
            idxs = torch.nonzero(torch.gt(cls, 0.5))
            
            for idx in idxs:
                boxes.append(self.__box(idx, offest, cls[idx[0], idx[1]], scale))

            scale *= 0.709
            _w = int(w * scale)
            _h = int(h * scale)
            img = img.resize((_w, _h))
            min_side_len = np.minimum(_w, _h)
        return  utils.nms(np.array(boxes), 0.3)



    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):
        _x1 = int(start_index[1] * stride) / scale  # W，x
        _y1 = int(start_index[0] * stride) / scale  # H,y
        _x2 = int(start_index[1] * stride + side_len) / scale
        _y2 = int(start_index[0] * stride + side_len) / scale

        ow = _x2 - _x1  # 12
        oh = _y2 - _y1  # 12

        _offset = offset[:, start_index[0], start_index[1]]

        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]

    def __rnet_detect(self, image, pnet_boxes):
        _img_dataset = []
        _pnet_boxes = utils.convert_to_square(pnet_boxes)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        img_dataset = img_dataset.to(self.device)

        _cls, _offset, _landmark = self.rnet(img_dataset)

        _cls = _cls.cpu().data.numpy()

        offset = _offset.cpu().data.numpy()

        boxes = []

        idxs, _ = np.where(_cls > 0.45)

        for idx in idxs:
            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            cls = _cls[idx][0]

            boxes.append([x1, y1, x2, y2, cls])

        return utils.nms(np.array(boxes), 0.3)

    def __onet_detect(self, image, rnet_boxes):
        _img_dataset = []
        _rnet_boxes = utils.convert_to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        img_dataset = img_dataset.to(self.device)

        _cls, _offset, _landmark = self.onet(img_dataset)
        _cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()
        landmark = _landmark.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(_cls > 0.97)
        for idx in idxs:
            _box = _rnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            px1 = _x1 + ow * landmark[idx][0]
            py1 = _y1 + oh * landmark[idx][1]
            px2 = _x1 + ow * landmark[idx][2]
            py2 = _y1 + oh * landmark[idx][3]
            px3 = _x1 + ow * landmark[idx][4]
            py3 = _y1 + oh * landmark[idx][5]
            px4 = _x1 + ow * landmark[idx][6]
            py4 = _y1 + oh * landmark[idx][7]
            px5 = _x1 + ow * landmark[idx][8]
            py5 = _y1 + oh * landmark[idx][9]

            cls = _cls[idx][0]

            boxes.append([x1, y1, x2, y2, cls,px1,py1,px2,py2,px3,py3,px4,py4,px5,py5])
        return utils.nms(np.array(boxes), 0.3, isMin=True)


if __name__ == '__main__':
    with torch.no_grad() as grad:
        image_file = "test_images/{}".format(sys.argv[1])
        detector = Detector()

        with Image.open(image_file) as im:
            boxes = detector.detect(im)
            imDraw = ImageDraw.Draw(im)
            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])

                px1 = int(box[5])
                py1 = int(box[6])
                px2 = int(box[7])
                py2 = int(box[8])
                px3 = int(box[9])
                py3 = int(box[10])
                px4 = int(box[11])
                py4 = int(box[12])
                px5 = int(box[13])
                py5 = int(box[14])

                imDraw.rectangle((x1, y1, x2, y2), outline='red',width=3)
                imDraw.ellipse((px1-2,py1-2,px1+2,py1+2), 'red','red')
                imDraw.ellipse((px2-2,py2-2,px2+2,py2+2), 'red','red')
                imDraw.ellipse((px3-2,py3-2,px3+2,py3+2), 'red','red')
                imDraw.ellipse((px4-2,py4-2,px4+2,py4+2), 'red','red')
                imDraw.ellipse((px5-2,py5-2,px5+2,py5+2), 'red','red')
            im.save("test_images_output/{}".format(sys.argv[1]))
            im.show()