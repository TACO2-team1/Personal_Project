import numpy as np
import torch

def try_gpu():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

def iou(box, boxes, isMin=False):
    box_area=(box[2]-box[0])*(box[3]-box[1])
    boxes_area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    in_x1=np.maximum(box[0], boxes[:,0])
    in_y1=np.maximum(box[1], boxes[:,1])
    in_x2=np.minimum(box[2], boxes[:,2])
    in_y2=np.minimum(box[3], boxes[:,3])
    in_area=np.maximum(0, in_x2-in_x1)*np.maximum(0, in_y2-in_y1)
    if isMin:
        return np.true_divide(in_area, np.minimum(box_area, boxes_area))
    else:
        return np.true_divide(in_area, box_area+boxes_area-in_area)


def nms(raw_boxes, threshold=0.3, isMin=False):
    if raw_boxes.shape[0]==0:
        return np.array([])

    boxes=raw_boxes[(-raw_boxes[:, 4]).argsort()]
    picks=np.array([])
    picks_cnt=0

    while boxes.shape[0]>1:
        a=boxes[0]
        bs=boxes[1:]
        picks=np.append(picks, a)
        picks_cnt+=1

        index=np.where(iou(a, bs, isMin)<threshold)
        boxes=bs[index]

    if boxes.shape[0]==1:
        picks=np.append(picks, boxes[0])
        picks_cnt+=1
    return picks.reshape(picks_cnt, -1)


def convert_to_square(raw_boxes):
    if raw_boxes.shape[0]==0:
        return np.array([])
    w=raw_boxes[:,2]-raw_boxes[:,0]
    h=raw_boxes[:,3]-raw_boxes[:,1]
    max_hw=np.maximum(h, w)
    cx=(raw_boxes[:,2]+raw_boxes[:,0])*0.5
    cy=(raw_boxes[:,3]+raw_boxes[:,1])*0.5
    
    boxes=raw_boxes.copy()
    boxes[:,0]= cx-max_hw*0.5
    boxes[:,1]= cy-max_hw*0.5
    boxes[:,2]= cx+max_hw*0.5
    boxes[:,3]= cy+max_hw*0.5
    return boxes

if __name__=='__main__':
    a = np.array([0,0,5,5])
    bs = np.array([[1,1,10,10],[2,3,4,5]])
    print(iou(a,bs))

    bs = np.array([[1, 1, 10, 10, 0.98], [1, 1, 9, 9, 0.8], [9, 8, 13, 20, 0.7], [6, 11, 18, 17, 0.85]])
    print((-bs[:,4]).argsort())
    print(nms(bs))

    bs = np.array([[1, 1, 10, 10, 0.98], [1, 1, 9, 9, 0.8], [9, 8, 13, 20, 0.7], [6, 11, 18, 17, 0.85]])
    print(convert_to_square(bs))