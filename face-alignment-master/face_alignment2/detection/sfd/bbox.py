import math
import numpy as np
import torch

# def nms(dets, thresh):
#     if 0 == len(dets):
#         return []
#     x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     order = scores.argsort()[::-1]

#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
#         xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])

#         w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
#         ovr = w * h / (areas[i] + areas[order[1:]] - w * h)

#         inds = np.where(ovr <= thresh)[0]
#         order = order[inds + 1]

#     return keep

def nms(dets, thresh):
    if len(dets) == 0:
        return torch.tensor([], dtype=torch.long)
    
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort(descending=True)  # Sort scores in descending order

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1 + 1, min=0)
        h = torch.clamp(yy2 - yy1 + 1, min=0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = torch.where(ovr <= thresh)[0]
        order = order[inds + 1]
    
    return torch.tensor(keep, dtype=torch.long)




def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = np.log(g_wh) / variances[1]

    # return target for smooth_l1_loss
    return np.concatenate([g_cxcy, g_wh], 1)  # [num_priors,4]


# def decode(loc, priors, variances):
#     """Decode locations from predictions using priors to undo
#     the encoding we did for offset regression at train time.
#     Args:
#         loc (tensor): location predictions for loc layers,
#             Shape: [num_priors,4]
#         priors (tensor): Prior boxes in center-offset form.
#             Shape: [num_priors,4].
#         variances: (list[float]) Variances of priorboxes
#     Return:
#         decoded bounding box predictions
#     """

#     boxes = np.concatenate((
#         priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
#         priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
#     boxes[:, :2] -= boxes[:, 2:] / 2
#     boxes[:, 2:] += boxes[:, :2]
#     return boxes

# def decode(loc, priors, variances):
#     """Decode bounding box predictions using prior boxes."""
#     boxes = torch.cat((
#         priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
#         priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])
#     ), dim=1)
#     boxes[:, :2] -= boxes[:, 2:] / 2
#     boxes[:, 2:] += boxes[:, :2]
#     return boxes

def decode(loc, priors, variances):
    # Ensure the dimensions of `loc` and `priors` match correctly
    if loc.dim() == 3 and loc.size(0) == 1:  # If `loc` has batch dimension
        loc = loc.squeeze(0)  # Remove batch dimension for processing
    
    # Decode bounding box predictions
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],  # Center coordinates
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])        # Width and height
    ), dim=1)

    # Convert to corner coordinates
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes