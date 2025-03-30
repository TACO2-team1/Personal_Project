import torch
import cv2 as cv
import matplotlib.pyplot as plt
import sys

face_alignment_dir2 = r"C:\Users\KO SEONGHUN\Desktop\deepfake_curry\Code\face-alignment-master"
img_path= r"C:\Users\KO SEONGHUN\Desktop\deepfake_curry\Code\face-alignment-master\examples\two_face.jpeg"
sys.path.append(face_alignment_dir2)

from face_alignment2 import api as face_alignment2
import face_alignment

image = cv.imread(img_path)
frame = cv.cvtColor(image, cv.COLOR_BGR2RGB)

fa = face_alignment.FaceAlignment(face_alignment2.LandmarksType.TWO_HALF_D, device='cpu', face_detector='sfd')
det_truth=fa.get_landmarks_from_image(frame)
# print(det_truth[0].shape)

import torch.nn.functional as F

def load_image(img_path):
    # 加载图像并转为张量。
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    return img

def heatmap_loss(det_truth,pred_landmark):
    assert len(det_truth)==len(pred_landmark),"unequal length"
    loss=0
    for truth,pred in zip(det_truth,pred_landmark):
        truth=torch.tensor(truth)
        pred=torch.tensor(pred)
        cos_sim=F.cosine_similarity(truth,pred)
        loss+=cos_sim
    return loss

def fan_pgd_attack(model,image_orig, eps=0.6, alpha=1, attack_steps=1, device='cpu'):
    image_orig = image_orig.to(device)
    perturbation = torch.zeros_like(image_orig,dtype=torch.float32,requires_grad=True).to(device)

    input_data=torch.add(image_orig,perturbation)
    for step in range(attack_steps):
        pred_landmark=model.get_landmarks_from_image_pgd((input_data).clamp(0, 255))
        #self.face_detector.detect_from_image(image.copy()) 会不会影响梯度??
        loss = heatmap_loss(det_truth)
        print(loss)
        #清除模型参数的梯度
        model.face_alignment_net.zero_grad()
        
        loss.backward(retain_graph=True)
        # 梯度更新
        grad=perturbation.grad.sign()

        perturbation = torch.clamp(perturbation + alpha * grad,min=-eps,max=eps).detach()
        perturbation.requires_grad_()
        # print(perturbation.grad)   #None,因此不用puerturbation.grad.zero_()
        input_data = torch.clamp(torch.add(input_data,perturbation),min=0,max=255)

    return input_data.detach()

fa2 = face_alignment2.FaceAlignment(face_alignment2.LandmarksType.TWO_HALF_D, device='cpu', face_detector='sfd')
img=load_image(img_path)
fan_pgd_attack(fa2,img)

# def pgd_attack(model, image_orig, eps=0.6, alpha=1, attack_steps=15, device='cpu'):
#     image_orig = image_orig.to(device)
#     perturbation = torch.zeros_like(image_orig,dtype=torch.float32,requires_grad=True).to(device)

#     input_data=torch.add(image_orig,perturbation)
#     for step in range(attack_steps):
#         pred_result = model.detect_from_image((input_data).clamp(0, 255))
#         pred_box, pred_conf = [], []

#         for result in pred_result:
#             pred_box.append(result[:4])
#             pred_conf.append(result[4])
#         # print(pred_box)
#         # print(pred_conf)
        
#         potential_true_index,potential_false_index=classify_index_by_iou(pred_box,ground_truth_box,thresh_IOU=0.3)
#         print(potential_true_index)
#         #TODO:没有potential_ture_index，停止循环
#         if not potential_true_index:
#             print("no potential_true_index anymore")
#             break
#         # 计算损失
#         loss = loss_function(pred_conf,potential_true_index,potential_false_index)
#         #清除模型参数的梯度
#         model.face_detector.zero_grad()
#         loss.backward(retain_graph=True)
#         # 梯度更新
#         grad=perturbation.grad.sign()

#         perturbation = torch.clamp(perturbation + alpha * grad,min=-eps,max=eps).detach()
#         perturbation.requires_grad_()
#         # print(perturbation.grad)   #None,因此不用puerturbation.grad.zero_()
#         input_data = torch.clamp(torch.add(input_data,perturbation),min=0,max=255)

#     return input_data.detach()
