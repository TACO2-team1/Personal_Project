import torch
import cv2 as cv
import matplotlib.pyplot as plt
import sys
import time
import os
from torchvision import transforms as T
from face_alignment.detection.sfd import sfd_detector as sfd_detector0
import numpy as np

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
face_alignment2_dir = os.path.join(dir_path,'face_alignment2')
sys.path.append(face_alignment2_dir)
from detection.sfd import sfd_detector

def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # 判断是否有重叠区域
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # 没有交集

    # 交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # 并集面积
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    return iou

#IOU 阈值
def classify_index_by_iou(pred_box,ground_truth_box,thresh_IOU = 0.1):
    potential_true_index = []
    potential_false_index = []

    for i,pred in enumerate(pred_box):    
        match_found=False
        for truth in ground_truth_box:
            if calculate_iou(pred,truth)>thresh_IOU:
                potential_true_index.append(i)
                match_found=True
                break
        if not match_found:
            potential_false_index.append(i)
    return potential_true_index,potential_false_index

def loss_function(pred_conf, potential_true_index, potential_false_index):
    true_detection_loss = torch.tensor(0.0, dtype=torch.float32, device='cpu')
    false_detection_loss = torch.tensor(0.0, dtype=torch.float32, device='cpu')
    for i in potential_true_index:
        true_detection_loss += torch.log(1 - pred_conf[i])
    for j in potential_false_index:
        false_detection_loss += torch.log(pred_conf[j])
    loss = torch.add(true_detection_loss,false_detection_loss)
    print(loss)
    return loss

def pgd_attack(model, image_orig, ground_truth_box, eps, alpha, attack_steps, device='cpu'):
    image_orig = image_orig.to(device)
    perturbation = torch.zeros_like(image_orig,dtype=torch.float32,requires_grad=True).to(device)

    input_data=torch.add(image_orig,perturbation)
    for step in range(attack_steps):
        print(f"step: {step}")
        pred_result = model.detect_from_image((input_data).clamp(0, 255))
        pred_box, pred_conf = [], []
        #threshold count=300
        print(len(pred_result))
        for result in pred_result[:300]:
            pred_box.append(result[:4])
            pred_conf.append(result[4])
        
        potential_true_index,potential_false_index=classify_index_by_iou(pred_box,ground_truth_box)
        print(potential_true_index)

        # if not potential_true_index:
            # print("no potential_true_index anymore")
            # break
        # 计算损失
        loss = loss_function(pred_conf,potential_true_index,potential_false_index)
        #清除模型参数的梯度
        model.face_detector.zero_grad()
        loss.backward(retain_graph=True)
        # 梯度更新
        grad=perturbation.grad.sign()

        perturbation = torch.clamp(perturbation + alpha * grad,min=-eps,max=eps).detach()
        perturbation.requires_grad_()
        # print(perturbation.grad)   #None,因此不用puerturbation.grad.zero_()
        input_data = torch.clamp(torch.add(input_data,perturbation),min=0,max=255)

    return input_data.detach()

from PIL import Image
def load_image(img_path):
    # 加载图像并转为张量
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    return img

transform = T.Compose([
    # T.Resize(256),
    T.ToTensor(),
    # T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

def load_image2(img_path):
    image= Image.open(img_path).convert("RGB")
    img=transform(image) #自动归一化
    img=img*255.0
    img = img.unsqueeze(0).float()
    return img
# eps alpha steps
# 0.5 1~3 15 good
EPS=1.3
ALPHA=3
ATTACK_STEPS=12

def s3fd_pgd_attack(input_image_path, output_image_path,eps=EPS,alpha=ALPHA,attack_steps=ATTACK_STEPS):
    t0=time.time()
    # 加载 Ground Truth box
    s3fd_model0 = sfd_detector0.SFDDetector("cpu", filter_threshold=0.9)
    ground_truth_box = s3fd_model0.detect_from_image(input_image_path)
    print(ground_truth_box)
    draw_ground_truth(input_image_path,ground_truth_box)
    return
    # 加载待攻击的图像
    s3fd_model = sfd_detector.SFDDetector("cpu", filter_threshold=0.0001)
    image_orig = load_image(input_image_path)

    # 执行 PGD 攻击
    adv_image = pgd_attack(s3fd_model, image_orig, ground_truth_box,eps,alpha,attack_steps, device="cpu")

    # 保存对抗样本
    adv_image2 = adv_image.squeeze().permute(1, 2, 0).cpu().numpy().astype('uint8')
    cv.imwrite(output_image_path, cv.cvtColor(adv_image2, cv.COLOR_RGB2BGR))
    t1=time.time()
    return adv_image2
def draw_ground_truth(img_path,ground_truth_box):
    image = cv.imread(img_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    for box in ground_truth_box:
        x1, y1, x2, y2, confidence = box  
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv.rectangle(image, (x1, y1), (x2, y2), color=(0,255, 0), thickness=2) 
        text = f"Conf: {confidence:.1f}"
        cv.putText(image, text, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title("ground_truth Detection Results")
    plt.show()
def draw_adv_img(save_path,adv_image2):
    s3fd_model2 = sfd_detector0.SFDDetector("cpu", filter_threshold=0.8)
    result=s3fd_model2.detect_from_image(save_path)
    for detection in result:
        x1, y1, x2, y2, confidence = detection
        # 绘制矩形框
        adv_image_np = cv.rectangle(
            adv_image2,
            (int(x1), int(y1)),  # 左上角
            (int(x2), int(y2)),  # 右下角
            (0,255, 0),         # 蓝色框
            2                    # 框线厚度
        )
        # 绘制置信度
        adv_image_np = cv.putText(
            adv_image2,
            f"{confidence:.2f}",
            (int(x1), int(y1) - 10),  # 文字位置
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,                      # 字体大小
            (255, 0, 0),              # 蓝色文字
            1                         # 文字厚度
        )

    # 使用 matplotlib 显示图像
    plt.imshow(adv_image_np)
    plt.axis("off")
    plt.title("Adversarial Image with Bounding Boxes")
    plt.show()
# def contcat(img_path):
#     # img_path = r"C:\Users\KO SEONGHUN\Desktop\deepfake_curry\Code\combine_with_gradio\young_man.jpg"
#     adv_img_path = r"C:\Users\KO SEONGHUN\Desktop\deepfake_curry\Code\combine_with_gradio\result_image\model2_adv_img.jpg"
#     adv_img2 = s3fd_pgd_attack(img_path, adv_img_path)
#     adv_img2 = np.ascontiguousarray(adv_img2)
#     draw_adv_img(adv_img_path, adv_img2)

#     # 노이즈
#     original_img = np.array(Image.open(img_path).convert("RGB"))
#     adv_img = np.array(Image.open(adv_img_path).convert("RGB"))
#     noise = adv_img.astype(np.float32) - original_img.astype(np.float32)
#     np.save(r"C:\Users\KO SEONGHUN\Desktop\deepfake_curry\Code\combine_with_gradio\result_noise_np\model2_noise.npy", noise)
#     return noise
    # loaded_noise = np.load("model2_noise.npy")
    # normalized_noise = ((noise - noise.min()) / (noise.max() - noise.min()) * 255).astype(np.uint8)
    # print(loaded_noise)
if __name__=="__main__":
    #jpg만 가능 png 안됨
    # img_path="two_face.jpeg"
    img_path = r"C:\Users\KO SEONGHUN\Desktop\deepfake_curry\Code\combine_with_gradio\final_noise_image.jpg"
    adv_img_path=r"C:\Users\KO SEONGHUN\Desktop\deepfake_curry\Code\combine_with_gradio\result_image\adv_img.jpg"
    
    adv_img2=s3fd_pgd_attack(img_path,adv_img_path)
    adv_img2 = np.ascontiguousarray(adv_img2)
    draw_adv_img(adv_img_path,adv_img2)
    
    #노이즈
    original_img = np.array(Image.open(img_path).convert("RGB"))
    adv_img = np.array(Image.open(adv_img_path).convert("RGB")) 
    noise = adv_img.astype(np.float32) - original_img.astype(np.float32)
    np.save(r"C:\Users\KO SEONGHUN\Desktop\deepfake_curry\Code\combine_with_gradio\result_noise_np\model2_noise.npy",noise)
    # normalized_noise = ((noise - noise.min()) / (noise.max() - noise.min()) * 255).astype(np.uint8)  
    # 노이즈 시각화
    # plt.figure(figsize=(6, 6))
    # plt.imshow(normalized_noise)
    # plt.title("Noise (Difference)")
    # plt.axis("off")
    # plt.show()
