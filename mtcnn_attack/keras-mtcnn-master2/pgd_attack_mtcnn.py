import sys
import tools_matrix as tools
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from MTCNN import create_Kao_Onet, create_Kao_Rnet, create_Kao_Pnet
import tensorflow as tf
import warnings
# import tensorflow_graphics.math.interpolation.trilinear as trilinear

# 禁用 Keras 相关的警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*inputs.*')

Pnet = create_Kao_Pnet(r'12net.h5')
Rnet = create_Kao_Rnet(r'24net.h5')
Onet = create_Kao_Onet(r'48net.h5')  # will not work. caffe and TF incompatible
bce_loss = tf.keras.losses.BinaryCrossentropy()
#init
img_ct=1
epsilon = 0.4
alpha = 0.008
num_iter = 50
tv_weight = 1
tv_loss=False
ground_truth=[]
THRESHOLD=0.9
save_img =True


def total_variation_loss(image):
    # 计算水平和垂直方向的差分
    x_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    y_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    
    return tf.reduce_mean(tf.square(x_var)) + tf.reduce_mean(tf.square(y_var))

def pgd_attack(model, x, epsilon, alpha, num_iter, tv_weight=tv_weight,tv_loss=tv_loss,threshold=THRESHOLD):
    x_adv = tf.identity(x)
    x_adv = x_adv + tf.random.uniform(x_adv.shape, minval=-epsilon, maxval=epsilon)
    x_adv = tf.clip_by_value(x_adv, -1, 1)
    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            output_classifier, bbox_regress = model(x_adv)
            pred_conf = output_classifier[0,...,1]

            # 组合损失：对抗损失 + TV损失
            # adv_loss = bce_loss(tf.zeros_like(pred_conf), pred_conf)
            target_conf = tf.where(pred_conf > threshold, tf.zeros_like(pred_conf), tf.ones_like(pred_conf))
            adv_loss = bce_loss(target_conf, pred_conf)
            if tv_loss:
                tv_loss = total_variation_loss(x_adv)
                total_loss = adv_loss + tv_weight * tv_loss
            else:
                total_loss = adv_loss
            
        grad = tape.gradient(total_loss, x_adv)
        x_adv = x_adv - alpha * tf.sign(grad)
        
        x_adv = tf.clip_by_value(x_adv, x - epsilon, x + epsilon)
        x_adv = tf.clip_by_value(x_adv, -1, 1)
    
    return x_adv-x

def multiple_pgd_attack(img, epsilon=epsilon, alpha=alpha, num_iter=num_iter,img_ct=img_ct):
    t0 = time.time()
    # 归一化
    caffe_img = (img.copy() - 127.5) / 127.5
    img_t=caffe_img.copy()
    origin_h, origin_w, ch = caffe_img.shape
    # 只需要n个scales
    scales = tools.calculateScales_n(img,img_ct)
    for scale in scales:
        ws = int(origin_w * scale)
        hs = int(origin_h * scale)
        scale_img = cv2.resize(caffe_img, (ws, hs))

        caffe_img_tensor = tf.convert_to_tensor(scale_img, dtype=tf.float32)
        caffe_img_tensor = tf.expand_dims(caffe_img_tensor, axis=0) # 加批处理维度

        delta = pgd_attack(Pnet, caffe_img_tensor, epsilon, alpha, num_iter,ground_truth)
        delta=tf.squeeze(delta,axis=0)

        delta_resized = tf.image.resize(delta, size=(origin_h, origin_w), method='bilinear')
        # delta_resized = tf.image.resize(delta, size=(origin_h, origin_w), method='nearest') #噪音更显眼

        #合并不同scale的噪音
        img_t += delta_resized
        img_t = tf.clip_by_value(img_t, -1, 1)
    #恢复归一化
    restored_img_t=img_t*127.5+127.5
    restored_img = np.clip(restored_img_t, 0, 255).astype(np.uint8)
    t1 = time.time()
    print(f"pgd_attack_time: {t1-t0}")
    return restored_img

def detectFace(img, threshold):
    caffe_img = (img.copy() - 127.5) / 127.5
    origin_h, origin_w, ch = caffe_img.shape
    scales = tools.calculateScales(img)
    out = []
    t0 = time.time()

    for scale in scales:
        hs = int(origin_h * scale)
        ws = int(origin_w * scale)
        scale_img = cv2.resize(caffe_img, (ws, hs))
        input = scale_img.reshape(1, *scale_img.shape)
        ouput = Pnet.predict(input)  # .transpose(0,2,1,3) should add, but seems after process is wrong then.
        out.append(ouput)
    image_num = len(scales)
    rectangles = []
    for i in range(image_num):
        cls_prob = out[i][0][0][:, :,1]  # i = #scale, first 0 select cls score, second 0 = batchnum, alway=0. 1 one hot repr
        roi = out[i][1][0]
        out_h, out_w = cls_prob.shape
        out_side = max(out_h, out_w)
        cls_prob = np.swapaxes(cls_prob, 0, 1)
        roi = np.swapaxes(roi, 0, 2)
        rectangle = tools.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
        rectangles.extend(rectangle)
    rectangles = tools.NMS(rectangles, 0.7, 'iou')

    if len(rectangles) == 0:
        return rectangles

    crop_number = 0
    out = []
    predict_24_batch = []
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img, (24, 24))
        predict_24_batch.append(scale_img)
        crop_number += 1

    predict_24_batch = np.array(predict_24_batch)

    out = Rnet.predict(predict_24_batch)

    cls_prob = out[0]  # first 0 is to select cls, second batch number, always =0
    cls_prob = np.array(cls_prob)  # convert to numpy
    roi_prob = out[1]  # first 0 is to select roi, second batch number, always =0
    roi_prob = np.array(roi_prob)
    rectangles = tools.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])

    if len(rectangles) == 0:
        return rectangles
    
    crop_number = 0
    predict_batch = []
    for rectangle in rectangles:
        # print('calculating net 48 crop_number:', crop_number)
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img, (48, 48))
        predict_batch.append(scale_img)
        crop_number += 1

    predict_batch = np.array(predict_batch)

    output = Onet.predict(predict_batch)
    cls_prob = output[0]
    roi_prob = output[1]
    pts_prob = output[2]  # index
    # rectangles = tools.filter_face_48net_newdef(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h,
    #                                             threshold[2])
    rectangles = tools.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
    t3 = time.time()
    print ('time for detecting: ', t3-t0)

    return rectangles

img = cv2.imread('two_face.jpeg')
h,w,ch = img.shape
adv_img_bgr = multiple_pgd_attack(img)
# adv_img_rgb = cv2.cvtColor(adv_img_bgr, cv2.COLOR_RGB2BGR)

threshold = [0.6, 0.6, 0.7]
rectangles = detectFace(img, threshold)
# for rect in rectangles:
#     for j in rect:
#         if j%2==0:
#             (j-w/2)/(w/2)
print(ground_truth)
if save_img is True:
    cv2.imwrite(f'result_img/new_adv_img_{img_ct}_{epsilon}_{alpha}_{num_iter}.jpg', adv_img_bgr)
    for i in range(1):
        rectangles = detectFace(adv_img_bgr, threshold)
        draw = adv_img_bgr.copy()

        for rectangle in rectangles:
            if rectangle is not None:
                W = -int(rectangle[0]) + int(rectangle[2])
                H = -int(rectangle[1]) + int(rectangle[3])
                paddingH = 0.01 * W
                paddingW = 0.02 * H
                crop_img = adv_img_bgr[int(rectangle[1]+paddingH):int(rectangle[3]-paddingH), int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                if crop_img is None:
                    continue
                if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                    continue
                cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 1)

                for i in range(5, 15, 2):
                    cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))
        cv2.imshow("test", draw)
        cv2.waitKey(0)

        cv2.imwrite(
            f'result_img/new_adv_img_predicted_{img_ct}_{epsilon}_{alpha}_{num_iter}_{tv_weight}_{THRESHOLD}_{str(tv_loss)}.jpg', draw)


