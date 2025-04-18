import cv2
import tensorflow as tf

img=cv2.imread("two_face.jpeg")
w,h,_=img.shape
scale=0.71
w,h=int(w*scale),int(h*scale)
scale_img = cv2.resize(img, (w, h))
tf_img = tf.image.resize(img, size=(h,w), method='bilinear')
tf_img_numpy = tf_img.numpy().astype(scale_img.dtype)

print(tf_img_numpy)
print("——————")
print(scale_img)
    
