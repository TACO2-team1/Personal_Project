import sys
import os

face_alignment_dir = r"C:\Users\KO SEONGHUN\Downloads\face-alignment-master\face-alignment-master\face_alignment2"

print(face_alignment_dir)
# 将 face_alignment 目录添加到 sys.path
sys.path.append(face_alignment_dir)
print(sys.path)

from detection.sfd import sfd_detector
# from face_alignment2.detection.sfd import sfd_detector

s3fd_model0=sfd_detector.SFDDetector("cpu",filter_threshold=0.9)
img_path=r"C:\Users\KO SEONGHUN\Downloads\face-alignment-master\face-alignment-master\examples\two_face.jpeg"
# #ground_truth
ground_truth_box=s3fd_model0.detect_from_image(img_path)
print(ground_truth_box)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# face_detector = sfd_detector.SFDDetector(device=device)

# # 加载图像和执行人脸检测的代码（与前面类似）
# img_path = 'path_to_image.jpg'  # 替换为你的图片路径
# image = cv2.imread(img_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # 执行人脸检测
# detected_faces = face_detector.detect_from_image(image_rgb)

# # 输出检测结果
# print("Detected faces:", detected_faces)
