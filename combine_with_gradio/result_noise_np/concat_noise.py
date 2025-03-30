import numpy as np
import cv2
import matplotlib.pyplot as plt

# 定义文件路径
image_path = r"C:\Users\KO SEONGHUN\Desktop\deepfake_curry\Code\combine_with_gradio\result_noise_np\young_man.jpg"
adv_path=r"C:\Users\KO SEONGHUN\Desktop\deepfake_curry\Code\combine_with_gradio\result_noise_np\adv_img.jpg"

output_path="merged_noise_img.jpg"
noise_paths = [
    r"C:\Users\KO SEONGHUN\Desktop\deepfake_curry\Code\combine_with_gradio\result_noise_np\model1_noise.npy",
    r"C:\Users\KO SEONGHUN\Desktop\deepfake_curry\Code\combine_with_gradio\result_noise_np\model2_noise.npy",
]
image = cv2.imread(image_path)
adv_img=cv2.imread(adv_path)
adv_img=cv2.resize(adv_img, (image.shape[1], image.shape[0]))

print(image.shape)
print(adv_img.shape)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
delta=adv_img-image


noise_list = [np.load(path) for path in noise_paths]
noise_list[0]=cv2.resize(noise_list[0],(image.shape[1],image.shape[0]))
print(noise_list[0])
# for i, noise in enumerate(noise_list):
print(f"Noise {1} shape: {delta.shape}")
plt.imshow(delta)  # 使用 gray colormap 显示图像
# plt.imshow(noise, cmap='gray')  # 使用 gray colormap 显示图像
plt.axis('off')

plt.show()

combined_noise = sum(noise_list)

# 确保图片和噪声类型一致
noise_image = np.clip(image + combined_noise, 0, 255).astype(np.uint8)
cv2.imwrite(output_path, cv2.cvtColor(noise_image, cv2.COLOR_RGB2BGR))
# 5. 显示图片
plt.figure(figsize=(10, 5))
# 原图
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# 加噪图
plt.subplot(1, 2, 2)
plt.imshow(noise_image)
plt.title("Image with Noise")
plt.axis("off")

plt.tight_layout()
plt.show()
# cv2.imwrite(noise_list)
