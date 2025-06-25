# author:yrq
# date:2024/8/8 20:05
# comment:这是一些小工具

import os,logging
import cv2

def clear_files_in_directory(directory):# 指定要清空的文件夹路径
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                logging.info("删除{}".format(file_path))
                os.remove(file_path)  # 删除文件
        print(f"所有文件内容已清空: {directory}")
    except Exception as e:
        print(f"发生错误: {e}")

#调试的相关工具
def rgb_to_hsv_opencv(r, g, b):
    """
    将RGB颜色值转换为OpenCV格式的HSV颜色值。

    参数:
    r, g, b - 分别表示红色、绿色和蓝色的值，范围为0-255。

    返回:
    h, s, v - 分别表示色相(Hue)，饱和度(Saturation)，亮度(Value)的值。
              H的范围为0-179，S和V的范围为0-255。
    """

    r, g, b = r / 255.0, g / 255.0, b / 255.0

    max_val = max(r, g, b)
    min_val = min(r, g, b)
    delta = max_val - min_val

    # 计算色相H
    if delta == 0:
        h = 0
    elif max_val == r:
        h = (60 * ((g - b) / delta) + 360) % 360
    elif max_val == g:
        h = (60 * ((b - r) / delta) + 120) % 360
    elif max_val == b:
        h = (60 * ((r - g) / delta) + 240) % 360

    h = int(h / 2)  # 缩放色相范围到0-179

    # 计算饱和度S
    s = 0 if max_val == 0 else int((delta / max_val) * 255)

    # 计算亮度V
    v = int(max_val * 255)

    return h, s, v

def binary_thresholding(image_path, output_path, threshold=127):
    # 读取图片
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 应用阈值处理
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # 保存二值化图片
    cv2.imwrite(output_path, binary_image)

def process_directory(source_dir, output_dir, threshold=127):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历源目录中的所有文件
    for filename in os.listdir(source_dir):
        # 构建完整的文件路径
        file_path = os.path.join(source_dir, filename)
        output_file_path = os.path.join(output_dir, filename)

        # 对每个文件进行二值化处理并保存到输出目录
        binary_thresholding(file_path, output_file_path, threshold)


def recode(data):
    data_List=[]
    for cube in data.get("cube",[]):
        data_List.append((cube[0],cube[1],-1))
    for qrcode in data.get("qrcode",[]):
        data_List.append([qrcode[0][0],qrcode[0][1],qrcode[1]])

    return data_List
# # 替换为您的图像文件路径
# image_path = 'QRcode/train/1/1.jpg'
#
# # 读取图像
# image = cv2.imread(image_path)
# cv2.imshow("image",image)
# cv2.waitKey(0)
# # 检查图像是否成功加载
# if image is None:
#     print(f"Error: Unable to load image at {image_path}")
# else:
#     # 获取图像的尺寸和通道数
#     height, width, channels = image.shape
#
#     # 打印通道数
#     print(f"The image has {channels} channels.")

# from sklearn.datasets import load_digits
#
# # 加载数据集
# digits = load_digits()
#
# # 查看数据集的形状
# print(digits.data.shape)  # 输出 (1797, 64)
# print(digits.target.shape)  # 输出 (1797,)
#
# # 查看数据集的键
# print(digits.keys())
#
# # 查看前几个样本
# print(digits.data[:5])  # 输出前 5 个样本的像素值
