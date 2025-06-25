import cv2

# 读取图像
image = cv2.imread('path/to/your/image.jpg')

# 假设我们有一个点的坐标 (x, y)
x, y = 100.5, 50.2  # 这些值可能是浮点数

# 确保坐标是整数
x = int(x)
y = int(y)

# 绘制点
cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

# 获取图像的尺寸
height, width, _ = image.shape

# 计算底部中间点的坐标
bottom_middle_x = width // 2
bottom_middle_y = height - 1  # 确保不超出图像边界

# 绘制线
cv2.line(image, (x, y), (bottom_middle_x, bottom_middle_y), color=(255, 0, 0), thickness=2)

# 显示图像
cv2.imshow('Image with Point and Line', image)
cv2.waitKey(0)
cv2.destroyAllWindows()