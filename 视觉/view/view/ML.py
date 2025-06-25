import os
import cv2
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from Settings import Settings
import joblib

settings=Settings()
def load_data(data_dir):
    X = []
    y = []
    default_size =settings.QRcodeIdentifierSettings.get("default_size")
    # 遍历文件夹 0 到 5
    for label in range(6):
        folder_path = os.path.join(data_dir, str(label))

        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist.")
            continue

        # 遍历文件夹中的每个文件
        for filename in os.listdir(folder_path):
            if filename.endswith('.png') or filename.endswith('.jpg'):  # 根据需要更改文件扩展名
                img_path = os.path.join(folder_path, filename)

                image = cv2.imread(img_path)
                # cv2.imshow("image", image)
                # cv2.waitKey(0)
                image = cv2.resize(image, default_size)  # 假设我们将所有图像缩放到


                X.append(image.flatten())
                y.append(int(label))

    # 转换为 NumPy 数组
    X = np.array(X)
    y = np.array(y)

    print(X)
    return X, y

def train_model(data, labels, n_samples=50):
    # 将标签文本转换为整数
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # 应用t-SNE进行降维
    pca = PCA(n_components=10, random_state=42)
    data_pca = pca.fit_transform(data)
    print(data_pca.shape)

    # # 保存t-SNE模型
    joblib.dump(pca, 'pca.joblib')

    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_pca, labels, test_size=0.8, random_state=42)

    # 训练分类器
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 评估模型
    y_pred = clf.predict(X_test)
    target_names = le.classes_.astype(str)
    print(classification_report(y_test, y_pred, target_names=target_names))



    # 返回训练好的模型和标签编码器
    return clf, le

def visualize_results(X_tsne, y):
    # 定义颜色列表
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']

    # 创建图形
    plt.figure(figsize=(10, 10))

    # 设置x和y轴的限制
    plt.xlim(X_tsne[:, 0].min(), X_tsne[:, 0].max() + 1)
    plt.ylim(X_tsne[:, 1].min(), X_tsne[:, 1].max() + 1)

    # 遍历每个数据点
    for i in range(len(X_tsne)):
        # 将数据点的标签绘制成文本
        plt.text(X_tsne[i, 0], X_tsne[i, 1], str(y[i]),
                 color=colors[y[i]], fontdict={'weight': 'bold', 'size': 9})

    # 设置坐标轴标签
    plt.xlabel("t-SNE feature 0")
    plt.ylabel("t-SNE feature 1")

    # 显示图形
    plt.show()

def main():
    data_dir = 'QRcode/train'
    data, labels = load_data(data_dir)
    model, label_encoder = train_model(data, labels)
    model_filename = 'model.pkl'
    joblib.dump(model,model_filename)
    print(f"Model saved to {model_filename}")

    # 可视化训练集
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(data)
    visualize_results(data_tsne, labels)

if __name__ == "__main__":
    main()