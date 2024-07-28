import os
import json
from ultralytics import YOLO
from PIL import Image

# 设置路径
image_dir = os.path.join('..', 'pic-extracted')
label_dir = image_dir  # 假设标注文件与图片在同一目录
model_dir = os.path.join('..', 'model')

# 创建模型保存目录
os.makedirs(model_dir, exist_ok=True)

# 准备数据
data = []
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):
        image_path = os.path.join(image_dir, filename)
        json_path = os.path.join(label_dir, filename.replace('.jpg', '.json'))

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                label_data = json.load(f)

            img = Image.open(image_path)
            width, height = img.size

            labels = []
            for shape in label_data['shapes']:
                class_name = shape['label']
                points = shape['points']
                x1, y1 = points[0]
                x2, y2 = points[1]

                # 转换为YOLO格式 (center_x, center_y, width, height)
                center_x = (x1 + x2) / (2 * width)
                center_y = (y1 + y2) / (2 * height)
                box_width = abs(x2 - x1) / width
                box_height = abs(y2 - y1) / height

                labels.append(f"{class_name} {center_x} {center_y} {box_width} {box_height}")

            # 保存YOLO格式的标签文件
            txt_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))
            with open(txt_path, 'w') as f:
                f.write('\n'.join(labels))

            data.append((image_path, txt_path))

# 准备数据集配置文件
dataset_yaml = os.path.join(model_dir, 'dataset.yaml')
with open(dataset_yaml, 'w') as f:
    f.write(f"train: {image_dir}\n")
    f.write(f"val: {image_dir}\n")  # 这里使用相同的数据作为验证集，您可能需要分割数据集
    f.write("nc: 1\n")  # 类别数量，请根据实际情况修改
    f.write("names: ['object']\n")  # 类别名称，请根据实际情况修改

# 训练模型
model = YOLO('yolov8n.yaml')  # 创建一个新的YOLOv8n模型
results = model.train(data=dataset_yaml, epochs=100, imgsz=640, batch=16, save=True, project=model_dir)

# 保存模型
model.save(os.path.join(model_dir, 'best.pt'))