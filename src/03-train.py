import os
import json
import shutil
from ultralytics import YOLO
from PIL import Image
from collections import Counter

def get_bounding_box(points):
    x_coords, y_coords = zip(*points)
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

def main():
    # 获取当前脚本的绝对路径
    current_script_path = os.path.abspath(__file__)

    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(current_script_path)

    # 设置路径
    image_dir = os.path.abspath(os.path.join(current_dir, '..', 'pic-extracted'))
    dataset_dir = os.path.abspath(os.path.join(current_dir, '..', 'dataset'))
    model_dir = os.path.abspath(os.path.join(current_dir, '..', 'model'))

    # 打印路径以进行验证
    print(f"原始图片目录: {image_dir}")
    print(f"数据集目录: {dataset_dir}")
    print(f"模型目录: {model_dir}")

    # 确保目录存在
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 准备数据并统计类别
    data = []
    class_names = set()

    # 第一遍扫描，收集所有唯一的类别
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            json_path = os.path.join(image_dir, filename.replace('.jpg', '.json'))
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    label_data = json.load(f)
                for shape in label_data['shapes']:
                    class_name = shape['label']
                    shape_type = shape['shape_type']
                    combined_class = f"{class_name}_{shape_type}"
                    class_names.add(combined_class)

    # 为每个类别分配一个唯一的ID
    class_to_id = {name: i for i, name in enumerate(sorted(class_names))}

    # 第二遍扫描，生成标签文件并复制图片到dataset目录
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, filename)
            json_path = os.path.join(image_dir, filename.replace('.jpg', '.json'))

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    label_data = json.load(f)

                img = Image.open(image_path)
                width, height = img.size

                labels = []
                for shape in label_data['shapes']:
                    class_name = shape['label']
                    shape_type = shape['shape_type']
                    points = shape['points']

                    combined_class = f"{class_name}_{shape_type}"
                    class_id = class_to_id[combined_class]

                    if shape_type == 'circle':
                        center_x, center_y = points[0]
                        radius_point_x, radius_point_y = points[1]
                        radius = ((radius_point_x - center_x) ** 2 + (radius_point_y - center_y) ** 2) ** 0.5
                        x1, y1 = center_x - radius, center_y - radius
                        x2, y2 = center_x + radius, center_y + radius
                    else:  # line 或其他形状
                        x1, y1, x2, y2 = get_bounding_box(points)

                    center_x = (x1 + x2) / (2 * width)
                    center_y = (y1 + y2) / (2 * height)
                    box_width = abs(x2 - x1) / width
                    box_height = abs(y2 - y1) / height

                    labels.append(f"{class_id} {center_x} {center_y} {box_width} {box_height}")

                # 复制图片到dataset目录
                new_image_path = os.path.join(dataset_dir, filename)
                shutil.copy2(image_path, new_image_path)

                # 保存标签文件到dataset目录
                txt_path = os.path.join(dataset_dir, filename.replace('.jpg', '.txt'))
                with open(txt_path, 'w') as f:
                    f.write('\n'.join(labels))

                data.append((new_image_path, txt_path))

    # 准备数据集配置文件
    dataset_yaml = os.path.join(model_dir, 'dataset.yaml')
    with open(dataset_yaml, 'w') as f:
        f.write(f"train: {dataset_dir}\n")
        f.write(f"val: {dataset_dir}\n")
        f.write(f"nc: {len(class_to_id)}\n")
        f.write("names:\n")
        for name, id in class_to_id.items():
            f.write(f"  {id}: '{name}'\n")

    print(f"检测到的类别数量: {len(class_to_id)}")
    print("类别名称和ID:")
    for name, id in class_to_id.items():
        print(f"  {id}: {name}")

    # 验证数据集
    print("验证数据集:")
    print(f"处理的图片数量: {len(data)}")
    print(f"标签文件数量: {len([f for f in os.listdir(dataset_dir) if f.endswith('.txt')])}")

    # 检查几个随机的标签文件
    import random

    label_files = [f for f in os.listdir(dataset_dir) if f.endswith('.txt')]
    for _ in range(min(5, len(label_files))):
        random_label = random.choice(label_files)
        print(f"\n内容of {random_label}:")
        with open(os.path.join(dataset_dir, random_label), 'r') as f:
            print(f.read())

    # 训练模型
    model = YOLO('yolov8n.yaml')  # 创建一个新的YOLOv8n模型
    try:
        results = model.train(data=dataset_yaml, epochs=100, imgsz=640, batch=16, save=True, project=model_dir)

        # 获取最新的train*文件夹
        train_folders = [f for f in os.listdir(model_dir) if
                         f.startswith('train') and os.path.isdir(os.path.join(model_dir, f))]
        latest_train_folder = max(train_folders, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))

        # 删除其他旧的train*文件夹
        for folder in train_folders:
            if folder != latest_train_folder:
                folder_path = os.path.join(model_dir, folder)
                shutil.rmtree(folder_path)

        # 复制weights/best.pt到model文件夹下
        best_weights_path = os.path.join(model_dir, latest_train_folder, 'weights', 'best.pt')
        shutil.copy2(best_weights_path, os.path.join(model_dir, 'best.pt'))

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()