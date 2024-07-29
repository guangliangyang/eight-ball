import os
import shutil
import yaml
from ultralytics import YOLO
import json  # 确保 json 已导入
from PIL import Image
import numpy as np  # 确保 numpy 已安装并导入
from utils.format_conversion import yolo_to_labelme

def main():
    # 获取当前脚本的绝对路径
    current_script_path = os.path.abspath(__file__)
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(current_script_path)

    # 设置路径
    image_dir = os.path.abspath(os.path.join(current_dir, '..', 'pic-extracted'))
    auto_annotation_dir = os.path.abspath(os.path.join(current_dir, '..', 'pic-auto-annotation'))
    model_dir = os.path.abspath(os.path.join(current_dir, '..', 'model'))

    # 确保目录存在
    os.makedirs(auto_annotation_dir, exist_ok=True)

    # 加载训练好的模型
    model_path = os.path.join(model_dir, 'best.pt')
    model = YOLO(model_path)

    # 从 dataset.yaml 文件中获取类别名称并创建 classes.txt 文件
    dataset_yaml_path = os.path.join(model_dir, 'dataset.yaml')
    with open(dataset_yaml_path, 'r') as file:
        dataset_yaml = yaml.safe_load(file)
    class_names = list(dataset_yaml['names'].values())

    # 每次都重新生成 classes.txt 文件
    classes_txt_path = os.path.join(auto_annotation_dir, 'classes.txt')
    with open(classes_txt_path, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

    # 创建类ID到类名的映射
    id_to_class = {i: name for i, name in enumerate(class_names)}

    # 处理没有 JSON 标注文件的前10张图片
    count = 0
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            json_path = os.path.join(image_dir, filename.replace('.jpg', '.json'))
            if not os.path.exists(json_path):
                image_path = os.path.join(image_dir, filename)

                count += 1
                if count > 10:
                    break

                # 使用模型进行推理，使用默认置信度阈值
                try:
                    results = model.predict(source=image_path, save=False)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue

                # 获取图片尺寸
                img = Image.open(image_path)
                width, height = img.size

                # 创建标注结果
                annotations = []
                for result in results:
                    boxes = result.boxes.data.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2, conf, cls = box
                        class_id = int(cls)

                        x_center = (x1 + x2) / 2 / width
                        y_center = (y1 + y2) / 2 / height
                        box_width = (x2 - x1) / width
                        box_height = (y2 - y1) / height

                        annotations.append((class_id, f"{class_id} {x_center} {y_center} {box_width} {box_height}"))

                # 按 class_id 排序
                annotations.sort(key=lambda x: x[0])
                sorted_annotations = [ann[1] for ann in annotations]

                # 保存标注结果
                txt_save_path = os.path.join(auto_annotation_dir, filename.replace('.jpg', '.txt'))
                with open(txt_save_path, 'w') as f:
                    f.write('\n'.join(sorted_annotations))

                # 生成 Labelme 格式的 JSON 文件
                labelme_data = yolo_to_labelme(txt_save_path, width, height, id_to_class)
                labelme_json_path = os.path.join(auto_annotation_dir, filename.replace('.jpg', '.json'))
                with open(labelme_json_path, 'w') as f:
                    json.dump(labelme_data, f, indent=4)

                # 复制原图到 auto_annotation_dir
                shutil.copy(image_path, auto_annotation_dir)

if __name__ == '__main__':
    main()
