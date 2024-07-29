import json
import os
from PIL import Image


def get_bounding_box(points):
    x_coords, y_coords = zip(*points)
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)


def labelme_to_yolo(json_path, img_width, img_height, class_to_id):
    with open(json_path, 'r') as f:
        label_data = json.load(f)

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

        center_x = (x1 + x2) / (2 * img_width)
        center_y = (y1 + y2) / (2 * img_height)
        box_width = abs(x2 - x1) / img_width
        box_height = abs(y2 - y1) / img_height

        labels.append(f"{class_id} {center_x} {center_y} {box_width} {box_height}")

    return labels


def yolo_to_labelme(txt_path, img_width, img_height, id_to_class):
    shapes = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        center_x = float(parts[1]) * img_width
        center_y = float(parts[2]) * img_height
        box_width = float(parts[3]) * img_width
        box_height = float(parts[4]) * img_height

        x1 = center_x - box_width / 2
        y1 = center_y - box_height / 2
        x2 = center_x + box_width / 2
        y2 = center_y + box_height / 2

        label, shape_type = id_to_class[class_id].rsplit('_', 1)
        if shape_type == 'circle':
            radius = box_width / 2
            points = [[center_x, center_y], [center_x + radius, center_y]]
        else:  # line 或其他形状
            points = [[x1, y1], [x2, y2]]

        shape = {
            "label": label,
            "points": points,
            "shape_type": shape_type
        }
        shapes.append(shape)

    labelme_data = {
        "version": "4.5.12",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(txt_path.replace('.txt', '.jpg')),
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width
    }

    return labelme_data
