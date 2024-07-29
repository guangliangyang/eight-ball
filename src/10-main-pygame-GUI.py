import os
import sys
import threading
import pygame
import queue
import cv2
import mediapipe as mp
import numpy as np
# from PIL import Image, ImageTk
import csv
# from threading import Thread
import time

import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')
warnings.filterwarnings("ignore", category=UserWarning, module='inference_feedback_manager')


from ultralytics import YOLO


# Load the YOLO model
model_file_path = os.path.join('..', 'model', 'eight-ball.pt')
model = YOLO(model_file_path)

# 增加 CSV 字段大小限制
csv.field_size_limit(2147483647)

# 全局常量 295（桌腿到地毯），343（桌腿到窗口踢脚线），(棋盘到右侧边缘地毯)129， 76*25（三脚架中心点）
# Tl之间114 , Tc之间149 ， Tn 高度11.5
REAL_TABLE_WIDTH_M = 1.525  # 乒乓球台宽度，单位：米
REAL_TABLE_LENGTH_M = 2.74  # 乒乓球台长度，单位：米
REAL_TABLE_HEIGHT_M = 0.76 + 0.1525  # 乒乓球台台面高加网高，单位：米
REAL_TABLE_DIAGONAL_M = (REAL_TABLE_WIDTH_M ** 2 + REAL_TABLE_LENGTH_M ** 2) ** 0.5  # 乒乓球台对角线长度，单位：米
#FPS = 30  # 假设的帧率，单位：帧每秒
NOISE_THRESHOLD = 0.0006  # 噪音阈值
yolo_work = False
SYS_TITLE = "Statistics of Footwork & Arm Swings of Table Tennis"
GOLDEN_RATIO = 1.618
DEBUG = True


def calculate_calories_burned(met, weight_kg, duration_minutes):
    calories_burned_per_minute = (met * weight_kg * 3.5) / 200
    total_calories_burned = calories_burned_per_minute * duration_minutes
    return total_calories_burned


def calculate_calories_burned_per_hour(calories_burned, total_time_minutes):
    if total_time_minutes == 0:
        return 0, "Entertainment"

    calories_burned_per_hour = (calories_burned / total_time_minutes) * 60

    # 确定运动强度类别
    if calories_burned_per_hour < 300:
        intensity = "Entertainment"
    elif 300 <= calories_burned_per_hour <= 400:
        intensity = "Moderate"
    else:
        intensity = "Competition"

    return calories_burned_per_hour, intensity


def estimate_met(average_speed, steps_count, swings_count):
    # 根据平均速度、步数和挥拍次数估算MET值
    base_met = 3  # 基础活动的MET值，例如走路
    speed_factor = average_speed / 3.0  # 假设3.0 m/s 是一个高强度的运动速度
    steps_factor = steps_count / 1000  # 每1000步增加1个MET值
    swings_factor = swings_count / 100  # 每100次挥拍增加1个MET值

    estimated_met = base_met + speed_factor + steps_factor + swings_factor
    return min(estimated_met, 12)  # 限制最大MET值为12，防止过高


def calculate_layout(total_width, total_height, title_label_height, video_height, left_ratio, mode_label_height):
    left_width = int(total_width * left_ratio)
    right_width = total_width - left_width

    def region(x, y, width, height):
        return {"x": x, "y": y, "width": width, "height": height}

    regions = {
        "region1": region(0, 0, left_width, title_label_height),
        "region2": region(0, title_label_height, int(left_width / 3), video_height),
        "region3": region(int(left_width / 3), title_label_height, left_width - int(left_width / 3), video_height),
        "region4": region(0, title_label_height + video_height, left_width, mode_label_height),
        "region5": region(0, title_label_height + video_height + mode_label_height, left_width,
                          total_height - (title_label_height + video_height + mode_label_height)),
        "region6": region(left_width, 0, right_width, int(right_width * 9 / 16)),
        "region7": region(left_width, int(right_width * 9 / 16), right_width, total_height - int(right_width * 9 / 16))
    }

    if DEBUG:
        print(regions)

    return regions


def get_heatmap_settings():
    # 定义自定义颜色映射，从黑色 -> 蓝色 -> 白色
    colors = [(0, 0, 0), (0, 0, 1), (1, 1, 1)]  # 黑色, 蓝色, 白色
    cmap_name = 'custom_blue_white'
    n_bins = 100  # 使用100个颜色等级
    cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    norm = mcolors.Normalize(vmin=0, vmax=100)
    return norm, cmap


def calculate_chessboard_data(frame,
                              small_chessboard_size=(8, 8),
                              small_square_size=10.0,
                              large_square_width=100.0,
                              large_square_height=75.0,
                              vertical_offset=-15.0,
                              num_large_squares_x=3,
                              num_large_squares_y=6):


    def clamp_point(point, width, height):
        """确保点在图像范围内"""
        x, y = point
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        return x, y

    # 确保 frame 非空
    if frame is None:
        raise ValueError("输入帧为空，请检查图像路径是否正确")

    height, width, _ = frame.shape

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, small_chessboard_size, None)

    if ret:
        # 精细化角点位置
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    else:
        raise ValueError("无法检测到棋盘格角点")

    # 定义小棋盘在物理空间中的实际坐标
    objp = np.zeros((small_chessboard_size[0] * small_chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:small_chessboard_size[0], 0:small_chessboard_size[1]].T.reshape(-1, 2)
    objp *= small_square_size

    # 计算相机标定参数
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray.shape[::-1], None, None)

    # 获取棋盘左下角的点作为原点
    left_bottom_corner = corners[-1][0]  # 选择左下角的角点作为原点

    # 创建大棋盘格的物理坐标，以左下角为原点，并向下移动30cm
    chessboard_physical_points = []
    for i in range(num_large_squares_y + 1):
        for j in range(num_large_squares_x + 1):
            chessboard_physical_points.append([j * large_square_width, -i * large_square_height - vertical_offset, 0])

    chessboard_physical_points = np.array(chessboard_physical_points, dtype=np.float32)

    # 将物理坐标转换为图像坐标
    def project_points(physical_points, rvec, tvec, mtx, dist):
        image_points, _ = cv2.projectPoints(physical_points, rvec, tvec, mtx, dist)
        return image_points.reshape(-1, 2)

    chessboard_image_points_px = project_points(chessboard_physical_points, rvecs[0], tvecs[0], mtx, dist)

    # 计算每个大格子的顶点在图像中的位置
    chessboard_vertices = []
    for i in range(num_large_squares_y):
        for j in range(num_large_squares_x):
            top_left = chessboard_image_points_px[i * (num_large_squares_x + 1) + j]
            top_right = chessboard_image_points_px[i * (num_large_squares_x + 1) + j + 1]
            bottom_right = chessboard_image_points_px[(i + 1) * (num_large_squares_x + 1) + j + 1]
            bottom_left = chessboard_image_points_px[(i + 1) * (num_large_squares_x + 1) + j]
            chessboard_vertices.append([top_left, top_right, bottom_right, bottom_left])

    # 计算棋盘平面的法向量
    p1 = chessboard_physical_points[0]
    p2 = chessboard_physical_points[1]
    p3 = chessboard_physical_points[num_large_squares_x + 1]
    v1 = p2 - p1
    v2 = p3 - p1
    normal_vector = np.cross(v1, v2)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # 归一化法向量

    # 找到最右侧中间大格子的右上角顶点的物理坐标
    middle_right_square_idx = (num_large_squares_y // 2) * num_large_squares_x + (num_large_squares_x - 1)
    right_top_vertex_idx = (num_large_squares_x + 1) * (middle_right_square_idx // num_large_squares_x) + (
            middle_right_square_idx % num_large_squares_x) + 1
    right_top_vertex_phys = chessboard_physical_points[right_top_vertex_idx]

    error_ratio = 0.8 # 3d计算的错误率
    normal_vector = normal_vector * error_ratio
    # 在该顶点上绘制一条垂直于棋盘的向上直线，长度为76 cm
    vertical_end_point_phys = right_top_vertex_phys + normal_vector * 76  # 向上76 cm

    right_top_vertex_img, _ = cv2.projectPoints(right_top_vertex_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
    vertical_end_point_img, _ = cv2.projectPoints(vertical_end_point_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx,
                                                  dist)
    right_top_vertex_img = tuple(map(int, right_top_vertex_img.reshape(2)))
    vertical_end_point_img = tuple(map(int, vertical_end_point_img.reshape(2)))

    # 绘制水平线，长度为152 cm，与棋盘的Y方向平行，垂直线的终点为中点
    horizontal_start_point_phys = vertical_end_point_phys - np.array([0, 76, 0], dtype=np.float32)
    horizontal_end_point_phys = vertical_end_point_phys + np.array([0, 76, 0], dtype=np.float32)

    horizontal_start_point_img, _ = cv2.projectPoints(horizontal_start_point_phys.reshape(1, 1, 3), rvecs[0], tvecs[0],
                                                      mtx, dist)
    horizontal_end_point_img, _ = cv2.projectPoints(horizontal_end_point_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx,
                                                    dist)
    horizontal_start_point_img = tuple(map(int, horizontal_start_point_img.reshape(2)))
    horizontal_end_point_img = tuple(map(int, horizontal_end_point_img.reshape(2)))

    # 计算垂直线的3D物理坐标
    vertical_line_1_phys = horizontal_start_point_phys - np.array([0, 0, -76 * error_ratio], dtype=np.float32)
    vertical_line_2_phys = horizontal_end_point_phys - np.array([0, 0, - 76 * error_ratio], dtype=np.float32)

    vertical_line_1_end_img, _ = cv2.projectPoints(vertical_line_1_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
    vertical_line_2_end_img, _ = cv2.projectPoints(vertical_line_2_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
    vertical_line_1_end_img = tuple(map(int, vertical_line_1_end_img.reshape(2)))
    vertical_line_2_end_img = tuple(map(int, vertical_line_2_end_img.reshape(2)))

    height, width, _ = frame.shape

    # 归一化坐标
    normalized_chessboard_vertices = []
    for vertices in chessboard_vertices:
        normalized_vertices = [(pt[0] / width, pt[1] / height) for pt in vertices]
        normalized_chessboard_vertices.append(normalized_vertices)

    chessboard_data = {
        'chessboard_vertices': normalized_chessboard_vertices,
        'right_top_vertex_img': (right_top_vertex_img[0] / width, right_top_vertex_img[1] / height),
        'vertical_end_point_img': (vertical_end_point_img[0] / width, vertical_end_point_img[1] / height),
        'horizontal_start_point_img': (horizontal_start_point_img[0] / width, horizontal_start_point_img[1] / height),
        'horizontal_end_point_img': (horizontal_end_point_img[0] / width, horizontal_end_point_img[1] / height),
        'vertical_line_1_end_img': (vertical_line_1_end_img[0] / width, vertical_line_1_end_img[1] / height),
        'vertical_line_2_end_img': (vertical_line_2_end_img[0] / width, vertical_line_2_end_img[1] / height),
        'mtx': mtx,
        'dist': dist,
        'rvecs': rvecs[0],
        'tvecs': tvecs[0]
    }

    return chessboard_data


def draw_chessboard_on_frame(frame, chessboard_data, show_overlay=False, covered_area=None, highlight_ratios=None):
    if not show_overlay:
        return frame

    height, width, _ = frame.shape
    output_image = frame.copy()

    # 获取热力图设置
    norm, cmap = get_heatmap_settings()

    for idx, vertices in enumerate(chessboard_data['chessboard_vertices']):
        pts = np.array([(int(pt[0] * width), int(pt[1] * height)) for pt in vertices], dtype=np.int32)

        # 绘制热力图颜色
        vertex_tuples = tuple(map(tuple, vertices))
        if highlight_ratios and vertex_tuples in highlight_ratios:
            color = cmap(norm(highlight_ratios[vertex_tuples]))[:3]
            color = tuple(int(c * 255) for c in color)
            cv2.fillPoly(output_image, [pts], color=color)

        cv2.polylines(output_image, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

        # 绘制格子编号
        center_x = int(np.mean([pt[0] for pt in pts]))
        center_y = int(np.mean([pt[1] for pt in pts]))

        text = str(idx + 1)  # 默认的格子编号
        special_texts = {9: "R00", 6: "R01", 3: "R02", 8: "R10", 5: "R11", 2: "R12", 7: "R20", 4: "R21", 1: "R22",
                         12: "L00", 15: "L01", 18: "L02", 11: "L10", 14: "L11", 17: "L12", 10: "L20", 13: "L21",
                         16: "L22"}
        if idx + 1 in special_texts:
            text = special_texts[idx + 1]

        cv2.putText(output_image, text, (center_x, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 如果高亮占比大于0，用红色边框勾勒
        if highlight_ratios and vertex_tuples in highlight_ratios and highlight_ratios[vertex_tuples] > 0:
            cv2.polylines(output_image, [pts], isClosed=True, color=(255, 0, 0), thickness=1)

    if 'right_top_vertex_img' in chessboard_data and 'vertical_end_point_img' in chessboard_data:
        right_top_vertex_img = (int(chessboard_data['right_top_vertex_img'][0] * width),
                                int(chessboard_data['right_top_vertex_img'][1] * height))
        vertical_end_point_img = (int(chessboard_data['vertical_end_point_img'][0] * width),
                                  int(chessboard_data['vertical_end_point_img'][1] * height))
        cv2.line(output_image, right_top_vertex_img, vertical_end_point_img, (255, 255, 255), 1)

    if 'horizontal_start_point_img' in chessboard_data and 'horizontal_end_point_img' in chessboard_data:
        horizontal_start_point_img = (int(chessboard_data['horizontal_start_point_img'][0] * width),
                                      int(chessboard_data['horizontal_start_point_img'][1] * height))
        horizontal_end_point_img = (int(chessboard_data['horizontal_end_point_img'][0] * width),
                                    int(chessboard_data['horizontal_end_point_img'][1] * height))
        cv2.line(output_image, horizontal_start_point_img, horizontal_end_point_img, (255, 255, 255), 1)

    if 'vertical_line_1_end_img' in chessboard_data and 'vertical_line_2_end_img' in chessboard_data:
        vertical_line_1_end_img = (int(chessboard_data['vertical_line_1_end_img'][0] * width),
                                   int(chessboard_data['vertical_line_1_end_img'][1] * height))
        vertical_line_2_end_img = (int(chessboard_data['vertical_line_2_end_img'][0] * width),
                                   int(chessboard_data['vertical_line_2_end_img'][1] * height))
        cv2.line(output_image, horizontal_start_point_img, vertical_line_1_end_img, (255, 255, 255), 1)
        cv2.line(output_image, horizontal_end_point_img, vertical_line_2_end_img, (255, 255, 255), 1)

    return output_image


class PoseEstimation:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.templates = {"Arm": [], "Footwork": []}
        self.recording = False
        self.keypoints_data = []
        self.video_playing = False
        self.video_length = 0
        self.current_frame = 0
        self.pingpong_class = 15
        self.cap = None
        self.TEMPLATES_FILE = 'templates.csv'
        self.video_path = os.path.join('..', 'mp4', '01.mov')
        self.load_templates_from_csv()
        self.reset_variables()
        # Initialize global variables for image dimensions
        self.image_width = None
        self.image_height = None
        self.grid_rects = None
        self.show_overlay = False
        self.camera_params = None
        self.red_cross_coords = None  # 初始化 red_cross_coords
        self.load_chessboard_pattern_config()  # 初始化时加载棋盘配置
        self.covered_area = set()  # 初始化covered_area
        self.highlight_counts = {}  # 初始化highlight_counts
        self.large_square_width = 100.0  # 大格子的宽度（厘米）
        self.large_square_height = 75.0  # 大格子的高度（厘米）
        self.cap = None
        self.fps = 0
        self.delay = 0
        self.CV_CUDA_ENABLED = cv2.cuda.getCudaEnabledDeviceCount() > 0

    def reset_variables(self):
        self.previous_midpoint = None
        self.previous_foot_points = None
        self.previous_hand_points = None
        self.previous_time = None
        self.start_time = time.time()
        self.covered_area = set()  # 重置覆盖区域
        self.highlight_counts = {}  # 重置高亮次数统计
        self.speeds = {
            'forward': [],
            'sideways': [],
            'depth': [],
            'overall': []
        }  # 重置速度统计
        self.template_match_counts = {"Arm": {}, "Footwork": {}}  # 重置模板匹配计数
        self.last_matched_templates = {"Arm": set(), "Footwork": set()}  # 重置最后匹配的模板

    def initialize_video_capture(self, source):
        self.cap = cv2.VideoCapture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print("FPS: {}".format(self.fps))
        self.delay = int(1000 / self.fps)

        # 设置分辨率为16:9
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_fps(self):
        return self.fps

    def stop_video_analysis(self):
        self.video_playing = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def is_point_in_quad(self, point, quad):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        b1 = sign(point, quad[0], quad[1]) < 0.0
        b2 = sign(point, quad[1], quad[2]) < 0.0
        b3 = sign(point, quad[2], quad[3]) < 0.0
        b4 = sign(point, quad[3], quad[0]) < 0.0

        return ((b1 == b2) and (b2 == b3) and (b3 == b4))

    def calculate_covered_area(self, highlight_ratios):
        # 假设每个大格子的宽度和高度分别是100cm和75cm
        square_width_m = 1.0  # 大格子的宽度，以米为单位
        square_height_m = 0.75  # 大格子的高度，以米为单位

        highlighted_squares = sum(1 for ratio in highlight_ratios.values() if ratio > 0)

        # 计算覆盖的总面积
        covered_area = highlighted_squares * square_width_m * square_height_m

        return covered_area

    def draw_skeleton(self, image, keypoints, connections, color, circle_radius=2):
        image_height = image.shape[0]
        image_width = image.shape[1]  # 提前获取图像的宽和高

        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(keypoints) and end_idx < len(keypoints) and start_idx > 10 and end_idx > 10:
                start_point = (
                    int(keypoints[start_idx][0] * image_width), int(keypoints[start_idx][1] * image_height))
                end_point = (int(keypoints[end_idx][0] * image_width), int(keypoints[end_idx][1] * image_height))
                cv2.line(image, start_point, end_point, color, 2)
                cv2.circle(image, start_point, circle_radius, color, -1)
                cv2.circle(image, end_point, circle_radius, color, -1)

        # Draw face triangle
        face_indices = [7, 8, 10, 9]
        if all(idx < len(keypoints) for idx in face_indices):
            face_points = [(int(keypoints[idx][0] * image_width), int(keypoints[idx][1] * image_height)) for idx in
                           face_indices]
            triangle_cnt = np.array(face_points, np.int32).reshape((-1, 1, 2))
            cv2.drawContours(image, [triangle_cnt], 0, color, 2)
            cv2.fillPoly(image, [triangle_cnt], color)

        # Draw connection line
        connection_indices = [9, 10, 11, 12]
        if all(idx < len(keypoints) for idx in connection_indices):
            connection_points = [(int(keypoints[idx][0] * image_width), int(keypoints[idx][1] * image_height)) for idx
                                 in connection_indices]
            mouth_mid_point = ((connection_points[0][0] + connection_points[1][0]) // 2,
                               (connection_points[0][1] + connection_points[1][1]) // 2)
            shoulder_mid_point = ((connection_points[2][0] + connection_points[3][0]) // 2,
                                  (connection_points[2][1] + connection_points[3][1]) // 2)
            cv2.line(image, mouth_mid_point, shoulder_mid_point, color, 2)

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def compare_keypoints(self, current_keypoints, template_keypoints, category, threshold=0.9):
        Arm_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        Footwork_indices = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

        indices = Arm_indices if category == "Arm" else Footwork_indices

        for frame_keypoints in template_keypoints:
            if len(current_keypoints) != len(frame_keypoints):
                continue

            angles_current = []
            angles_template = []

            for idxs in [
                (11, 13, 15), (12, 14, 16),
                (23, 11, 13), (24, 12, 14),
                (13, 15, 17), (14, 16, 18),
                (23, 25, 27), (24, 26, 28),
                (26, 28, 32), (25, 27, 31),
                (28, 24, 27), (27, 23, 28)
            ]:
                if idxs[0] in indices and idxs[1] in indices and idxs[2] in indices:
                    angles_current.append(self.calculate_angle(current_keypoints[idxs[0]], current_keypoints[idxs[1]],
                                                               current_keypoints[idxs[2]]))
                    angles_template.append(self.calculate_angle(frame_keypoints[idxs[0]], frame_keypoints[idxs[1]],
                                                                frame_keypoints[idxs[2]]))

            similarity = np.mean([1 - abs(a - b) / 180 for a, b in zip(angles_current, angles_template)])
            if similarity >= threshold:
                return similarity

        return 0

    def load_templates_from_csv(self):
        self.templates = {"Arm": [], "Footwork": []}
        if os.path.exists(self.TEMPLATES_FILE):
            try:
                with open(self.TEMPLATES_FILE, mode='r') as file:
                    reader = csv.reader(file)
                    next(reader)
                    for row in reader:
                        name = row[0]
                        category = row[1]
                        data = eval(row[2])
                        self.templates[category].append({'name': name, 'data': data})
            except (IOError, csv.Error) as e:
                print("Error", f"Failed to load templates from CSV: {e}")

    def convert_to_physical_coordinates(self, image_point, mtx, dist, rvec, tvec):
        """
        将图像坐标转换为物理坐标 (3D)

        参数:
        - image_point: 图像坐标 (x, y)
        - mtx: 相机内参矩阵
        - dist: 相机畸变系数
        - rvec: 旋转向量
        - tvec: 平移向量

        返回:
        - 物理坐标 (X, Y, Z)
        """
        image_point = np.array([image_point], dtype=np.float32)
        undistorted_point = cv2.undistortPoints(image_point, mtx, dist, P=mtx)
        # 使用反投影来获得物理坐标
        rvec = np.array(rvec).reshape((3, 1))  # 确保 rvec 是一个 NumPy 数组并且是正确的形状
        tvec = np.array(tvec).reshape((3, 1))  # 确保 tvec 是一个 NumPy 数组并且是正确的形状
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        camera_matrix_inv = np.linalg.inv(mtx)
        uv_point = np.array([undistorted_point[0][0][0], undistorted_point[0][0][1], 1.0])
        world_point = np.dot(camera_matrix_inv, uv_point) * np.linalg.norm(tvec)
        world_point = np.dot(rotation_matrix.T, (world_point - tvec.T).T)
        return world_point.flatten()

    def convert_to_real_coordinates(self, keypoints, scaling_factor):
        real_coords = []
        for (x, y, z) in keypoints:
            X = x * scaling_factor
            Y = y * scaling_factor
            Z = z * scaling_factor
            real_coords.append((X, Y, Z))
        return real_coords

    def process_video(self, frame, pose):
        timers = {}
        start_time = time.time()

        match_results = {"Arm": {}, "Footwork": {}}
        if self.CV_CUDA_ENABLED:
            cv2.cuda.setDevice(1)

        timers['initial_setup'] = time.time() - start_time
        start_time = time.time()

        # Use GPU for OpenCV operations if available
        if self.CV_CUDA_ENABLED:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            # gpu_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
            image = gpu_frame.download()
        else:
            image = frame
            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timers['gpu_operations'] = time.time() - start_time
        start_time = time.time()

        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        timers['pose_processing'] = time.time() - start_time
        start_time = time.time()

        if self.image_width is None or self.image_height is None:
            self.image_width = image.shape[1]
            self.image_height = image.shape[0]

        timers['update_dimensions'] = time.time() - start_time
        start_time = time.time()

        # 处理棋盘格数据
        chessboard_data, output_image = self.process_chessboard(frame)

        timers['chessboard_processing'] = time.time() - start_time
        start_time = time.time()

        keypoints = []
        foot_points = []
        hand_points = []
        current_speed = {
            'forward': 0,
            'sideways': 0,
            'depth': 0,
            'overall': 0
        }

        if results.pose_landmarks:
            keypoints, foot_points, hand_points, current_speed = self.process_keypoints_and_speed(
                results.pose_landmarks.landmark)
            match_results = self.match_all_templates(keypoints, foot_points, hand_points)

            # 统计 draw_skeleton 方法的耗时
            draw_start_time = time.time()
            self.draw_skeleton(output_image, keypoints, self.mp_pose.POSE_CONNECTIONS,
                               (0, 255, 0) if any(
                                   any(match_results[category].values()) for category in match_results) else (
                                   255, 255, 255))
            timers['draw_skeleton'] = time.time() - draw_start_time

            if self.show_overlay:
                # 绘制脚踩高亮
                cell_points_hits = {tuple(map(tuple, cell_points)): 0 for cell_points in self.grid_rects}

                for foot_point in foot_points:
                    foot_x, foot_y, foot_z = foot_point[0], foot_point[1], foot_point[2]
                    print("foot_z:", foot_z)
                    if foot_z < 0 or 1 == 1:  # todo 检查z轴，确保脚在地面上
                        foot_x, foot_y = foot_point[0], foot_point[1]
                        for cell_points in self.grid_rects:
                            if self.is_point_in_quad((foot_x, foot_y), cell_points):
                                cell_points_hits[tuple(map(tuple, cell_points))] += 1
                                break

                for cell_points, hit_count in cell_points_hits.items():
                    if hit_count > 0:
                        denormalized_points = [(int(pt[0] * self.image_width), int(pt[1] * self.image_height)) for pt in
                                               cell_points]
                        pts = np.array(denormalized_points, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(output_image, [pts], isClosed=True, color=(0, 255, 255), thickness=2)  # 画四边形

            if self.recording:
                self.keypoints_data.append(keypoints)

        timers['keypoints_processing'] = time.time() - start_time
        start_time = time.time()

        # YOLO inference for ping pong table detection
        if yolo_work:
            detected_objects = self.detect_pingpong_table(frame, model)
            for (center_x, center_y, coord_text) in detected_objects:
                cv2.circle(output_image, (center_x, center_y), 5, (0, 255, 0), -1)
                cv2.putText(output_image, coord_text, (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

            # 转换坐标
            # real_coords = self.convert_to_real_coordinates(keypoints, scaling_factor)

        timers['yolo_detection'] = time.time() - start_time
        start_time = time.time()

        output_image = self.process_speeds_and_highlight_ratios(keypoints, match_results, current_speed,
                                                                chessboard_data, foot_points, output_image)

        timers['process_speeds_and_highlight_ratios'] = time.time() - start_time

        # 输出各步骤的耗时
        if DEBUG:
            for step, duration in timers.items():
                print(f"{step}: {duration:.4f} seconds")

        return output_image

    def process_speeds_and_highlight_ratios(self, keypoints, match_results, current_speed, chessboard_data, foot_points,
                                            output_image):
        swing_count = sum(self.template_match_counts["Arm"].values())
        step_count = sum(self.template_match_counts["Footwork"].values())

        speeds = {
            'forward': {
                'current': current_speed['forward'],
                'max': max(self.speeds['forward']) if self.speeds['forward'] else 0,
                'avg': np.mean(self.speeds['forward']) if self.speeds['forward'] else 0
            },
            'sideways': {
                'current': current_speed['sideways'],
                'max': max(self.speeds['sideways']) if self.speeds['sideways'] else 0,
                'avg': np.mean(self.speeds['sideways']) if self.speeds['sideways'] else 0
            },
            'depth': {
                'current': current_speed['depth'],
                'max': max(self.speeds['depth']) if self.speeds['depth'] else 0,
                'avg': np.mean(self.speeds['depth']) if self.speeds['depth'] else 0
            },
            'overall': {
                'current': current_speed['overall'],
                'max': max(self.speeds['overall']) if self.speeds['overall'] else 0,
                'avg': np.mean(self.speeds['overall']) if self.speeds['overall'] else 0
            }
        }

        height_m = self.calculate_physical_height(keypoints, self.camera_params, self.image_width, self.image_height)

        self.app.update_data_panel(keypoints, match_results, speeds, swing_count, step_count, height_m)

        skeleton_canvas = self.calculate_skeleton_image(keypoints, match_results, foot_points,
                                                        chessboard_data)
        self.app.update_skeleton_surface(skeleton_canvas)

        # output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        # output_image = Image.fromarray(output_image)

        self.app.update_speed_stats(speeds, height_m)

        # 更新 highlight ratios
        highlight_ratios = {tuple(map(tuple, vertices)): 0 for vertices in self.grid_rects}
        for cell_points_tuple in self.highlight_counts:
            if cell_points_tuple in highlight_ratios:
                highlight_ratios[cell_points_tuple] = (self.highlight_counts[cell_points_tuple] / sum(
                    self.highlight_counts.values())) * 100

        # 计算覆盖面积
        covered_area = self.calculate_covered_area(highlight_ratios)

        if self.app.current_layout == 2:
            print("todo:// change region 3-6")

        self.app.update_grid_count_bar_chart(highlight_ratios, chessboard_data, covered_area)

        return output_image

    def process_chessboard(self, frame):
        if self.grid_rects and self.red_cross_coords and self.camera_params and not self.app.calculate_chessboard:
            chessboard_data = {
                'chessboard_vertices': self.grid_rects,
                'right_top_vertex_img': self.red_cross_coords.get("right_top_vertex", (0, 0)),
                'vertical_end_point_img': self.red_cross_coords.get("vertical_end_point", (0, 0)),
                'horizontal_start_point_img': self.red_cross_coords.get("horizontal_start_point", (0, 0)),
                'horizontal_end_point_img': self.red_cross_coords.get("horizontal_end_point", (0, 0)),
                'vertical_line_1_end_img': self.red_cross_coords.get("vertical_line_1_end_img", (0, 0)),
                'vertical_line_2_end_img': self.red_cross_coords.get("vertical_line_2_end_img", (0, 0))
            }
            output_image = draw_chessboard_on_frame(frame, chessboard_data, show_overlay=self.show_overlay)
        else:
            try:
                chessboard_data = calculate_chessboard_data(frame=frame)
                output_image = draw_chessboard_on_frame(frame=frame, chessboard_data=chessboard_data,
                                                        show_overlay=self.show_overlay)
                self.grid_rects = chessboard_data['chessboard_vertices']
                self.camera_params = (
                    chessboard_data['mtx'], chessboard_data['dist'], chessboard_data['rvecs'], chessboard_data['tvecs'])
                self.red_cross_coords = {
                    "right_top_vertex": chessboard_data['right_top_vertex_img'],
                    "vertical_end_point": chessboard_data['vertical_end_point_img'],
                    "horizontal_start_point": chessboard_data['horizontal_start_point_img'],
                    "horizontal_end_point": chessboard_data['horizontal_end_point_img'],
                    "vertical_line_1_end_img": chessboard_data['vertical_line_1_end_img'],
                    "vertical_line_2_end_img": chessboard_data['vertical_line_2_end_img']
                }
                chessboard_params = {
                    "small_chessboard_size": (8, 8),
                    "small_square_size": 10.0,
                    "large_square_width": 100.0,
                    "large_square_height": 75.0,
                    "vertical_offset": -15.0,
                    "num_large_squares_x": 3,
                    "num_large_squares_y": 6
                }
                self.save_chessboard_pattern(chessboard_params, self.grid_rects, self.red_cross_coords,
                                             self.camera_params)
                self.app.calculate_chessboard = False  # 重置控制变量
            except Exception as e:
                print(f"Error in drawing large chessboard pattern: {e}")
                chessboard_data = None

        return chessboard_data, output_image

    def process_keypoints_and_speed(self, landmarks):
        keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]

        # 提取脚的关键点，包含z轴
        left_foot_points = [(landmarks[idx].x, landmarks[idx].y, landmarks[idx].z) for idx in [29, 31] if
                            idx < len(landmarks)]
        right_foot_points = [(landmarks[idx].x, landmarks[idx].y, landmarks[idx].z) for idx in [30, 32] if
                             idx < len(landmarks)]
        foot_points = left_foot_points + right_foot_points

        # 提取手的关键点
        left_hand_points = [(landmarks[idx].x, landmarks[idx].y) for idx in [17, 19] if idx < len(landmarks)]
        right_hand_points = [(landmarks[idx].x, landmarks[idx].y) for idx in [18, 20] if idx < len(landmarks)]
        hand_points = left_hand_points + right_hand_points

        current_speed = {
            'forward': 0,
            'sideways': 0,
            'depth': 0,
            'overall': 0
        }

        if self.previous_midpoint is not None:
            delta_time = 1.0 / self.fps  # 每帧的时间间隔
            current_midpoint = [(landmarks[23].x + landmarks[24].x) / 2,
                                (landmarks[23].y + landmarks[24].y) / 2]

            # Convert to physical coordinates
            current_midpoint_phys = self.convert_to_physical_coordinates(current_midpoint, *self.camera_params)
            previous_midpoint_phys = self.convert_to_physical_coordinates(self.previous_midpoint, *self.camera_params)

            delta_distance = np.linalg.norm(current_midpoint_phys - previous_midpoint_phys)
            if delta_distance < NOISE_THRESHOLD:
                delta_distance = 0  # Ignore noise
                delta_distance_x = 0
                delta_distance_y = 0
                delta_distance_z = 0
            else:
                delta_distance_x = abs(current_midpoint_phys[0] - previous_midpoint_phys[0])
                delta_distance_y = abs(current_midpoint_phys[1] - previous_midpoint_phys[1])
                delta_distance_z = abs(current_midpoint_phys[2] - previous_midpoint_phys[2])

            current_speed['overall'] = delta_distance / delta_time
            current_speed['forward'] = delta_distance_y / delta_time
            current_speed['sideways'] = delta_distance_x / delta_time
            current_speed['depth'] = delta_distance_z / delta_time

            self.speeds['overall'].append(current_speed['overall'])
            self.speeds['forward'].append(current_speed['forward'])
            self.speeds['sideways'].append(current_speed['sideways'])
            self.speeds['depth'].append(current_speed['depth'])

        self.previous_midpoint = [(landmarks[23].x + landmarks[24].x) / 2,
                                  (landmarks[23].y + landmarks[24].y) / 2]

        if hand_points:
            if self.previous_hand_points is not None:
                delta_distance = np.mean([np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in
                                          zip(hand_points, self.previous_hand_points)])
                if delta_distance < NOISE_THRESHOLD:
                    hand_points = self.previous_hand_points  # Ignore noise
            self.previous_hand_points = hand_points

        return keypoints, foot_points, hand_points, current_speed

    def detect_pingpong_table(self, frame, model):
        table_results = model.predict(frame)
        label_map = {
            0: 'dog', 1: 'person', 2: 'cat', 3: 'tv', 4: 'car', 5: 'meatballs', 6: 'marinara sauce',
            7: 'tomato soup', 8: 'chicken noodle soup', 9: 'french onion soup', 10: 'chicken breast',
            11: 'ribs', 12: 'pulled pork', 13: 'hamburger', 14: 'cavity', 15: 'tc', 16: 'tl', 17: 'tn'
        }
        detected_objects = []

        for result in table_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = label_map.get(cls, 'Unknown')

                # 绘制中心点
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                coord_text = f'{label}({center_x}, {center_y})'
                detected_objects.append((center_x, center_y, coord_text))

                # 计算缩放因子
                box_width = x2 - x1
                box_height = y2 - y1
                scaling_factor = REAL_TABLE_DIAGONAL_M / ((box_width ** 2 + box_height ** 2) ** 0.5)

        return detected_objects

    def match_all_templates(self, current_keypoints, foot_points, hand_points):
        match_results = {"Arm": {}, "Footwork": {}}
        current_matched_templates = {"Arm": set(), "Footwork": set()}
        for category, templates in self.templates.items():
            max_similarity = 0
            best_template_name = None
            for template in templates:
                template_name = template['name']
                template_keypoints = template['data']
                similarity = self.compare_keypoints(current_keypoints, template_keypoints, category)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_template_name = template_name

            if best_template_name:
                current_matched_templates[category].add(best_template_name)
                if best_template_name not in self.last_matched_templates[category]:
                    if best_template_name not in self.template_match_counts[category]:
                        self.template_match_counts[category][best_template_name] = 0
                    self.template_match_counts[category][best_template_name] += 1
                match_results[category][best_template_name] = max_similarity

        self.last_matched_templates = current_matched_templates
        return match_results

    def analyze_video(self, queue):
        self.new_frame = False
        self.frame_to_show = None

        self.initialize_video_capture(self.video_path if self.app.mode == "video" else 0)
        self.keypoints_data = []
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.app.mode == "video" else 0
        self.current_frame = 0
        self.video_playing = True
        self.start_time = time.time()
        self.frame_count = 0

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as pose:
            while self.cap.isOpened() and self.video_playing:
                start_time = time.time()
                ret, frame = self.cap.read()
                time_read = time.time() - start_time

                if not ret:
                    break

                start_time = time.time()
                image = self.process_video(frame, pose)
                time_process_video = time.time() - start_time

                # 将图像和处理结果放入队列
                queue.put(image)

                self.frame_count += 1
                elapsed_time = (time.time() - self.start_time) * 1000  # 转换为毫秒
                expected_time = self.frame_count * self.delay
                wait_time = int(expected_time - elapsed_time)
                if DEBUG:
                    print("int(expected_time - elapsed_time):", wait_time)
                # processing is too fast
                if wait_time > 0:
                    time.sleep(wait_time / 1000.0)

                if DEBUG:
                    print(f"Read Frame Time: {time_read:.4f}s, Process Video Time: {time_process_video:.4f}s")

        self.video_playing = False
        self.cap.release()
        cv2.destroyAllWindows()

    def calculate_physical_height(self, keypoints, camera_params, image_width, image_height):
        """
        计算物理身高
        :param keypoints: 关键点
        :param camera_params: 相机参数（包括内参矩阵和畸变系数）
        :return: 物理身高（米）
        """
        if len(keypoints) > 0:
            mtx, dist, rvecs, tvecs = camera_params

            left_ankle = keypoints[27]
            right_ankle = keypoints[28]
            nose = keypoints[0]

            # 反归一化
            left_ankle = (left_ankle[0] * image_width, left_ankle[1] * image_height, left_ankle[2])
            right_ankle = (right_ankle[0] * image_width, right_ankle[1] * image_height, right_ankle[2])
            nose = (nose[0] * image_width, nose[1] * image_height, nose[2])

            # 使用脚踝和头顶（鼻子）之间的距离计算身高
            ankle_img_point = [(left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2]
            nose_img_point = [nose[0], nose[1]]

            ankle_phys_point = self.convert_to_physical_coordinates(ankle_img_point, mtx, dist, rvecs, tvecs)
            nose_phys_point = self.convert_to_physical_coordinates(nose_img_point, mtx, dist, rvecs, tvecs)

            height_m = np.linalg.norm(nose_phys_point - ankle_phys_point) * 2.88 / 100  # 转换为米,2.88 是误差

            return height_m
        return 0

    def calculate_skeleton_image(self, keypoints, match_results, foot_points, chessboard_data):
        if self.app.current_layout == 1:
            screen_height = self.app.layout['region6']['height']
            screen_width = self.app.layout['region6']['width']
        else:
            screen_height = self.app.layout['region3']['height']
            screen_width = self.app.layout['region3']['width']

        skeleton_canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

        # 初始化 highlight_ratios
        highlight_ratios = {tuple(map(tuple, vertices)): 0 for vertices in chessboard_data['chessboard_vertices']}

        arm_match = any(match_results["Arm"].values())
        if arm_match:
            foot_coords = [(int(foot_point[0] * screen_width), int(foot_point[1] * screen_height)) for foot_point in
                           foot_points]
            for foot_x, foot_y in foot_coords:
                if foot_x == 0 or foot_y == 0:
                    continue
                for cell_points in chessboard_data['chessboard_vertices']:
                    normalized_points = [(int(pt[0] * screen_width), int(pt[1] * screen_height)) for pt in cell_points]
                    if self.is_point_in_quad((foot_x, foot_y), normalized_points):
                        cell_points_tuple = tuple(map(tuple, cell_points))
                        self.covered_area.add(cell_points_tuple)

                        if cell_points_tuple not in self.highlight_counts:
                            self.highlight_counts[cell_points_tuple] = 0
                        self.highlight_counts[cell_points_tuple] += 1
                        break

        total_highlights = sum(self.highlight_counts.values())
        if total_highlights > 0:
            for cell_points_tuple in highlight_ratios.keys():
                highlight_ratios[cell_points_tuple] = self.highlight_counts.get(cell_points_tuple,
                                                                                0) / total_highlights * 100

        skeleton_canvas = draw_chessboard_on_frame(skeleton_canvas, chessboard_data, show_overlay=True,
                                                   covered_area=self.covered_area, highlight_ratios=highlight_ratios)

        # 被踩过，红框
        if highlight_ratios:
            for vertices in chessboard_data['chessboard_vertices']:
                vertex_tuples = tuple(map(tuple, vertices))
                if highlight_ratios[vertex_tuples] > 0:
                    pts = np.array([(int(pt[0] * screen_width), int(pt[1] * screen_height)) for pt in vertices],
                                   dtype=np.int32)
                    cv2.polylines(skeleton_canvas, [pts], isClosed=True, color=(255, 0, 0), thickness=1)

        if arm_match:
            # 创建一个字典来记录
            cell_points_hits = {
                tuple(map(tuple, [(int(pt[0] * screen_width), int(pt[1] * screen_height)) for pt in cell_points])): 0
                for
                cell_points in chessboard_data['chessboard_vertices']}

            # 统计每个 chessboard_vertices
            for foot_x, foot_y in foot_coords:
                for cell_points in chessboard_data['chessboard_vertices']:
                    normalized_points = [(int(pt[0] * screen_width), int(pt[1] * screen_height)) for pt in cell_points]
                    if self.is_point_in_quad((foot_x, foot_y), normalized_points):
                        cell_points_hits[tuple(map(tuple, normalized_points))] += 1
                        break

            # 绘制被踩中的 chessboard_vertices
            for cell_points, hit_count in cell_points_hits.items():
                if hit_count > 0:
                    pts = np.array(cell_points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(skeleton_canvas, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

        self.draw_skeleton(skeleton_canvas, keypoints, self.mp_pose.POSE_CONNECTIONS, (255, 255, 255), 3)

        return skeleton_canvas

    def close_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def save_chessboard_pattern(self, chessboard_params, grid_rects, red_cross_coords, camera_params):
        config = {
            "chessboard_params": chessboard_params,
            "grid_rects": grid_rects,
            "red_cross_coords": red_cross_coords,
            "camera_params": {
                "mtx": camera_params[0].tolist(),
                "dist": camera_params[1].tolist(),
                "rvecs": [rvec.tolist() for rvec in camera_params[2]],
                "tvecs": [tvec.tolist() for tvec in camera_params[3]]
            }
        }
        with open("chessboard_pattern_config.json", "w") as f:
            json.dump(config, f, indent=4)

    def load_chessboard_pattern_config(self):
        if os.path.exists("chessboard_pattern_config.json"):
            try:
                with open("chessboard_pattern_config.json", "r") as f:
                    config = json.load(f)
                    self.grid_rects = config["grid_rects"]
                    self.red_cross_coords = config["red_cross_coords"]
                    camera_params = config["camera_params"]
                    self.camera_params = (
                        np.array(camera_params["mtx"]),
                        np.array(camera_params["dist"]),
                        [np.array(rvec) for rvec in camera_params["rvecs"]],
                        [np.array(tvec) for tvec in camera_params["tvecs"]]
                    )
            except Exception as e:
                print(f"Failed to load chessboard pattern config: {e}")
                self.camera_params = None
                self.grid_rects = None
                self.red_cross_coords = None


class PoseApp:

    def __init__(self, pose_estimation):
        self.pose_estimation = pose_estimation
        self.calculate_chessboard = False  # 初始化控制变量
        self.pose_estimation.app = self  # 将 PoseApp 实例赋值给 PoseEstimation 的 app 属性
        self.current_layout = 2  # 初始化布局为第一种
        self.norm, self.cmap = get_heatmap_settings()
        self.first_data_update = True
        self.temp_templates = {}
        self.mode = "video"

        self.queue = queue.Queue(maxsize=1)  # 限制队列大小为1，确保最新的帧总是可用

        # 初始化 pygame 窗口
        pygame.init()
        self.screen = pygame.display.set_mode((1530, 930))
        pygame.display.set_caption(SYS_TITLE)

        # 设置窗口宽度和高度
        self.window_width = 1530
        self.window_height = 930

        # 设置布局参数
        left_ratio = 1020 / 1530
        title_label_height = 60
        mode_label_height = 30
        video_height = int(self.window_width * left_ratio * 9 / 16)

        # 获取布局
        self.layout = calculate_layout(self.window_width, self.window_height, title_label_height, video_height,
                                       left_ratio, mode_label_height)

        self.data_panel_update_interval = 5.0  # 更新间隔为5秒
        self.speed_update_interval = 1.0  # 更新间隔为1秒
        self.data_panel_last_update_time = None
        self.speed_last_update_time = None

        self.grid_count_bar_chart_update_interval = 5.0  # 高亮条形图更新间隔为5秒
        self.grid_count_bar_chart_last_update_time = None  # 上次高亮条形图更新时间

        self.setup_ui()

        self.fps = self.pose_estimation.get_fps()
        if self.fps is None:
            self.fps = 30  # 默认帧率为30

        # 启动视频分析线程
        self.video_thread = threading.Thread(target=self.pose_estimation.analyze_video, args=(self.queue,))
        self.video_thread.daemon = True
        self.video_thread.start()

    def start_real_time_analysis(self):
        self.stop_video_analysis_thread()  # 确保停止任何正在进行的视频分析线程
        self.pose_estimation.reset_variables()  # 重置变量
        self.pose_estimation.initialize_video_capture(0)  # 使用摄像头
        self.video_playing = True

        # 启动视频分析线程
        self.video_thread = threading.Thread(target=self.pose_estimation.analyze_video, args=(self.queue,))
        self.video_thread.daemon = True
        self.video_thread.start()

    def stop_video_analysis_thread(self):
        if self.video_thread is not None:
            self.pose_estimation.stop_video_analysis()
            self.video_thread.join(timeout=5)  # 设置超时等待线程结束
            self.video_thread = None

    def update_grid_count_bar_chart(self, highlight_ratios, chessboard_data, covered_area):
        current_time = time.time()

        if self.grid_count_bar_chart_last_update_time is None:
            self.grid_count_bar_chart_last_update_time = current_time - 100

        if current_time - self.grid_count_bar_chart_last_update_time < self.grid_count_bar_chart_update_interval:
            return  # 如果距离上次更新时间不足，直接返回

        self.grid_count_bar_chart_last_update_time = current_time

        chart_width = self.layout['region2']['width']
        chart_height = self.layout['region2']['height']

        print("region2 height:", chart_height)

        # 创建一个图形对象，并设置尺寸
        fig, ax = plt.subplots(figsize=(chart_width / 100, chart_height / 100), dpi=100)  # 调整图形的尺寸，以适应在帧左侧绘制

        # 将格子编号特殊化处理，并按照指定顺序排列
        label_order = [
            "L22", "L21", "L20", "L12", "L11", "L10", "L02", "L01", "L00",
            "R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"
        ]

        label_map = {
            9: "R00", 6: "R01", 3: "R02", 8: "R10", 5: "R11", 2: "R12",
            7: "R20", 4: "R21", 1: "R22", 12: "L00", 15: "L01", 18: "L02",
            11: "L10", 14: "L11", 17: "L12", 10: "L20", 13: "L21", 16: "L22"
        }

        # 提取高亮占比数据并按照指定顺序排列
        percentages = [highlight_ratios.get(tuple(map(tuple, chessboard_data['chessboard_vertices'][idx - 1])), 0) for
                       label in label_order for idx, val in label_map.items() if val == label]

        # 创建热力图颜色映射
        norm, cmap = get_heatmap_settings()

        # 绘制水平柱状图
        bars = ax.barh(label_order, percentages, color=cmap(norm(percentages)), edgecolor='black')
        # 在每个条形图后面显示百分比数值
        for bar, percentage in zip(bars, percentages):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{percentage:.1f}%', va='center', ha='left')

        # 设置X轴为百分比
        ax.set_xlim(0, 100)
        ax.set_xlabel('')

        # 添加图表标题
        ax.set_title(f'Covered Area: {covered_area:.2f} m²', fontsize=16)

        # 添加热力刻度柱子
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_ticks([0, 20, 40, 60, 80, 100])
        cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

        # 将图表绘制到一个 numpy 数组
        fig.canvas.draw()
        chart_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        chart_img = chart_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))

        # 关闭图形对象，释放资源
        plt.close(fig)

        # 去掉alpha通道
        chart_img = chart_img[..., :3]
        # 翻转
        chart_img = np.rot90(chart_img, -1)
        chart_img = np.fliplr(chart_img)

        # 更新高亮条形图的显示
        # chart_img_resized = cv2.resize(chart_img, (self.layout['region2']['width'], self.layout['region2']['height']))

        # 将图表转换为 Pygame 表面
        chart_surface = pygame.surfarray.make_surface(chart_img)
        self.screen.blit(chart_surface, (self.layout['region2']['x'], self.layout['region2']['y']))
        pygame.display.update()

    def update_mode_surface(self):
        mode_text = self.mode.replace("_", " ").title()
        text = f"Mode: {mode_text} Analysis"
        self.mode_surface = self.create_label_surface(text, ("Arial", 22), "blue", "white")

        # 计算居中位置
        region4_x = self.layout['region4']['x']
        region4_y = self.layout['region4']['y']
        region4_width = self.layout['region4']['width']
        region4_height = self.layout['region4']['height']

        mode_surface_width = self.mode_surface.get_width()
        mode_surface_height = self.mode_surface.get_height()

        # 居中计算
        centered_x = region4_x + (region4_width - mode_surface_width) // 2
        centered_y = region4_y + (region4_height - mode_surface_height) // 2

        self.screen.fill((0, 0, 255), rect=[region4_x, region4_y, region4_width, region4_height])

        # 将 mode_surface 绘制到屏幕上
        self.screen.blit(self.mode_surface, (centered_x, centered_y))
        pygame.display.update()

    def update_title_surface(self):
        title_text = f"{SYS_TITLE}"
        self.title_surface = self.create_label_surface(title_text, ("Arial", 28), "blue", "white")

        # 计算居中位置
        region1_x = self.layout['region1']['x']
        region1_y = self.layout['region1']['y']
        region1_width = self.layout['region1']['width']
        region1_height = self.layout['region1']['height']

        title_surface_width = self.title_surface.get_width()
        title_surface_height = self.title_surface.get_height()

        # 居中计算
        centered_x = region1_x + (region1_width - title_surface_width) // 2
        centered_y = region1_y + (region1_height - title_surface_height) // 2

        # 设置 region1 背景为绿色
        self.screen.fill((0, 0, 255), rect=[region1_x, region1_y, region1_width, region1_height])

        # 将 title_surface 绘制到屏幕上
        self.screen.blit(self.title_surface, (centered_x, centered_y))
        pygame.display.update()

    def create_label_surface(self, text, font, bg, fg):
        pygame_font = pygame.font.SysFont(font[0], font[1])
        label_surface = pygame_font.render(text, True, pygame.Color(fg), pygame.Color(bg))
        return label_surface

    def update_video_panel(self, image):
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV processing
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally

        frame_height, frame_width = frame.shape[:2]

        if self.current_layout == 1:
            # 计算右2/3部分的起始列, 省出1/3放区域命中统计图
            end_col = (frame_width * 2) // 3
            frame = frame[:, :end_col]
            frame_height, frame_width = frame.shape[:2]

        if self.current_layout == 1:
            scale = min(self.layout['region3']['width'] / frame_width, self.layout['region3']['height'] / frame_height)
        else:
            scale = min(self.layout['region6']['width'] / frame_width, self.layout['region6']['height'] / frame_height)

        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        frame_resized = cv2.resize(frame, (new_width, new_height))

        self.video_surface.fill((0, 0, 0))

        frame_resized = np.rot90(frame_resized)
        frame_resized = pygame.surfarray.make_surface(frame_resized)

        if self.current_layout == 1:
            self.video_surface.blit(frame_resized, (
                (self.layout['region3']['width'] - new_width) // 2,
                (self.layout['region3']['height'] - new_height) // 2))
            self.screen.blit(self.video_surface, (self.layout['region3']['x'], self.layout['region3']['y']))
        else:
            self.video_surface.blit(frame_resized, (
                (self.layout['region6']['width'] - new_width) // 2,
                (self.layout['region6']['height'] - new_height) // 2))
            self.screen.blit(self.video_surface, (self.layout['region6']['x'], self.layout['region6']['y']))

        pygame.display.update()

    def update_skeleton_surface(self, skeleton_canvas):
        # 颜色转换和图像旋转/翻转合并
        skeleton_image_np = np.rot90(skeleton_canvas, -1)
        skeleton_image_np = np.fliplr(skeleton_image_np)

        # 使用pygame显示图像，并在此处调整大小
        skeleton_surface = pygame.surfarray.make_surface(skeleton_image_np)

        if self.current_layout == 1:
            skeleton_surface = pygame.transform.scale(skeleton_surface, (
                self.layout['region6']['width'], self.layout['region6']['height']))  # 调整显示大小
            self.screen.blit(skeleton_surface, (self.layout['region6']['x'], self.layout['region6']['y']))
        else:
            # 计算右2/3部分的起始列
            width = skeleton_surface.get_width()
            height = skeleton_surface.get_height()
            start_col = width // 3

            # 裁剪右2/3部分
            cropped_surface = skeleton_surface.subsurface((start_col, 0, width - start_col, height))

            # 调整大小以适应目标区域
            cropped_surface = pygame.transform.scale(cropped_surface, (
                self.layout['region3']['width'], self.layout['region3']['height']))

            self.screen.blit(cropped_surface, (self.layout['region3']['x'], self.layout['region3']['y']))

        pygame.display.update()

    def update_data_panel(self, keypoints, match_results, speeds, swing_count, step_count, height_m):
        current_time = time.time()

        if self.data_panel_last_update_time is None:
            self.data_panel_last_update_time = current_time - 100

        if current_time - self.data_panel_last_update_time < self.data_panel_update_interval:
            return  # 如果距离上次更新时间不足，直接返回

        self.data_panel_last_update_time = current_time

        total_matches = {category: sum(self.pose_estimation.template_match_counts[category].values()) for category in
                         self.pose_estimation.template_match_counts}

        panel_surface = pygame.Surface((self.layout['region7']['width'], self.layout['region7']['height']))
        panel_surface.fill((255, 255, 255))
        y_offset = 10

        # 计算卡路里消耗部分
        # 假设用户体重为70kg
        weight_kg = 70
        total_frames = pose_estimation.frame_count
        fps = pose_estimation.get_fps()
        total_time_minutes = total_frames / (fps * 60)

        # 估算MET值
        average_speed = speeds['overall']['avg']
        estimated_met = estimate_met(average_speed, step_count, swing_count)
        calories_burned = calculate_calories_burned(estimated_met, weight_kg, total_time_minutes)  # 计算平均每小时消耗的卡路里和运动强度
        calories_burned_per_hour, intensity = calculate_calories_burned_per_hour(calories_burned, total_time_minutes)

        for category, templates in self.pose_estimation.templates.items():
            count_text = f"Arm Swings: {swing_count}" if category == "Arm" else f"Steps: {step_count}"

            font = pygame.font.SysFont("Arial", 22)
            text_surface = font.render(count_text, True, (0, 0, 0))
            text_width, text_height = text_surface.get_size()
            bg_rect = pygame.Rect(10, y_offset, panel_surface.get_width() - 20, text_height + 10)
            pygame.draw.rect(panel_surface, (211, 211, 211), bg_rect)

            text_x = bg_rect.x + (bg_rect.width - text_width) // 2
            text_y = bg_rect.y + 5
            panel_surface.blit(text_surface, (text_x, text_y))
            y_offset += bg_rect.height + 5

            template_names = [template['name'] for template in templates]
            match_counts = [self.pose_estimation.template_match_counts[category].get(template['name'], 0) for template
                            in templates]
            match_percentages = [(count / total_matches[category] * 100) if total_matches[category] > 0 else 0 for
                                 count in match_counts]

            for template_name, percentage in zip(template_names, match_percentages):
                font = pygame.font.SysFont("Arial", 20)
                text_surface = font.render(f"{template_name}: {percentage:.1f}%", True, (0, 0, 0))
                panel_surface.blit(text_surface, (10, y_offset))
                y_offset += text_surface.get_height() + 5

                bar_x = 10
                bar_y = y_offset
                bar_width = self.layout['region7']['width'] - 20
                bar_height = 20
                pygame.draw.rect(panel_surface, (211, 211, 211), (bar_x, bar_y, bar_width, bar_height))

                norm, cmap = get_heatmap_settings()
                color = cmap(norm(percentage))[:3]
                color = tuple(int(c * 255) for c in color)
                fill_width = int(bar_width * (percentage / 100))
                pygame.draw.rect(panel_surface, color, (bar_x, bar_y, fill_width, bar_height))
                y_offset += bar_height + 5

            y_offset += 10  # 每个类别之间增加20像素的间隔

        # 绘制分割线
        pygame.draw.line(panel_surface, (0, 0, 0), (10, y_offset), (self.layout['region7']['width'] - 10, y_offset), 2)
        y_offset += 10  # 分割线和卡路里信息之间的间隔

        # 显示卡路里消耗
        font = pygame.font.SysFont("Arial", 22)
        text_surface = font.render(f"Calories Burned: {calories_burned:.1f} kcal", True, (0, 0, 0))
        panel_surface.blit(text_surface, (10, y_offset))
        y_offset += text_surface.get_height() + 5

        # 显示卡路里消耗
        font = pygame.font.SysFont("Arial", 22)
        text_surface = font.render(f"Average Calories Burned per Hour: {calories_burned_per_hour:.1f} kcal", True,
                                   (0, 0, 0))
        panel_surface.blit(text_surface, (10, y_offset))
        y_offset += text_surface.get_height() + 5

        # 显示运动强度
        text_surface = font.render(f"Intensity: {intensity}", True, (0, 0, 0))
        panel_surface.blit(text_surface, (10, y_offset))
        y_offset += text_surface.get_height() + 5

        # 显示运动时间长度
        total_seconds = current_time - self.pose_estimation.start_time
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        text_surface = font.render(f"Duration: {int(hours):02}:{int(minutes):02}:{int(seconds):02}", True, (0, 0, 0))
        panel_surface.blit(text_surface, (10, y_offset))
        y_offset += text_surface.get_height() + 5

        self.screen.blit(panel_surface, (self.layout['region7']['x'], self.layout['region7']['y']))
        pygame.display.update()

    def setup_ui(self):

        # 创建各个区域的显示表面
        self.title_surface = pygame.Surface((self.layout['region1']['width'], self.layout['region1']['height']))
        self.title_surface.fill((255, 255, 255))

        self.grid_count_bar_chart_ske = pygame.Surface(
            (self.layout['region2']['width'], self.layout['region2']['height']))
        self.grid_count_bar_chart_ske.fill((255, 255, 255))

        if self.current_layout == 1:
            self.video_surface = pygame.Surface((self.layout['region3']['width'], self.layout['region3']['height']))
            self.video_surface.fill((0, 0, 0))
            self.skeleton_surface = pygame.Surface((self.layout['region6']['width'], self.layout['region6']['height']))
            self.skeleton_surface.fill((255, 255, 255))
        else:
            self.video_surface = pygame.Surface((self.layout['region6']['width'], self.layout['region6']['height']))
            self.video_surface.fill((0, 0, 0))
            self.skeleton_surface = pygame.Surface((self.layout['region3']['width'], self.layout['region3']['height']))
            self.skeleton_surface.fill((255, 255, 255))

        self.mode_surface = pygame.Surface((self.layout['region4']['width'], self.layout['region4']['height']))
        self.mode_surface.fill((255, 255, 255))

        self.speed_stats_surface = pygame.Surface((self.layout['region5']['width'], self.layout['region5']['height']))
        self.speed_stats_surface.fill((255, 255, 255))

        self.data_panel_surface = pygame.Surface((self.layout['region7']['width'], self.layout['region7']['height']))
        self.data_panel_surface.fill((255, 255, 255))

        self.update_title_surface()
        self.update_mode_label_and_reset_var()

    def update_speed_stats(self, speeds, height_m):
        current_time = time.time()

        if self.speed_last_update_time == None:
            self.speed_last_update_time = current_time - 100

        if current_time - self.speed_last_update_time < self.speed_update_interval:
            return  # 如果距离上次更新时间不足1秒，直接返回

        self.speed_last_update_time = current_time

        # 绘制速度统计到 Pygame surface 上

        # 获取区域宽高
        region_width = self.layout['region5']['width']
        region_height = self.layout['region5']['height']

        # 绘制速度统计到 Pygame surface 上
        fig, ax = plt.subplots(figsize=(region_width / 100, region_height / 100), dpi=100)

        table_data = [
            ["", "Current (km/h)", "Max (km/h)", "Average (km/h)"],
            ["Horizontal Speed", f"{self.mps_to_kph(speeds['forward']['current']):.2f}",
             f"{self.mps_to_kph(speeds['forward']['max']):.2f}", f"{self.mps_to_kph(speeds['forward']['avg']):.2f}"],
            ["Vertical Speed", f"{self.mps_to_kph(speeds['sideways']['current']):.2f}",
             f"{self.mps_to_kph(speeds['sideways']['max']):.2f}", f"{self.mps_to_kph(speeds['sideways']['avg']):.2f}"],
            ["Depth Speed", f"{self.mps_to_kph(speeds['depth']['current']):.2f}",
             f"{self.mps_to_kph(speeds['depth']['max']):.2f}", f"{self.mps_to_kph(speeds['depth']['avg']):.2f}"],
            ["Overall Speed", f"{self.mps_to_kph(speeds['overall']['current']):.2f}",
             f"{self.mps_to_kph(speeds['overall']['max']):.2f}", f"{self.mps_to_kph(speeds['overall']['avg']):.2f}"]
        ]

        table = ax.table(cellText=table_data, cellLoc='center', loc='center')
        table.scale(1, 2)
        ax.axis('off')

        fig.canvas.draw()
        speed_stats_np = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        speed_stats_np = speed_stats_np.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        speed_stats_np = cv2.cvtColor(speed_stats_np, cv2.COLOR_RGB2BGR)
        speed_stats_np = np.rot90(speed_stats_np, 1)
        speed_stats_np = np.flipud(speed_stats_np)
        speed_stats_surface = pygame.surfarray.make_surface(speed_stats_np)

        self.screen.blit(speed_stats_surface, (self.layout['region5']['x'], self.layout['region5']['y']))
        pygame.display.update()
        plt.close(fig)

    def mps_to_kph(self, speed_mps):
        return speed_mps * 3.6

    def update_mode_label_and_reset_var(self):
        self.update_mode_surface()
        self.pose_estimation.reset_variables()  # 调用重置方法

    def on_key_press(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.pose_estimation.close_camera()
                pygame.quit()
                sys.exit()
            elif event.key == pygame.K_F5:
                self.stop_video_analysis_thread()  # 确保停止当前视频分析线程
                if self.mode != "real_time":
                    self.pose_estimation.close_camera()
                    self.mode = "real_time"
                    self.update_mode_label_and_reset_var()
                    self.start_real_time_analysis()  # 正确调用 start_real_time_analysis 方法
            elif event.key == pygame.K_F6:
                self.stop_video_analysis_thread()  # 确保停止当前视频分析线程
                if any(self.pose_estimation.templates.values()):
                    self.pose_estimation.close_camera()
                    self.mode = "video"
                    self.update_mode_label_and_reset_var()
                    self.start_video_analysis()  # 启动视频分析
            elif event.key == pygame.K_F1:
                chessboard_params = {
                    "small_chessboard_size": (8, 8),
                    "small_square_size": 10.0,
                    "large_square_width": 100.0,
                    "large_square_height": 75.0,
                    "vertical_offset": -15.0,
                    "num_large_squares_x": 3,
                    "num_large_squares_y": 6
                }
                self.pose_estimation.save_chessboard_pattern(chessboard_params, self.pose_estimation.grid_rects,
                                                             self.pose_estimation.red_cross_coords,
                                                             self.pose_estimation.camera_params)
            elif event.key == pygame.K_F2:
                self.calculate_chessboard = True
            elif event.key == pygame.K_F3:
                self.pose_estimation.show_overlay = not self.pose_estimation.show_overlay
            elif event.key == pygame.K_F4:
                self.current_layout = 2 if self.current_layout == 1 else 1
                self.setup_ui()

    def start_video_analysis(self):
        self.stop_video_analysis_thread()  # 确保停止任何正在进行的视频分析线程
        self.pose_estimation.reset_variables()  # 重置变量
        self.pose_estimation.initialize_video_capture(self.pose_estimation.video_path)  # 使用视频文件
        self.video_playing = True

        # 启动视频分析线程
        self.video_thread = threading.Thread(target=self.pose_estimation.analyze_video, args=(self.queue,))
        self.video_thread.daemon = True
        self.video_thread.start()

    def main_loop(self):
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.pose_estimation.close_camera()
                    pygame.quit()
                    sys.exit()
                self.on_key_press(event)

            # 从队列中读取图像并更新视频面板
            if self.queue.empty():
                time.sleep(0.1)
                pass
            else:
                image = self.queue.get()
                self.update_video_panel(image)
                pygame.display.update()

            # clock.tick(5)  # 控制帧率


if __name__ == "__main__":
    pose_estimation = PoseEstimation()
    app = PoseApp(pose_estimation)
    app.main_loop()
