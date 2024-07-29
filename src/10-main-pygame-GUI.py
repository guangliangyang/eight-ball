import os
import sys
import threading
import pygame
import queue
import cv2
import numpy as np
import csv
import time
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import certifi
import itertools

os.environ['SSL_CERT_FILE'] = certifi.where()

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')
warnings.filterwarnings("ignore", category=UserWarning, module='inference_feedback_manager')

from ultralytics import YOLO

# Load the YOLO model
model_file_path = os.path.join('..', 'model', 'best.pt')
model = YOLO(model_file_path)

# 增加 CSV 字段大小限制
csv.field_size_limit(2147483647)

# 全局常量
REAL_TABLE_WIDTH_M = 1.27  # 球台宽度，单位：米
REAL_TABLE_LENGTH_M = 2.54  # 球台长度，单位：米
GRID_SIZE = (16, 32)

SYS_TITLE = "Statistics of Eight Ball"
GOLDEN_RATIO = 1.618
DEBUG = True
CONFIG_FILE = 'corner_pockets_config.json'
CORNER_POCKET_YOLO_CLASS_INDEX = 0




def normalize_coordinates(coords, width, height):
    return [(int(x * width), int(y * height)) for (x, y) in coords]


def save_corners_to_file(corners, filename=CONFIG_FILE):
    with open(filename, 'w') as file:
        json.dump(corners, file)

def load_corners_from_file(filename=CONFIG_FILE):
    if not os.path.exists(filename):
        return None
    with open(filename, 'r') as file:
        return json.load(file)

def calculate_area(quad):
    # 使用四边形的四个角点计算面积
    return 0.5 * abs(
        quad[0][0]*quad[1][1] + quad[1][0]*quad[2][1] + quad[2][0]*quad[3][1] + quad[3][0]*quad[0][1] -
        (quad[1][0]*quad[0][1] + quad[2][0]*quad[1][1] + quad[3][0]*quad[2][1] + quad[0][0]*quad[3][1])
    )


def find_best_four_corners(corners):
    if len(corners) < 4:
        print("Error: At least four corner pockets are required.")
        return None
    if len(corners) == 4:
        return corners
    max_area = 0
    best_quad = None
    for quad in itertools.combinations(corners, 4):
        area = calculate_area(quad)
        if area > max_area:
            max_area = area
            best_quad = quad
    return best_quad

def draw_grid_on_pockets(frame, corners, grid_size):
    if len(corners) != 4:
        print("Error: Exactly four corner pockets are required.")
        return frame

    # 按x坐标排序找到左上和右下
    corners = sorted(corners, key=lambda k: [k[1], k[0]])  # 先按y排序，再按x排序
    top_left = corners[0]
    top_right = corners[1]
    bottom_left = corners[2]
    bottom_right = corners[3]

    # 绘制四个角点之间的矩形
    cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
    cv2.line(frame, top_left, bottom_left, (0, 255, 0), 2)
    cv2.line(frame, bottom_left, bottom_right, (0, 255, 0), 2)
    cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)

    # 计算网格间距
    grid_rows, grid_cols = grid_size
    width = top_right[0] - top_left[0]
    height = bottom_left[1] - top_left[1]
    row_step = height / grid_rows
    col_step = width / grid_cols

    # 绘制网格
    for i in range(1, grid_rows):
        start_point = (top_left[0], int(top_left[1] + i * row_step))
        end_point = (top_right[0], int(top_right[1] + i * row_step))
        cv2.line(frame, start_point, end_point, (255, 0, 0), 1)

    for j in range(1, grid_cols):
        start_point = (int(top_left[0] + j * col_step), top_left[1])
        end_point = (int(bottom_left[0] + j * col_step), bottom_left[1])
        cv2.line(frame, start_point, end_point, (255, 0, 0), 1)

    return frame


def calculate_calories_burned(met, weight_kg, duration_minutes):
    calories_burned_per_minute = (met * weight_kg * 3.5) / 200
    total_calories_burned = calories_burned_per_minute * duration_minutes
    return total_calories_burned


def calculate_calories_burned_per_hour(calories_burned, total_time_minutes):
    if total_time_minutes == 0:
        return 0, "Entertainment"

    calories_burned_per_hour = (calories_burned / total_time_minutes) * 60

    if calories_burned_per_hour < 300:
        intensity = "Entertainment"
    elif 300 <= calories_burned_per_hour <= 400:
        intensity = "Moderate"
    else:
        intensity = "Competition"

    return calories_burned_per_hour, intensity


def estimate_met(average_speed, steps_count, swings_count):
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
        "region2_3_combined": region(0, title_label_height, left_width, video_height),  # 合并后的区域
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
    colors = [(0, 0, 0), (0, 0, 1), (1, 1, 1)]  # 黑色, 蓝色, 白色
    cmap_name = 'custom_blue_white'
    n_bins = 100  # 使用100个颜色等级
    cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    norm = mcolors.Normalize(vmin=0, vmax=100)
    return norm, cmap


class EightBallGame:
    def __init__(self):
        self.templates = {"EightBall": []}
        self.recording = False
        self.keypoints_data = []
        self.video_playing = False
        self.video_length = 0
        self.current_frame = 0
        self.cap = None
        self.TEMPLATES_FILE = 'templates.csv'
        self.video_path = os.path.join('..', 'mp4', '2024-07-03 18-01-12.mp4')
        self.reset_variables()
        self.load_templates_from_csv()
        self.cap = None
        self.fps = 0
        self.delay = 0
        self.CV_CUDA_ENABLED = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.image_width = None
        self.image_height = None
        self.corners = load_corners_from_file()

    def reset_variables(self):
        self.previous_midpoint = None
        self.previous_time = None
        self.start_time = time.time()
        self.speeds = {
            'forward': [],
            'sideways': [],
            'depth': [],
            'overall': []
        }
        self.template_match_counts = {"EightBall": {}}
        self.last_matched_templates = {"EightBall": set()}

    def initialize_video_capture(self, source):
        self.cap = cv2.VideoCapture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print("FPS: {}".format(self.fps))
        self.delay = int(1000 / self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_fps(self):
        return self.fps

    def stop_video_analysis(self):
        self.video_playing = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def load_templates_from_csv(self):
        self.templates = {"EightBall": []}
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

    def process_video(self, frame):
        timers = {}
        start_time = time.time()

        match_results = {"EightBall": {}}
        if self.CV_CUDA_ENABLED:
            cv2.cuda.setDevice(1)

        timers['initial_setup'] = time.time() - start_time
        start_time = time.time()

        image = frame

        if self.image_width is None or self.image_height is None:
            self.image_width = image.shape[1]
            self.image_height = image.shape[0]

        keypoints = []
        current_speed = {
            'overall': 0
        }

        if self.recording:
            self.keypoints_data.append(keypoints)

        timers['keypoints_processing'] = time.time() - start_time
        start_time = time.time()

        # YOLO inference for eight-ball detection
        detected_objects = self.detect_eight_ball(frame, model)
        print(detected_objects)

        # Draw bounding boxes and grid on a separate canvas
        canvas = self.draw_bounding_boxes_and_grid(image, detected_objects, self.corners)

        timers['yolo_detection'] = time.time() - start_time
        start_time = time.time()

        output_image = image

        timers['process_speeds_and_highlight_ratios'] = time.time() - start_time

        if DEBUG:
            for step, duration in timers.items():
                print(f"{step}: {duration:.4f} seconds")

        return output_image, canvas

    def draw_bounding_boxes_and_grid(self, frame, detected_objects, corners):
        # 创建一个空白的与frame大小相同的画布
        canvas = np.zeros_like(frame)

        # 标注YOLO对象点
        for (x1, y1, x2, y2, coord_text, cls) in detected_objects:
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(canvas, (center_x, center_y), 5, (0, 255, 0), -1)
            cv2.putText(canvas, coord_text, (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 绘制网格
        if corners:
            canvas = draw_grid_on_pockets(canvas, corners, GRID_SIZE)

        return canvas

    def detect_eight_ball(self, frame, model):
        results = model(frame)
        detected_objects = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])

                coord_text = f'{cls}({x1}, {y1}, {x2}, {y2})'
                detected_objects.append((x1, y1, x2, y2, coord_text, cls))

        return detected_objects

    def analyze_video(self, queue):
        self.new_frame = False
        self.frame_to_show = None

        self.initialize_video_capture(self.video_path)
        self.keypoints_data = []
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.video_playing = True
        self.start_time = time.time()
        self.frame_count = 0

        while self.cap.isOpened() and self.video_playing:
            start_time = time.time()
            ret, frame = self.cap.read()
            time_read = time.time() - start_time

            if not ret:
                break

            start_time = time.time()
            image, canvas = self.process_video(frame)
            time_process_video = time.time() - start_time

            queue.put((image, canvas))

            self.frame_count += 1
            elapsed_time = (time.time() - self.start_time) * 1000
            expected_time = self.frame_count * self.delay
            wait_time = int(expected_time - elapsed_time)
            if DEBUG:
                print("int(expected_time - elapsed_time):", wait_time)
            if wait_time > 0:
                time.sleep(wait_time / 1000.0)

            if DEBUG:
                print(f"Read Frame Time: {time_read:.4f}s, Process Video Time: {time_process_video:.4f}s")

        self.video_playing = False
        self.cap.release()
        cv2.destroyAllWindows()

    def close_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class EightBallApp:

    def __init__(self, eight_ball_game):
        self.eight_ball_game = eight_ball_game
        self.eight_ball_game.app = self
        self.norm, self.cmap = get_heatmap_settings()
        self.first_data_update = True
        self.mode = "video"

        self.queue = queue.Queue(maxsize=1)

        pygame.init()
        self.screen = pygame.display.set_mode((1530, 930))
        pygame.display.set_caption(SYS_TITLE)

        self.window_width = 1530
        self.window_height = 930

        left_ratio = 1020 / 1530
        title_label_height = 60
        mode_label_height = 30
        video_height = int(self.window_width * left_ratio * 9 / 16)

        self.layout = calculate_layout(self.window_width, self.window_height, title_label_height, video_height,
                                       left_ratio, mode_label_height)

        self.data_panel_update_interval = 5.0
        self.speed_update_interval = 1.0
        self.data_panel_last_update_time = None
        self.speed_last_update_time = None

        self.setup_ui()

        self.fps = self.eight_ball_game.get_fps()
        if self.fps is None:
            self.fps = 30

        self.video_thread = threading.Thread(target=self.eight_ball_game.analyze_video, args=(self.queue,))
        self.video_thread.daemon = True
        self.video_thread.start()

    def update_statistics_table(self, stats):
        region_width = self.layout['region5']['width']
        region_height = self.layout['region5']['height']

        fig, ax = plt.subplots(figsize=(region_width / 100, region_height / 100), dpi=100)

        table_data = [
            ["", "Highest", "Average"],
            ["Cue Ball Speed (m/s)", f"{stats['cue_ball']['max_speed']:.2f}", f"{stats['cue_ball']['avg_speed']:.2f}"],
            ["Cue Ball Kinetic Energy (J)", f"{stats['cue_ball']['max_energy']:.2f}", f"{stats['cue_ball']['avg_energy']:.2f}"],
            ["Total Shots", f"{stats['player']['total_shots']}", ""],
            ["Successful Potting Rate", f"{stats['player']['potting_rate']:.2f}%", ""],
            ["Foul Counts", f"{stats['player']['foul_counts']}", ""],
            ["Highest Consecutive Potting", f"{stats['player']['max_consecutive_pots']}", ""]
        ]

        table = ax.table(cellText=table_data, cellLoc='center', loc='center')
        table.scale(1, 2)
        ax.axis('off')

        fig.canvas.draw()
        stats_table_np = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        stats_table_np = stats_table_np.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        stats_table_np = cv2.cvtColor(stats_table_np, cv2.COLOR_RGB2BGR)
        stats_table_np = np.rot90(stats_table_np, 1)
        stats_table_np = np.flipud(stats_table_np)
        stats_table_surface = pygame.surfarray.make_surface(stats_table_np)

        self.screen.blit(stats_table_surface, (self.layout['region5']['x'], self.layout['region5']['y']))
        pygame.display.update()
        plt.close(fig)

    def update_region6(self, canvas):
        skeleton_image_np = np.rot90(canvas, -1)
        skeleton_image_np = np.fliplr(skeleton_image_np)

        skeleton_surface = pygame.surfarray.make_surface(skeleton_image_np)

        skeleton_surface = pygame.transform.scale(skeleton_surface, (
            self.layout['region6']['width'], self.layout['region6']['height']))
        self.screen.blit(skeleton_surface, (self.layout['region6']['x'], self.layout['region6']['y']))

        pygame.display.update()

    def stop_video_analysis_thread(self):
        if self.video_thread is not None:
            self.eight_ball_game.stop_video_analysis()
            self.video_thread.join(timeout=5)
            self.video_thread = None

    def update_grid_count_bar_chart(self):
        pass  # Implement as needed

    def update_mode_surface(self):
        mode_text = self.mode.replace("_", " ").title()
        text = f"Mode: {mode_text} Analysis"
        self.mode_surface = self.create_label_surface(text, ("Arial", 22), "blue", "white")

        region4_x = self.layout['region4']['x']
        region4_y = self.layout['region4']['y']
        region4_width = self.layout['region4']['width']
        region4_height = self.layout['region4']['height']

        mode_surface_width = self.mode_surface.get_width()
        mode_surface_height = self.mode_surface.get_height()

        centered_x = region4_x + (region4_width - mode_surface_width) // 2
        centered_y = region4_y + (region4_height - mode_surface_height) // 2

        self.screen.fill((0, 0, 255), rect=[region4_x, region4_y, region4_width, region4_height])

        self.screen.blit(self.mode_surface, (centered_x, centered_y))
        pygame.display.update()

    def update_title_surface(self):
        title_text = f"{SYS_TITLE}"
        self.title_surface = self.create_label_surface(title_text, ("Arial", 28), "blue", "white")

        region1_x = self.layout['region1']['x']
        region1_y = self.layout['region1']['y']
        region1_width = self.layout['region1']['width']
        region1_height = self.layout['region1']['height']

        title_surface_width = self.title_surface.get_width()
        title_surface_height = self.title_surface.get_height()

        centered_x = region1_x + (region1_width - title_surface_width) // 2
        centered_y = region1_y + (region1_height - title_surface_height) // 2

        self.screen.fill((0, 0, 255), rect=[region1_x, region1_y, region1_width, region1_height])

        self.screen.blit(self.title_surface, (centered_x, centered_y))
        pygame.display.update()

    def create_label_surface(self, text, font, bg, fg):
        pygame_font = pygame.font.SysFont(font[0], font[1])
        label_surface = pygame_font.render(text, True, pygame.Color(fg), pygame.Color(bg))
        return label_surface

    def update_video_panel(self, image):
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, 1)

        frame_height, frame_width = frame.shape[:2]

        scale = min(self.layout['region2_3_combined']['width'] / frame_width,
                    self.layout['region2_3_combined']['height'] / frame_height)

        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        frame_resized = cv2.resize(frame, (new_width, new_height))

        self.video_surface.fill((0, 0, 0))

        frame_resized = np.rot90(frame_resized)
        frame_resized = pygame.surfarray.make_surface(frame_resized)

        self.video_surface.blit(frame_resized, (
            (self.layout['region2_3_combined']['width'] - new_width) // 2,
            (self.layout['region2_3_combined']['height'] - new_height) // 2))
        self.screen.blit(self.video_surface,
                         (self.layout['region2_3_combined']['x'], self.layout['region2_3_combined']['y']))

        pygame.display.update()

    def update_skeleton_surface(self, skeleton_canvas):
        skeleton_image_np = np.rot90(skeleton_canvas, -1)
        skeleton_image_np = np.fliplr(skeleton_image_np)

        skeleton_surface = pygame.surfarray.make_surface(skeleton_image_np)

        skeleton_surface = pygame.transform.scale(skeleton_surface, (
            self.layout['region6']['width'], self.layout['region6']['height']))
        self.screen.blit(skeleton_surface, (self.layout['region6']['x'], self.layout['region6']['y']))

        pygame.display.update()

    def update_data_panel(self, keypoints, match_results, speeds, swing_count, step_count):
        current_time = time.time()

        if self.data_panel_last_update_time is None:
            self.data_panel_last_update_time = current_time - 100

        if current_time - self.data_panel_last_update_time < self.data_panel_update_interval:
            return

        self.data_panel_last_update_time = current_time

        total_matches = {category: sum(self.eight_ball_game.template_match_counts[category].values()) for category in
                         self.eight_ball_game.template_match_counts}

        panel_surface = pygame.Surface((self.layout['region7']['width'], self.layout['region7']['height']))
        panel_surface.fill((255, 255, 255))
        y_offset = 10

        weight_kg = 70
        total_frames = eight_ball_game.frame_count
        fps = eight_ball_game.get_fps()
        total_time_minutes = total_frames / (fps * 60)

        average_speed = speeds['overall']['avg']
        estimated_met = estimate_met(average_speed, step_count, swing_count)
        calories_burned = calculate_calories_burned(estimated_met, weight_kg, total_time_minutes)
        calories_burned_per_hour, intensity = calculate_calories_burned_per_hour(calories_burned, total_time_minutes)

        for category, templates in self.eight_ball_game.templates.items():
            count_text = f"Swings: {swing_count}" if category == "EightBall" else f"Steps: {step_count}"

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
            match_counts = [self.eight_ball_game.template_match_counts[category].get(template['name'], 0) for template
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

            y_offset += 10

        pygame.draw.line(panel_surface, (0, 0, 0), (10, y_offset), (self.layout['region7']['width'] - 10, y_offset), 2)
        y_offset += 10

        font = pygame.font.SysFont("Arial", 22)
        text_surface = font.render(f"Calories Burned: {calories_burned:.1f} kcal", True, (0, 0, 0))
        panel_surface.blit(text_surface, (10, y_offset))
        y_offset += text_surface.get_height() + 5

        text_surface = font.render(f"Average Calories Burned per Hour: {calories_burned_per_hour:.1f} kcal", True,
                                   (0, 0, 0))
        panel_surface.blit(text_surface, (10, y_offset))
        y_offset += text_surface.get_height() + 5

        text_surface = font.render(f"Intensity: {intensity}", True, (0, 0, 0))
        panel_surface.blit(text_surface, (10, y_offset))
        y_offset += text_surface.get_height() + 5

        total_seconds = current_time - self.eight_ball_game.start_time
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        text_surface = font.render(f"Duration: {int(hours):02}:{int(minutes):02}:{int(seconds):02}", True, (0, 0, 0))
        panel_surface.blit(text_surface, (10, y_offset))
        y_offset += text_surface.get_height() + 5

        self.screen.blit(panel_surface, (self.layout['region7']['x'], self.layout['region7']['y']))
        pygame.display.update()

    def setup_ui(self):
        self.title_surface = pygame.Surface((self.layout['region1']['width'], self.layout['region1']['height']))
        self.title_surface.fill((255, 255, 255))

        # 合并后的区域
        self.video_surface = pygame.Surface(
            (self.layout['region2_3_combined']['width'], self.layout['region2_3_combined']['height']))
        self.video_surface.fill((0, 0, 0))

        self.skeleton_surface = pygame.Surface((self.layout['region6']['width'], self.layout['region6']['height']))
        self.skeleton_surface.fill((255, 255, 255))

        self.mode_surface = pygame.Surface((self.layout['region4']['width'], self.layout['region4']['height']))
        self.mode_surface.fill((255, 255, 255))

        self.speed_stats_surface = pygame.Surface((self.layout['region5']['width'], self.layout['region5']['height']))
        self.speed_stats_surface.fill((255, 255, 255))

        self.data_panel_surface = pygame.Surface((self.layout['region7']['width'], self.layout['region7']['height']))
        self.data_panel_surface.fill((255, 255, 255))

        self.update_title_surface()
        self.update_mode_label_and_reset_var()

        # Call update_statistics_table to draw the initial table
        stats = {
            'cue_ball': {
                'max_speed': 2.5,
                'avg_speed': 1.2,
                'max_energy': 1.8,
                'avg_energy': 0.9
            },
            'player': {
                'total_shots': 150,
                'potting_rate': 75.0,
                'foul_counts': 3,
                'max_consecutive_pots': 7
            }
        }
        self.update_statistics_table(stats)


    def update_speed_stats(self, speeds):
        # Define some example statistics data
        stats = {
            'cue_ball': {
                'max_speed': 2.5,
                'avg_speed': 1.2,
                'max_energy': 1.8,
                'avg_energy': 0.9
            },
            'player': {
                'total_shots': 150,
                'potting_rate': 75.0,
                'foul_counts': 3,
                'max_consecutive_pots': 7
            }
        }

        # Update statistics table
        self.update_statistics_table(stats)

    def mps_to_kph(self, speed_mps):
        return speed_mps * 3.6

    def update_mode_label_and_reset_var(self):
        self.update_mode_surface()
        self.eight_ball_game.reset_variables()

    def on_key_press(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.eight_ball_game.close_camera()
                pygame.quit()
                sys.exit()
            elif event.key == pygame.K_F5:
                self.stop_video_analysis_thread()
                if self.mode != "real_time":
                    self.eight_ball_game.close_camera()
                    self.mode = "real_time"
                    self.update_mode_label_and_reset_var()
                    self.start_real_time_analysis()
            elif event.key == pygame.K_F6:
                self.stop_video_analysis_thread()
                if any(self.eight_ball_game.templates.values()):
                    self.eight_ball_game.close_camera()
                    self.mode = "video"
                    self.update_mode_label_and_reset_var()
                    self.start_video_analysis()
            elif event.key == pygame.K_F1:
                self.detect_and_save_corners()

    def detect_and_save_corners(self):
        frame = self.queue.get_nowait()
        detected_objects = self.eight_ball_game.detect_eight_ball(frame, model)
        corners = [(x1, y1) for (x1, y1, x2, y2, txt, cls) in detected_objects if cls == CORNER_POCKET_YOLO_CLASS_INDEX]

        best_corners = find_best_four_corners(corners)
        if best_corners is None:
            return

        save_corners_to_file(best_corners)
        self.eight_ball_game.corners = best_corners
        print("Corners saved to configuration file.")


    def start_real_time_analysis(self):
        self.stop_video_analysis_thread()
        self.eight_ball_game.reset_variables()
        self.eight_ball_game.initialize_video_capture(0)
        self.video_playing = True

        self.video_thread = threading.Thread(target=self.eight_ball_game.analyze_video, args=(self.queue,))
        self.video_thread.daemon = True
        self.video_thread.start()

    def start_video_analysis(self):
        self.stop_video_analysis_thread()
        self.eight_ball_game.reset_variables()
        self.eight_ball_game.initialize_video_capture(self.eight_ball_game.video_path)
        self.video_playing = True

        self.video_thread = threading.Thread(target=self.eight_ball_game.analyze_video, args=(self.queue,))
        self.video_thread.daemon = True
        self.video_thread.start()

    def main_loop(self):
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.eight_ball_game.close_camera()
                    pygame.quit()
                    sys.exit()
                self.on_key_press(event)

            if self.queue.empty():
                time.sleep(0.1)
                pass
            else:
                image, canvas = self.queue.get()
                self.update_video_panel(image)
                self.update_region6(canvas)
                pygame.display.update()


if __name__ == "__main__":
    eight_ball_game = EightBallGame()
    app = EightBallApp(eight_ball_game)
    app.main_loop()
