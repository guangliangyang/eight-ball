import cv2
import os


def extract_frames(video_path, output_dir, frame_interval=15):
    """
    从视频中每隔指定帧数提取图像帧。

    :param video_path: 视频文件的路径。
    :param output_dir: 存储图像帧的输出目录。
    :param frame_interval: 提取帧的间隔（默认为15，即每15帧提取一次）。
    """
    video_name = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"{video_name}_frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {extracted_count} frames from {video_path}")


def process_directory(input_dir, output_dir, frame_interval=15):
    """
    处理目录中的所有MP4视频文件。

    :param input_dir: 包含视频文件的输入目录。
    :param output_dir: 存储提取的图像帧的输出目录。
    :param frame_interval: 提取帧的间隔（默认为15，即每15帧提取一次）。
    """
    print(f"Current working directory: {os.getcwd()}")
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist")
        return

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_dir, filename)
            extract_frames(video_path, output_dir, frame_interval)


# 设置输入目录和输出目录
input_directory = os.path.join('..', 'mp4')
output_directory = os.path.join('..', 'pic-extracted')

# 提取视频中的图像帧，每隔15帧提取一次
process_directory(input_directory, output_directory, frame_interval=15)
