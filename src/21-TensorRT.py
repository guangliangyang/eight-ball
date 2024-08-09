import torch
from torch2trt import torch2trt
import cv2


class EightBallGame:
    def __init__(self):
        self.cap = None
        self.model_trt = None
        self.video_path = os.path.join('..', 'mp4', '2024-07-03 18-01-12.mp4')
        self.initialize_video_capture(self.video_path)
        self.load_trt_model()

    def initialize_video_capture(self, source):
        self.cap = cv2.VideoCapture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video FPS: {self.fps}, Frame Size: {self.frame_width}x{self.frame_height}")

    def load_trt_model(self):
        # Generate example input based on the video frame size
        example_input = torch.randn(1, 3, self.frame_height, self.frame_width).cuda()

        # Load the PyTorch YOLO model
        model_file_path = os.path.join('..', 'model', 'best.pt')
        model = YOLO(model_file_path)

        # Convert the model to TensorRT format
        self.model_trt = torch2trt(model.model, [example_input], fp16_mode=True)

        # Optionally, save the converted model
        torch.save(self.model_trt.state_dict(), 'best_trt.pth')

    def detect_eight_ball(self, frame, nms_threshold=0.4):
        # Convert the OpenCV image to a tensor
        img = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().cuda()

        # Perform inference using TensorRT
        with torch.no_grad():
            results = self.model_trt(img)

        detected_objects = []
        boxes = []
        confidences = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                score = float(box.conf[0])  # Ensure the score is a float
                cls = int(box.cls[0])
                boxes.append([x1, y1, x2 - x1, y2 - y1])  # Convert to (x, y, width, height) format
                confidences.append(score)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.05, nms_threshold=nms_threshold)

        if len(indices) > 0:
            if isinstance(indices[0], list):
                indices = [i[0] for i in indices]
            for i in indices:
                x, y, w, h = boxes[i]
                detected_objects.append((x, y, x + w, y + h, confidences[i], cls))

        return detected_objects

