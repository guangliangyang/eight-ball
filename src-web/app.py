from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import os
import threading
import time
import uuid
import cv2
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load the YOLO model
model_file_path = os.path.join('model', 'best.pt')
model = YOLO(model_file_path)

# In-memory storage for progress tracking
progress_store = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        unique_id = str(uuid.uuid4())
        filename = unique_id + "_" + file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        progress_store[unique_id] = {
            'progress': 0,
            'elapsed_time': '0s',
            'estimated_time_remaining': 'Unknown',
            'status': 'processing'
        }

        thread = threading.Thread(target=process_video, args=(unique_id, filepath))
        thread.start()

        return jsonify({'status': 'processing', 'unique_id': unique_id, 'filename': filename})
    return jsonify({'status': 'failed'})


@app.route('/progress_and_results/<unique_id>', methods=['GET'])
def progress_and_results(unique_id):
    if unique_id in progress_store:
        return jsonify(progress_store[unique_id])
    return jsonify({'status': 'not found'}), 404


def process_video(unique_id, filepath):
    start_time = time.time()
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    # Dummy processing loop, replace with actual YOLO processing
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Example YOLO detection
        results = model(frame)

        # Update progress
        processed_frames += 1
        progress = (processed_frames / total_frames) * 100
        elapsed_time = time.time() - start_time
        estimated_time_remaining = (elapsed_time / processed_frames) * (total_frames - processed_frames)

        progress_store[unique_id]['progress'] = progress
        progress_store[unique_id]['elapsed_time'] = f"{elapsed_time:.2f}s"
        progress_store[unique_id]['estimated_time_remaining'] = f"{estimated_time_remaining:.2f}s"

    cap.release()
    progress_store[unique_id]['status'] = 'completed'
    progress_store[unique_id]['results'] = {
        'total_shots': 150,  # Example results
        'potting_rate': 75.0,
        'foul_counts': 3,
        'max_consecutive_pots': 7
    }


if __name__ == '__main__':
    app.run(debug=True)
