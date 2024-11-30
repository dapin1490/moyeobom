from flask import Flask, render_template, Response
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# YOLO 모델 로드
model = YOLO(r'Service Design Prototyping\project\web_demo\yolo_pt\yolo11x.pt')  # YOLOv8 모델

# 웹캠 캡처
cap = cv2.VideoCapture(1)

def process_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Retrying...")
            continue  # 프레임 읽기 실패 시 다음 반복으로 넘어감

        # YOLO를 이용한 객체 탐지
        results = model(frame)
        detections = results[0]

        # 사람만 필터링 및 경계 상자 그리기
        for result in detections.boxes:
            if result.cls == 0:  # YOLO의 '0' 클래스는 사람
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 프레임 인코딩
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
