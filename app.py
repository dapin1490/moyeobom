import torch
import cv2
import numpy as np
from flask import Flask, render_template, Response
from ultralytics import YOLO
from norfair import Detection, Tracker


# 이동 방향 통계 변수 초기화
direction_counts = {"left": 0, "right": 0, "up": 0, "down": 0}
previous_positions = {}

def calculate_direction(prev_pos, curr_pos):
    """두 점을 비교하여 이동 방향을 계산합니다."""
    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]
    if abs(dx) > abs(dy):  # 수평 이동
        return "right" if dx > 0 else "left"
    else:  # 수직 이동
        return "down" if dy > 0 else "up"

def process_frames():
    global direction_counts, previous_positions

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from webcam.")
            break

        # YOLO 탐지
        results = model(frame)
        detections = []

        # 탐지 결과에서 사람만 필터링
        for result in results[0].boxes:
            if result.cls == 0:  # 사람 클래스 (YOLO 기준 0)
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                detections.append(Detection(np.array([center_x, center_y])))

        # 추적 업데이트
        tracked_objects = tracker.update(detections)

        # 추적 결과 시각화 및 방향 계산
        for obj in tracked_objects:
            obj_id = obj.id
            position = obj.estimate
            x, y = map(int, position[0])

            # 이전 위치와 비교하여 이동 방향 계산
            if obj_id in previous_positions:
                prev_x, prev_y = previous_positions[obj_id]
                direction = calculate_direction((prev_x, prev_y), (x, y))
                direction_counts[direction] += 1
            previous_positions[obj_id] = (x, y)

            # 객체 ID 및 현재 위치 표시
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"ID: {obj_id}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 방향 통계 표시
        directions_text = " | ".join([f"{key}: {value}" for key, value in direction_counts.items()])
        cv2.putText(frame, directions_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 프레임을 JPEG로 인코딩
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


app = Flask(__name__)

# YOLO 모델 로드
model = YOLO("yolo11x.pt")

# Norfair 추적기 초기화
tracker = Tracker(distance_function="euclidean", distance_threshold=60)

# 웹캠 캡처
cap = cv2.VideoCapture(0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(process_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
