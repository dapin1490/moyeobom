from django.shortcuts import render

# Create your views here.
import torch
import cv2
import numpy as np
from django.http import StreamingHttpResponse, JsonResponse
from ultralytics import YOLO
from norfair import Detection, Tracker

# 이전 위치 저장
previous_positions = {}
complex_ratio = [30, 70]
message_count = ""
message_ratio = ""

# YOLO 모델 로드
model = YOLO("yolo11x.pt")

# Norfair 추적기 초기화
tracker = Tracker(distance_function="euclidean", distance_threshold=120)

# 웹캠 캡처
cap = cv2.VideoCapture(0)

def calculate_direction(prev_pos, curr_pos):
    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]
    if abs(dx) > abs(dy):  # 수평 이동
        return "right" if dx > 0 else "left"
    else:  # 수직 이동
        return "down" if dy > 0 else "up"

def process_frames():
    global previous_positions, message_count
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = []
        people_count = 0

        for result in results[0].boxes:
            if result.cls == 0:  # 사람 클래스
                people_count += 1
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                detections.append(Detection(np.array([center_x, center_y])))

        tracked_objects = tracker.update(detections)
        current_direction_counts = {"left": 0, "right": 0, "up": 0, "down": 0}

        for obj in tracked_objects:
            obj_id = obj.id
            position = obj.estimate
            x, y = map(int, position[0])
            if obj_id in previous_positions:
                prev_x, prev_y = previous_positions[obj_id]
                direction = calculate_direction((prev_x, prev_y), (x, y))
                current_direction_counts[direction] += 1
            previous_positions[obj_id] = (x, y)

        most_movement_direction = max(current_direction_counts, key=current_direction_counts.get)
        most_movement_count = current_direction_counts[most_movement_direction]
        message_count = f"People: {people_count}\nMost movement: {most_movement_direction} ({most_movement_count})"

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

def process_area_frames():
    global complex_ratio, message_ratio
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_area = frame.shape[0] * frame.shape[1]
        total_person_area = 0
        results = model(frame)

        for result in results[0].boxes:
            if result.cls == 0:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                person_area = (x2 - x1) * (y2 - y1)
                total_person_area += person_area

        area_ratio = (total_person_area / frame_area) * 100
        message_ratio = "Person Area Ratio: "
        if area_ratio < complex_ratio[0]:
            message_ratio += "여유"
        elif complex_ratio[0] <= area_ratio < complex_ratio[1]:
            message_ratio += "보통"
        else:
            message_ratio += "혼잡"

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

def index(request):
    return render(request, "index.html")

def count_view(request):
    return render(request, "count_view.html")

def area_view(request):
    return render(request, "area_view.html")

def video_feed(request):
    return StreamingHttpResponse(process_frames(), content_type="multipart/x-mixed-replace; boundary=frame")

def area_feed(request):
    return StreamingHttpResponse(process_area_frames(), content_type="multipart/x-mixed-replace; boundary=frame")

def get_count_data(request):
    global message_count
    return JsonResponse({"message": message_count})

def get_ratio_data(request):
    global message_ratio
    return JsonResponse({"message": message_ratio})
