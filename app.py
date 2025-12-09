import torch
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, send_from_directory
from ultralytics import YOLO
from norfair import Detection, Tracker

# ============================================================================
# 전역 변수
# ============================================================================
previous_positions = {}  # 이전 위치 저장
complex_ratio = [30, 70]  # 혼잡도 기준 비율
message_count = ""
message_ratio = ""
ratio_code = ''

# ============================================================================
# 유틸리티 함수
# ============================================================================
def calculate_direction(prev_pos, curr_pos):
    """두 점을 비교하여 이동 방향을 계산합니다."""
    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]
    if abs(dx) > abs(dy):  # 수평 이동
        return "right" if dx > 0 else "left"
    else:  # 수직 이동
        return "down" if dy > 0 else "up"


# ============================================================================
# 프레임 처리 함수
# ============================================================================
def process_frames():
    global previous_positions, message_count

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from webcam.")
            break

        # YOLO 탐지
        results = model(frame)
        detections = []

        # 사람 수
        people_count = 0

        # 탐지 결과에서 사람만 필터링
        for result in results[0].boxes:
            if result.cls == 0:  # 사람 클래스 (YOLO 기준 0)
                people_count += 1
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                detections.append(Detection(np.array([center_x, center_y])))

        # 추적 업데이트
        tracked_objects = tracker.update(detections)

        # 현재 프레임의 방향 통계
        current_direction_counts = {"left": 0, "right": 0, "up": 0, "down": 0}

        # 추적 결과 시각화 및 방향 계산
        for obj in tracked_objects:
            obj_id = obj.id
            position = obj.estimate
            x, y = map(int, position[0])

            # 이전 위치와 비교하여 이동 방향 계산
            if obj_id in previous_positions:
                prev_x, prev_y = previous_positions[obj_id]
                direction = calculate_direction((prev_x, prev_y), (x, y))
                current_direction_counts[direction] += 1
            previous_positions[obj_id] = (x, y)

            # 객체 ID 및 현재 위치 표시
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"ID: {obj_id}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 현재 프레임에서 가장 많은 이동 방향 찾기
        most_movement_direction = max(current_direction_counts, key=current_direction_counts.get)
        most_movement_count = current_direction_counts[most_movement_direction]

        # 방향 결과 표시
        message_count = f"People: {people_count}\nMost movement: {most_movement_direction} ({most_movement_count})"

        # 프레임을 JPEG로 인코딩
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


def process_area_frames():
    global complex_ratio, message_ratio
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from webcam.")
            break

        frame_area = frame.shape[0] * frame.shape[1]
        total_person_area = 0

        # YOLO 탐지
        results = model(frame)

        for result in results[0].boxes:
            if result.cls == 0:  # 사람 클래스
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                person_area = (x2 - x1) * (y2 - y1)
                total_person_area += person_area

        # 면적 비율 계산
        area_ratio = (total_person_area / frame_area) * 100
        if area_ratio < complex_ratio[0]:
            message_ratio = "Person Area Ratio: 여유"
        elif complex_ratio[0] <= area_ratio < complex_ratio[1]:
            message_ratio = "Person Area Ratio: 보통"
        else:
            message_ratio = "Person Area Ratio: 혼잡"

        # 프레임을 JPEG로 인코딩
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


def process_cam_feed():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from webcam.")
            break

        # 프레임을 JPEG로 인코딩
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


def detect_video(video_path):
    global previous_positions, message_count, complex_ratio, message_ratio

    while True:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:  # 동영상 끝
                break

            # YOLO 탐지 수행
            results = model(frame)
            detections = []
            # 사람 수
            people_count = 0
            frame_area = frame.shape[0] * frame.shape[1]
            total_person_area = 0

            for result in results[0].boxes:
                if result.cls == 0:  # 사람 클래스
                    people_count += 1
                    x1, y1, x2, y2 = map(int, result.xyxy[0])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    detections.append(Detection(np.array([center_x, center_y])))
                    person_area = (x2 - x1) * (y2 - y1)
                    total_person_area += person_area

            # 추적 업데이트
            tracked_objects = tracker.update(detections)

            # 현재 프레임의 방향 통계
            current_direction_counts = {"left": 0, "right": 0, "up": 0, "down": 0}

            # 추적 결과 시각화 및 방향 계산
            for obj in tracked_objects:
                obj_id = obj.id
                position = obj.estimate
                x, y = map(int, position[0])

                # 이전 위치와 비교하여 이동 방향 계산
                if obj_id in previous_positions:
                    prev_x, prev_y = previous_positions[obj_id]
                    direction = calculate_direction((prev_x, prev_y), (x, y))
                    current_direction_counts[direction] += 1
                previous_positions[obj_id] = (x, y)

                # 객체 ID 및 현재 위치 표시
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"ID: {obj_id}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 현재 프레임에서 가장 많은 이동 방향 찾기
            most_movement_direction = max(current_direction_counts, key=current_direction_counts.get)
            most_movement_count = current_direction_counts[most_movement_direction]

            # 방향 결과 표시
            message_count = f"People: {people_count}\nMost movement: {most_movement_direction} ({most_movement_count})"

            # 면적 비율 계산
            area_ratio = (total_person_area / frame_area) * 100
            if area_ratio < complex_ratio[0]:
                message_ratio = "Person Area Ratio: 여유"
            elif complex_ratio[0] <= area_ratio < complex_ratio[1]:
                message_ratio = "Person Area Ratio: 보통"
            else:
                message_ratio = "Person Area Ratio: 혼잡"

            # 프레임을 JPEG로 인코딩
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()


# ============================================================================
# Flask 앱 초기화
# ============================================================================
app = Flask(__name__)

# YOLO 모델 로드
model = YOLO("yolo11x.pt")

# Norfair 추적기 초기화
tracker = Tracker(distance_function="euclidean", distance_threshold=120)

# 웹캠 캡처
cap = cv2.VideoCapture(0)


# ============================================================================
# Flask 라우트
# ============================================================================

# 템플릿 라우트
@app.route('/static/fonts/<path:filename>')
def custom_font(filename):
    return send_from_directory('static/fonts', filename)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/count_view")
def count_view():
    return render_template("count_view.html")


@app.route("/area_view")
def area_view():
    return render_template("area_view.html")


@app.route("/map_view")
def map_view():
    return render_template("map_view.html")


@app.route("/map_view_info")
def map_view_info():
    return render_template("map_view_info.html")


@app.route('/video_view')
def video_page():
    # 동영상 목록 가져오기
    import os
    video_files = os.listdir('static/videos/')
    video_files.remove('.placeholder')
    return render_template('video_view.html', video_files=video_files)


# 비디오 피드 라우트
@app.route("/video_feed")
def video_feed():
    return Response(process_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/area_feed")
def area_feed():
    return Response(process_area_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/cam_feed")
def cam_feed():
    return Response(process_cam_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/video_feed/<video_name>')
def video_view_feed(video_name):
    video_path = f'static/videos/{video_name}'
    return Response(detect_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

# 데이터 API 라우트
@app.route("/detection_data_feed")
def process_area_data():
    global complex_ratio, ratio_code
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from webcam.")
            break

        frame_area = frame.shape[0] * frame.shape[1]
        total_person_area = 0
        people_count = 0

        # YOLO 탐지
        results = model(frame)
        detections = []

        for result in results[0].boxes:
            if result.cls == 0:  # 사람 클래스
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                person_area = (x2 - x1) * (y2 - y1)
                total_person_area += person_area
                people_count += 1
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                detections.append(Detection(np.array([center_x, center_y])))

        # 면적 비율 계산
        area_ratio = (total_person_area / frame_area) * 100
        if area_ratio < complex_ratio[0]:
            ratio_code = '1'
        elif complex_ratio[0] <= area_ratio < complex_ratio[1]:
            ratio_code = '2'
        else:
            ratio_code = '3'

        # 추적 업데이트
        tracked_objects = tracker.update(detections)

        # 현재 프레임의 방향 통계
        current_direction_counts = {"left": 0, "right": 0, "up": 0, "down": 0}

        # 추적 결과 시각화 및 방향 계산
        for obj in tracked_objects:
            obj_id = obj.id
            position = obj.estimate
            x, y = map(int, position[0])

            # 이전 위치와 비교하여 이동 방향 계산
            if obj_id in previous_positions:
                prev_x, prev_y = previous_positions[obj_id]
                direction = calculate_direction((prev_x, prev_y), (x, y))
                current_direction_counts[direction] += 1
            previous_positions[obj_id] = (x, y)

        # 현재 프레임에서 가장 많은 이동 방향 찾기
        most_movement_direction = max(current_direction_counts, key=current_direction_counts.get)
        most_movement_count = current_direction_counts[most_movement_direction]

        # JSON 데이터 생성
        data = {
            "ratio_code": ratio_code,
            "people_count": people_count,
            "most_movement_direction": most_movement_direction,
            "most_movement_count": most_movement_count
        }
        return jsonify(data)


@app.route("/get_count_data")
def get_count_data():
    global message_count
    text_data = {
        "message": f"{message_count}"
    }
    return jsonify(text_data)


@app.route("/get_ratio_data")
def get_ratio_data():
    global message_ratio
    text_data = {
        "message": f"{message_ratio}"
    }
    return jsonify(text_data)


@app.route("/get_ratio_code")
def get_ratio_code():
    global ratio_code
    text_data = {
        "message": f"{ratio_code}"
    }
    return jsonify(text_data)


# ============================================================================
# 메인 실행
# ============================================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
