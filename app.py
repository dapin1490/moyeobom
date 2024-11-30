import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# YOLO 모델 로드
model = YOLO("yolo11x.pt")

# 웹캠 캡처
cap = cv2.VideoCapture(0)

# OpenCV 추적기 초기화
tracker = cv2.legacy.TrackerCSRT_create()

@app.route("/")
def index():
    return render_template("index.html")

def process_frames():
    initBB = None  # 추적 초기화 상태

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from webcam.")
            break

        # 추적 중이 아닌 경우 YOLO로 탐지
        if initBB is None:
            results = model(frame)
            for result in results[0].boxes:
                if result.cls == 0:  # 사람 클래스 (YOLO 기준 0)
                    x1, y1, x2, y2 = map(int, result.xyxy)
                    initBB = (x1, y1, x2 - x1, y2 - y1)
                    tracker.init(frame, initBB)
                    break

        # 추적
        else:
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = map(int, box)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                initBB = None  # 추적 실패 시 다시 초기화

        # 프레임을 JPEG로 인코딩
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(process_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
