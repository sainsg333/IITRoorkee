from flask import Flask, Response, render_template_string, stream_with_context
import cv2
from ultralytics import YOLO
import pygame
import numpy as np
import requests
import time
import os
from huggingface_hub import hf_hub_download
port=7000
app = Flask(__name__)
import requests
from ultralytics import settings
settings.update({"cache": False, "device": "cpu"})  # Reduce memory
import os
os.environ["ULTRALYTICS_CONFIG_DIR"] = "/tmp/.ultralytics"


# Authenticate and download model
hf_token = "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"

# Download the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="sainsg/IITR",  # Repository ID
    filename="bot_detector_optimized.pt",  # Filename of the model
    use_auth_token=hf_token  # Authentication token
)

# Load the YOLOv8 model using the downloaded file
model = YOLO(model_path)


# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize pygame for alarm
path_alarm = "Alarm/alarm.wav"
pygame.mixer.init()
pygame.mixer.music.load(path_alarm)

# Global variables
is_inside = True
BUZZER_SERVER_IP = "127.0.0.1:5000"
pts = []
roi_set = False
save_folder = "Detected_Bots"
os.makedirs(save_folder, exist_ok=True)
last_save_time = time.time()
bot_outside_roi = False  # Track if a bot is outside the ROI

def trigger_buzzer():
    """Send an HTTP request to the mock ESP server to activate the buzzer"""
    try:
        response = requests.get(f"http://{BUZZER_SERVER_IP}/buzzer")
        print(f"Buzzer Triggered: {response.text}")
    except Exception as e:
        print(f"Failed to trigger buzzer: {e}")

def draw_polygon(event, x, y, flags, param):
    global pts, roi_set
    if event == cv2.EVENT_LBUTTONDOWN and not roi_set:
        pts.append([x, y])
        print(f"Added point: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if not roi_set and len(pts) >= 3:
            roi_set = True
            print("ROI set:", pts)
        else:
            pts.clear()
            roi_set = False
            print("ROI reset.")

def inside_polygon(point, polygon):
    result = cv2.pointPolygonTest(polygon, (point[0], point[1]), False)
    return result >= 0

def play_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

def generate_frames():
    global pts, roi_set, is_inside, last_save_time, bot_outside_roi

    cv2.namedWindow('Bot Monitoring')
    cv2.setMouseCallback('Bot Monitoring', draw_polygon)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5, save=False, save_txt=False, save_conf=False, verbose=False)
        bot_detected = False
        bot_outside_roi = False  # Reset for each frame

        if len(pts) >= 3:
            polygon = np.array(pts, np.int32)

        for result in results:
            for box in result.boxes:
                if int(box.cls) == 0:
                    bot_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, 'Bot', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
                    if len(pts) >= 3:
                        is_inside = inside_polygon((center_x, center_y), polygon)
                    if not is_inside:
                        bot_outside_roi = True
                        print("🚨 BOT OUTSIDE ROI! 🚨")

        if len(pts) > 0:
            if roi_set and len(pts) >= 3:
                cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)
            else:
                for i in range(len(pts)-1):
                    cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), (255, 0, 0), 1)
                if len(pts) >= 3:
                    cv2.line(frame, tuple(pts[-1]), tuple(pts[0]), (255, 0, 0), 1)

        status = f"Bots: {bot_detected}, Outside ROI: {bot_outside_roi}"
        cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if bot_outside_roi:
            cv2.putText(frame, "ALERT: Bot outside ROI!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2)
            play_alarm()
            generate_frames.notified = True  # Prevent duplicate requests
        elif not bot_outside_roi:
            generate_frames.notified = False

        current_time = time.time()
        if bot_detected and (current_time - last_save_time >= 5):
            save_path = os.path.join(save_folder, f"bot_{int(current_time)}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"Saved bot image: {save_path}")
            last_save_time = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template_string('''
        <html>
            <head>
                <title>Bot Monitoring</title>
            </head>
            <body>
                <h1>Bot Monitoring</h1>
                <img src="{{ url_for('video_feed') }}" width="640" height="480">
            </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/notify')
def notifier():
    def event_stream():
        global bot_outside_roi
        while True:
            if bot_outside_roi:
                yield f"data: Bot detected outside ROI!\n\n"
                bot_outside_roi = False  # Reset after sending the notification
            time.sleep(1)  # Polling interval
    return Response(stream_with_context(event_stream()), content_type='text/event-stream')
if __name__ == '__main__':
    app.run(port=port, debug=True)
