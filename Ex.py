from mpi4py import MPI
import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
import time
import smtplib
import pygame

# Define Kalman filter class
class KalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.center = None

    def update(self, measurement):
        self.center = measurement
        self.kalman.correct(self.center)

    def predict(self):
        prediction = self.kalman.predict()
        return prediction

def detect_objects(video_path, rank):
    print(f'Process {rank} started processing {video_path}')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Error: Process {rank} could not open video {video_path}')
        return
    
    model = YOLO('best.pt')
    kalman_filter = KalmanFilter()
    classnames = ['Smoke', 'Spark', 'fire', 'flame']
    fps_start_time = time.time()
    fps_counter = 0
    lock_time = 1
    locked_objects = {}
    lock_start_times = {}
    initial_confidence = {}
    max_confidence = {}
    class_colors = {'Smoke': (0, 255, 255), 'Spark': (0, 165, 255), 'fire': (255, 0, 0), 'flame': (0, 0, 255)}
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'nasarullah.lak@gmail.com'
    smtp_password = 'auafxkdopxruvvcj'
    recipient_email = 'nasarullah.lak@gmail.com'
    pygame.init()
    alarm_sound = pygame.mixer.Sound("alarm-sound.mp3")
    alert_sent = False
    last_detection_time = 0
    previous_confidence = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f'Process {rank} finished processing {video_path}')
            break

        frame = cv2.resize(frame, (640, 480))
        result = model(frame, stream=True)
        detection_occurred = False

        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0] * 100
                Class = int(box.cls[0])
                if confidence > 75:
                    detection_occurred = True
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    color = class_colors[classnames[Class]]
                    if Class not in locked_objects or (time.time() - lock_start_times[Class]) > lock_time:
                        locked_objects[Class] = (x1, y1, x2, y2)
                        lock_start_times[Class] = time.time()
                        initial_confidence[Class] = confidence
                        max_confidence[Class] = max_confidence.get(Class, 0)
                        previous_confidence[Class] = confidence
                    elif confidence > max_confidence[Class]:
                        max_confidence[Class] = confidence
                        if confidence > previous_confidence[Class]:
                            previous_confidence[Class] = confidence

        if detection_occurred:
            if not alert_sent or time.time() - last_detection_time > 60:
                subject = 'Fire Alert!'
                body = f'Fire detected: {classnames[Class]} with confidence {confidence}%'
                message = f'Subject: {subject}\n\n{body}'
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(smtp_username, smtp_password)
                    server.sendmail(smtp_username, recipient_email, message)
                pygame.mixer.Sound.play(alarm_sound)
                alert_sent = True
                last_detection_time = time.time()

        for Class in locked_objects:
            if time.time() - lock_start_times[Class] <= lock_time:
                x1, y1, x2, y2 = locked_objects[Class]
                if max_confidence[Class] > 50:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    color = class_colors[classnames[Class]]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                    cvzone.putTextRect(frame, f'{classnames[Class]} {int(max_confidence[Class])}%',
                                       [x1 + 8, y1 + 100], scale=1.5, thickness=2)

        fps_counter += 1
        if time.time() - fps_start_time >= 1:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_counter = 0

        cv2.putText(frame, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(f'Frame {rank}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    video_paths = [
        'mongphali.mp4',
        'VID_20240506_093141.mp4',
        'mongphali.mp4'
          # Add the path to your fourth video here
    ]

    if rank < len(video_paths):
        detect_objects(video_paths[rank], rank)
    else:
        print(f'Process {rank} is idle.')

    comm.Barrier()  # Wait for all processes to complete
