import cv2
import mediapipe as mp
import serial
import time
from scipy.spatial import distance

# serial ports change according to your system
arduino = serial.Serial("COM3", 9600)   # Arduino
esp32 = serial.Serial("COM6", 9600)     # ESP32
time.sleep(2)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

cap = cv2.VideoCapture(0)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def EAR(eye_points, landmarks, w, h):
    points = []
    for point in eye_points:
        x = int(landmarks[point].x * w)
        y = int(landmarks[point].y * h)
        points.append((x, y))


    A = distance.euclidean(points[1], points[5])
    B = distance.euclidean(points[2], points[4])
    C = distance.euclidean(points[0], points[3])

    return (A + B) / (2.0 * C), points

# setting
THRESHOLD = 0.23
FPS = 20

FRAME_LIMIT_3SEC = FPS * 3
FRAME_LIMIT_5SEC = FPS * 5

counter = 0
arduino_triggered = False
esp_triggered = False

while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_ear, left_points = EAR(LEFT_EYE, landmarks, w, h)
            right_ear, right_points = EAR(RIGHT_EYE, landmarks, w, h)

            ear = (left_ear + right_ear) / 2

            # square eyes
            for eye in [left_points, right_points]:
                x_vals = [p[0] for p in eye]
                y_vals = [p[1] for p in eye]

                x_min, x_max = min(x_vals), max(x_vals)
                y_min, y_max = min(y_vals), max(y_vals)

                color = (0, 255, 0)  # Green

                if ear < THRESHOLD:
                    color = (0, 0, 255)  # Red if sleepy

                cv2.rectangle(frame,
                              (x_min, y_min),
                              (x_max, y_max),
                              color, 2)

            # main logic
            if ear < THRESHOLD:
                counter += 1

                # 3 seconds = Arduino
                if counter > FRAME_LIMIT_3SEC and not arduino_triggered:
                    print("3 sec -> Sending S to Arduino")
                    arduino.write(b'S')
                    arduino_triggered = True

                # 5 seconds = ESP32
                if counter > FRAME_LIMIT_5SEC and not esp_triggered:
                    print("5 sec -> Sending S to ESP32")
                    esp32.write(b'S')
                    esp_triggered = True

                cv2.putText(frame,
                            "SLEEP DETECTED!",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2)

            else:
                counter = 0

                if arduino_triggered:
                    print("Driver Awake -> Sending A")
                    arduino.write(b'A')

                arduino_triggered = False
                esp_triggered = False

            # ear value
            cv2.putText(frame,
                        f"EAR: {ear:.2f}",
                        (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
esp32.close()
cv2.destroyAllWindows()
    