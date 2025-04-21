import cv2
import numpy as np
import requests
import time
from dotenv import load_dotenv
import os

load_dotenv()

prototxt_path = "MobileNetSSD_deploy.prototxt"
model_path = "MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

FLOW_URL = os.getenv("FLOW_URL")

def send_alert(motion_detected, threat_detected):
    """Send alert to Power Automate flow."""
    payload = {
        "motionDetected": motion_detected,
        "threatDetected": threat_detected
    }
    try:
        response = requests.post(
            FLOW_URL,
            headers={"Content-Type": "application/json"},
            json=payload
        )
        if response.status_code in (200, 202):
            print("✅ Alert sent successfully!")
        else:
            print(f"⚠️ Failed to send alert. Status code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"⚠️ Error sending alert: {e}")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("⚠️ Error: Could not open webcam.")
    exit()

print("Human detection started. Press 'q' to quit anytime.")

last_alert_time = 0
cooldown_seconds = 15

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(resized_frame, 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    human_detected = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] == "person":
                human_detected = True
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"Person: {confidence * 100:.1f}%"
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    current_time = time.time()

    if human_detected:
        if current_time - last_alert_time >= cooldown_seconds:
            print("🚨 Human detected! Sending real alert...")
            send_alert(motion_detected=True, threat_detected=True)
            last_alert_time = current_time

    cv2.imshow("Live Human Detection", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Exiting...")
