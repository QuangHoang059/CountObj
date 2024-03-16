from ultralytics import YOLO
import cv2
import math
import cvzone
import os
from sort import *
model = YOLO('yolov8n.pt')
VIDEOS_DIR = os.path.join('.', 'source')
video_path = os.path.join(VIDEOS_DIR, 'traffic.mp4')
video_path_out = '{}_out.mp4'.format(video_path)
cap = cv2.VideoCapture(video_path)
cap.set(3, 1288)
cap.set(4, 720)

mask = cv2.imread('./source/Untitled design.png')
mask = cv2.resize(mask, (1280, 720))
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
line = [280, 350, 1000, 350]
count = []
while True:
    _, img = cap.read()
    if img is None:
        break
    imgregion = cv2.bitwise_and(img, mask)
    results = model(imgregion)[0]
    cv2.line(img, line[0:2], line[2:4], (0, 0, 255), 5)
    detections = np.empty((0, 5))
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        curenclass = model.model.names[int(class_id)]
        if curenclass == "car" or curenclass == "truck" or curenclass == "bus" or curenclass == "motorbike" and score > 0.3:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            score = math.ceil(score*100)/100

            cv2.putText(img, f'{curenclass}  {score}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            arr = np.array([x1, y1, x2, y2, int(score)])
            detections = np.vstack((detections, arr))
    resulttracker = tracker.update(detections)
    for result in resulttracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1
        cx, cy = x1+w//2, y1+h//2
        if line[0] < cx < line[2] and line[1]-15 < cy < line[1]+15 and count.count(id) == 0:
            count.append(id)
    # viêt bằng cvzone.putTextRect có khung còn cv2.putText không có khung
    cvzone.putTextRect(img, f'Count: {len(count)}', (50, 50))
    cv2.imshow("Image", img)
    cv2.waitKey(1)
