import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

# ---------- SETTINGS ----------
CAMERA_INDEX = 1
ENTRY_IS_DOWN = True      # True: moving down the screen counts as entering
MARGIN = 20               # pixels either side of the line, stops flicker

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(CAMERA_INDEX)

id_map = {}
nex_id = 1
appear = defaultdict(int)
side = {}
recent = deque(maxlen=15)

entered, exited, peak = 0, 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    res = model.track(frame, classes=[0], persist=True, verbose=False)
    annotated = frame.copy()
    h, w = frame.shape[:2]
    line_y = h // 2
    cv2.line(annotated, (0, line_y), (w, line_y), (0, 255, 255), 2)
    current = 0

    if res[0].boxes.id is not None:
        boxes = res[0].boxes.xyxy.numpy()
        ids = res[0].boxes.id.numpy()

        for box, oid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            appear[oid] += 1

            if appear[oid] >= 5 and oid not in id_map:
                id_map[oid] = nex_id
                nex_id += 1

            if oid in id_map:
                current += 1
                sid = id_map[oid]

                # which side of the line is this person on?
                if cy > line_y + MARGIN:
                    now = "D"
                elif cy < line_y - MARGIN:
                    now = "U"
                else:
                    now = side.get(oid)  # inside the margin: keep old side

                before = side.get(oid)
                if now is not None:
                    if before is not None and before != now:
                        moved_down = (now == "D")
                        if moved_down == ENTRY_IS_DOWN:
                            entered += 1
                        else:
                            exited += 1
                        peak = max(peak, max(entered - exited, 0))
                    side[oid] = now

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"ID: {sid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)

    recent.append(current)
    in_view = int(np.median(recent))
    inside = max(entered - exited, 0)

    cv2.putText(annotated, f"In view: {in_view}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(annotated, f"Entered: {entered}  Exited: {exited}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(annotated, f"Inside: {inside}  Peak: {peak}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Headcount", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
