import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ---------- SETTINGS ----------
CAMERA_INDEX = 1
ENTRY_IS_RIGHT = True
MARGIN = 20

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

id_map = {}
nex_id = 1
appear = defaultdict(int)
side = {}

entered, exited, peak = 0, 0, 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    res = model.track(frame, classes=[0], persist=True, verbose=False)

    annotated = frame.copy()

    h, w = frame.shape[:2]

    # ---------- VERTICAL COUNTING LINE ----------
    line_x = w // 2

    cv2.line(
        annotated,
        (line_x, 0),
        (line_x, h),
        (0, 255, 255),
        2
    )

    if res[0].boxes.id is not None:

        boxes = res[0].boxes.xyxy.cpu().numpy()
        ids = res[0].boxes.id.cpu().numpy()

        for box, oid in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            appear[oid] += 1

            if appear[oid] >= 5 and oid not in id_map:
                id_map[oid] = nex_id
                nex_id += 1

            if oid in id_map:

                sid = id_map[oid]

                # ---------- CHECK WHICH SIDE OF LINE ----------
                if cx > line_x + MARGIN:
                    now = "R"

                elif cx < line_x - MARGIN:
                    now = "L"

                else:
                    now = side.get(oid)

                before = side.get(oid)

                # ---------- COUNT ENTRY AND EXIT ----------
                if now is not None:

                    if before is not None and before != now:

                        moved_right = (now == "R")

                        if moved_right == ENTRY_IS_RIGHT:
                            entered += 1
                        else:
                            exited += 1

                        inside = max(entered - exited, 0)
                        peak = max(peak, inside)

                    side[oid] = now

                # ---------- DRAW DETECTION ----------
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated,
                    f"ID: {sid}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

                cv2.circle(
                    annotated,
                    (cx, cy),
                    5,
                    (0, 255, 0),
                    -1
                )

    # ---------- HEADCOUNT ----------
    inside = max(entered - exited, 0)

    # ---------- DISPLAY STATISTICS ----------
    cv2.putText(
        annotated,
        f"Entered: {entered}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        annotated,
        f"Exited: {exited}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        annotated,
        f"Total People Inside: {inside}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        annotated,
        f"Peak: {peak}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow("Headcount", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
