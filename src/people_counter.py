import cv2
from ultralytics import YOLO
from collections import defaultdict

# ---------- SETTINGS ----------

# Video file
VIDEO_PATH = "people_on_the_road.mp4"

# Display dimensions
DISPLAY_WIDTH = 1000
DISPLAY_HEIGHT = 1000

# Direction of movement considered as entry
# True: Moving from left to right counts as entering
ENTRY_IS_RIGHT = True

# Buffer around the counting line to prevent repeated counting
MARGIN = 20

# Camera settings (for future live webcam implementation)
# CAMERA_INDEX = 1
# cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)


# ---------- LOAD MODEL AND VIDEO ----------

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()


# ---------- TRACKING VARIABLES ----------

# Map original tracking IDs to custom display IDs
id_map = {}
nex_id = 1

# Track how many frames each person appears in
appear = defaultdict(int)

# Remember the last side of the counting line for each person
side = {}

# Counting statistics
entered = 0
exited = 0
peak = 0

# Initial occupancy
initial_people = 0
initialized = False


# ---------- MAIN VIDEO PROCESSING LOOP ----------

while True:

    # Read the next video frame
    ret, frame = cap.read()

    if not ret:
        break

    # Detect and track people (class 0 represents people)
    res = model.track(
        frame,
        classes=[0],
        persist=True,
        verbose=False
    )

    # Create a copy for annotations
    annotated = frame.copy()

    h, w = frame.shape[:2]

    # ---------- VERTICAL COUNTING LINE ----------

    # Position counting line at the centre of the frame
    line_x = w // 2

    cv2.line(
        annotated,
        (line_x, 0),
        (line_x, h),
        (0, 255, 255),
        2
    )

    # ---------- PERSON DETECTION AND TRACKING ----------

    if res[0].boxes.id is not None:

        boxes = res[0].boxes.xyxy.cpu().numpy()
        ids = res[0].boxes.id.cpu().numpy()

        # Initialize occupancy using people detected
        # when tracking IDs first become available
        if not initialized:

            initial_people = len(ids)
            peak = initial_people
            initialized = True

        for box, oid in zip(boxes, ids):

            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box)

            # Calculate centre of detected person
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Count the number of frames each person appears
            appear[oid] += 1

            # Assign custom ID after five appearances
            if appear[oid] >= 5 and oid not in id_map:
                id_map[oid] = nex_id
                nex_id += 1

            # Process confirmed tracking IDs
            if oid in id_map:

                sid = id_map[oid]

                # ---------- DETERMINE PERSON'S POSITION ----------

                # Check which side of the vertical line the person occupies
                if cx > line_x + MARGIN:
                    now = "R"

                elif cx < line_x - MARGIN:
                    now = "L"

                else:
                    # Keep previous position inside the margin
                    now = side.get(oid)

                before = side.get(oid)

                # ---------- COUNT ENTRY AND EXIT ----------

                if now is not None:

                    # Detect when a person changes sides
                    if before is not None and before != now:

                        # Determine direction of movement
                        moved_right = (now == "R")

                        # Update entry or exit count
                        if moved_right == ENTRY_IS_RIGHT:
                            entered += 1
                        else:
                            exited += 1

                    # Save person's latest position
                    side[oid] = now

                # ---------- VISUALIZE DETECTIONS ----------

                # Draw bounding box
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Display tracking ID
                cv2.putText(
                    annotated,
                    f"ID: {sid}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

                # Draw centre point
                cv2.circle(
                    annotated,
                    (cx, cy),
                    5,
                    (0, 255, 0),
                    -1
                )

    # ---------- HEADCOUNT CALCULATION ----------

    # Calculate current occupancy
    inside = max(initial_people + entered - exited, 0)

    # Update highest occupancy recorded
    peak = max(peak, inside)

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

    # ---------- RESIZE VIDEO FOR DISPLAY ----------

    display_frame = cv2.resize(
        annotated,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT)
    )

    # Display processed video
    cv2.imshow("YOLOv8 People Counting System", display_frame)

    # Press Q to stop processing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ---------- RELEASE RESOURCES ----------

cap.release()
cv2.destroyAllWindows()


# ---------- FINAL RESULTS ----------

print("\nVIDEO PROCESSING COMPLETED")
print("--------------------------")
print(f"Initial occupancy: {initial_people}")
print(f"Total entered: {entered}")
print(f"Total exited: {exited}")
print(f"Final occupancy: {inside}")
print(f"Peak occupancy: {peak}")
