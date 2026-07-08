from ultralytics import YOLO
import cv2 as cv
from collections import deque, defaultdict
import math
import glob
import os


# ---------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------
model = YOLO("yolov8n.pt")

# ---------------------------------------------------------------------------------
# 2. Point to your sequential frames (a folder of images, sorted in time order)
# ---------------------------------------------------------------------------------

# ADD A FILEPATH HERE
framesDIR = " "

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

classNames = {'person', 'car', 'bicycle', 'dog'}


# ---------------------------------------------------------------
# 4. Helper: bounding box -> centroid
# ---------------------------------------------------------------
def centroid(x1, y1, x2, y2):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return (cx, cy)


def getFramePaths(folder):
    """Search the given day folder (not the top-level framesDIR) for images
    matching any of the valid extensions."""
    paths = []
    for ext in valid_extensions:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(paths)


# ---------------------------------------------------------
# 5. Direction calculation
# ---------------------------------------------------------
def computeDirection(history):
    if len(history) < 2:
        return None
    xOLD, yOLD = history[0]
    xNEW, yNEW = history[-1]
    dx = xNEW - xOLD
    dy = yNEW - yOLD
    if math.hypot(dx, dy) < 2.0:
        return None
    angle = math.degrees(math.atan2(dy, dx))
    return angle


def angleToCompass(angle):
    if angle is None:
        return "stationary"
    angle = angle % 360
    directions = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
    index = int((angle + 22.5) // 45) % 8
    return directions[index]


# ---------------------------------------------------------
# 6. Output setup -- where annotated frames get saved
# ---------------------------------------------------------

# ADD A FILE PATH HERE
outputDIR = " "  
os.makedirs(outputDIR, exist_ok=True)


def drawOverlay(frame, box, trackID, className, directionLabel, history):
    x1, y1, x2, y2 = [int(v) for v in box.tolist()] 
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = f"ID {trackID} {className} {directionLabel}"
    cv.putText(frame, label, (x1, max(y1 - 10, 0)),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    points = list(history)
    for i in range(1, len(points)):
        pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
        pt2 = (int(points[i][0]), int(points[i][1]))
        cv.line(frame, pt1, pt2, (0, 0, 255), 2)

    cx, cy = points[-1]
    cv.circle(frame, (int(cx), int(cy)), 4, (255, 0, 0), -1)


# ----------------------------------------------------------------------
# Main loop: process each frame in order, run tracking, store centroids
# ----------------------------------------------------------------------
dayFolders = sorted(
    d for d in glob.glob(os.path.join(framesDIR, "*"))
    if os.path.isdir(d)
)

if len(dayFolders) == 0:
    print("Looking in:", os.path.abspath(framesDIR))
    print("Contents:", os.listdir(framesDIR) if os.path.isdir(framesDIR) else "folder doesn't exist")
    raise FileNotFoundError(f"No files found in {framesDIR}")

for dayFolder in dayFolders:
    dayName = os.path.basename(dayFolder)
    framePaths = getFramePaths(dayFolder)

    if len(framePaths) == 0:
        print(f"Warning: no image files found in {dayFolder}, skipping day")
        continue

    print(f"\n--- Starting new tracking session for {dayName} ---")

    queue = defaultdict(lambda: deque(maxlen=3))

    model.predictor = None 

    dayOutputDIR = os.path.join(outputDIR, dayName)
    os.makedirs(dayOutputDIR, exist_ok=True)

    for framePath in framePaths:
        frame = cv.imread(framePath)
        if frame is None:
            print(f"Warning: could not read {framePath}, skipping")
            continue

        result = model.track(frame, persist=True, verbose=False)
        boxes = result[0].boxes
        if boxes.id is None:
            continue

        for box, trackID, clsIDX in zip(boxes.xyxy, boxes.id, boxes.cls):
            className = model.names[int(clsIDX)]

            if className not in classNames:
                continue

            x1, y1, x2, y2 = box.tolist()
            cx, cy = centroid(x1, y1, x2, y2)

            trackID = int(trackID)
            queue[trackID].append((cx, cy))

            angle = computeDirection(queue[trackID])
            directionLabel = angleToCompass(angle)

            if directionLabel == "stationary":
                continue

            drawOverlay(frame, box, trackID, className, directionLabel, queue[trackID])

            print(f"frame={os.path.basename(framePath)} id={trackID} "
                  f"class={className} centroid=({cx:.1f}, {cy:.1f}) "
                  f"direction={directionLabel}")

        outPath = os.path.join(dayOutputDIR, os.path.basename(framePath))
        cv.imwrite(outPath, frame)