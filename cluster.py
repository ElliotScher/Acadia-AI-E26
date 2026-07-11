import shutil
from pathlib import Path
import cv2 as cv
from ultralytics import YOLO

yoloModel = "yolov8n.pt"
targetClasses = {"person", "car", "bicycle"}
confidence = 0.4
groupDistance = 60

# INSERT FILE PATH HERE
inputPath = " "
outputPath = "output"
extension = {".jpg"}

batchLimit = None
clearOutput = True

# Deletes all contents of a folder (but keeps the folder itself).
# Safe no-op if the folder doesn't exist yet.
def clearOutputDIR(outputDIR):
    outputDIR = Path(outputDIR)
    if not outputDIR.exists():
        return
    for item in outputDIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    print(f"Cleared existing contents of {outputDIR}")

def load_image(inputDIR):
    imgInput = Path(inputDIR)
    return [p for p in imgInput.rglob("*") if p.suffix.lower() in extension]

# Checks whether two boxes are within distance pixels of each other.
def _boxes_close(a, b, distance):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]

    ax1e, ay1e = ax1 - distance, ay1 - distance
    ax2e, ay2e = ax2 + distance, ay2 + distance

    noOverlap = bx2 < ax1e or bx1 > ax2e or by2 < ay1e or by1 > ay2e
    return not noOverlap

# Groups nearby detections into single combined boxes, REGARDLESS of class --
# e.g. a person and a bike standing close together become one cluster.
# Each resulting cluster reports how many of each class it contains.
def groupDetections(detections, distance=groupDistance):
    n = len(detections)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # compare every pair of boxes regardless of class -- proximity alone
    # decides whether they belong to the same cluster
    for i in range(n):
        for j in range(i+1, n):
            if _boxes_close(detections[i], detections[j], distance):
                union(i, j)

    group = {}
    for i in range(n):
        root = find(i)
        group.setdefault(root, []).append(i)
    
    merge = []
    for idxs in group.values():
        x1 = min(detections[i][0] for i in idxs)
        y1 = min(detections[i][1] for i in idxs)
        x2 = max(detections[i][2] for i in idxs)
        y2 = max(detections[i][3] for i in idxs)

        classCounts = {}
        for i in idxs:
            label = detections[i][4]
            classCounts[label] = classCounts.get(label, 0) + 1

        merge.append((x1, y1, x2, y2, classCounts))
    return merge


# Draw rectangles and guessed labels onto a copy of the image.
def draw_boxes(image, boxes): 
    output = image.copy()
    for (x1, y1, x2, y2, classCounts) in boxes:
        cv.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = ", ".join(f"{label} x{count}" for label, count in classCounts.items())
        cv.putText(output, text, (x1, max(y1 - 8, 0)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return output


def process_image(model, imagePaths, inputDIR, outputDIR): 
    inputDIR = Path(inputDIR)
    outputDIR = Path(outputDIR)
    total = len(imagePaths)

    # per-class running stats: count, confidence sum, min, max
    stats = {}
    def recordStat(label, conf):
        s = stats.setdefault(label, {"count": 0, "sum": 0.0, "min": conf, "max": conf})
        s["count"] += 1
        s["sum"] += conf
        s["min"] = min(s["min"], conf)
        s["max"] = max(s["max"], conf)

    for i, imgPath in enumerate(imagePaths, 1): 
        try:
            result = model(str(imgPath), conf=confidence, verbose=False)[0]
        except Exception as e:
            print(f"[{i}/{total}] {imgPath.name}: SKIPPED ({e})")
            continue
        image = result.orig_img
        detections = []
        for box in result.boxes:
            clsId = int(box.cls[0])
            label = model.names[clsId]
            if label not in targetClasses:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((x1, y1, x2, y2, label, conf))
            recordStat(label, conf)

        if not detections:
            continue

        groups = groupDetections(detections)
        annotated = draw_boxes(image, groups)
        relative = imgPath.relative_to(inputDIR)
        savePath = outputDIR / relative
        savePath.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(savePath), annotated)

        classCounts = {}
        for (_, _, _, _, counts) in groups:
            for label, count in counts.items():
                classCounts[label] = classCounts.get(label, 0) + count
        
        breakdown = ", ".join(f"{label}: {count}" for label, count in classCounts.items())

        print(f"[{i}/{total}] {imgPath.name}: {len(groups)} cluster(s) -- {breakdown}")
    return stats

def summary(stats):
    print("\n=== Detection Confidence Summary ===")
    if not stats:
        print("No detections matching target classes were found.")
        return
    
    for label, s in stats.items():
        avgConf = s["sum"] / s["count"]
        count = s["count"]
        minConf = s["min"]
        maxConf = s["max"]
        print(f"{label:10s} count={count:<6d}"
              f"avg conf={avgConf:.3f} min={minConf:.3f} max={maxConf:.3f}")

def main():
    inputDIR = Path(inputPath)
    outputDIR = Path(outputPath)
    outputDIR.mkdir(parents=True, exist_ok=True)
    if clearOutput:
        clearOutputDIR(outputDIR)
    imagePaths = load_image(inputDIR)
    if not imagePaths:
        print(f"No images found in {inputDIR}")
        return
    if batchLimit is not None:
        imagePaths = imagePaths[:batchLimit]
    print(f"Found {len(imagePaths)} image(s) to process (batch limit = {batchLimit})")
    model = YOLO(yoloModel)
    stats = process_image(model, imagePaths, inputDIR, outputDIR)
    summary(stats)

if __name__ == "__main__":
    main()