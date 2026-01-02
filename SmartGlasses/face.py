# filepath: c:\Users\dell\Desktop\SmartGlasses\face.py
# Requires: pip install opencv-python
import os
import sys
import time
import urllib.request

try:
    import cv2
except Exception:
    print("Error: OpenCV (cv2) not found. Install with: pip install opencv-python")
    sys.exit(1)

# Face detector
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Object detection (MobileNet-SSD)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PROTO = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
MODEL = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(PROTO) or not os.path.exists(MODEL):
    try:
        print("Downloading MobileNet-SSD model (~20MB)...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt",
            PROTO,
        )
        urllib.request.urlretrieve(
            "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel",
            MODEL,
        )
    except Exception as e:
        print("Failed to download model:", e)
        print("Place MobileNetSSD_deploy.prototxt and .caffemodel into", MODEL_DIR)
        # continue; object detection will be skipped if model missing

net = None
if os.path.exists(PROTO) and os.path.exists(MODEL):
    net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

prev = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Object detection (if model loaded) - draw emphasized boxes for bottle (and phone if detected by model)
    objects = []
    bottle_boxes = []
    if net is not None:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        (h, w) = frame.shape[:2]
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf > 0.5:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")
                label = CLASSES[idx] if idx < len(CLASSES) else str(idx)
                objects.append(label)

                # choose color/weight for key targets
                if label == "bottle":
                    color = (0, 0, 255)       # red for bottle
                    thickness = 3
                    bottle_boxes.append((startX, startY, endX, endY))
                elif label in ("cell phone", "phone", "mobile"):
                    color = (0, 255, 255)     # cyan for phone (if model provides it)
                    thickness = 3
                else:
                    color = (255, 128, 0)
                    thickness = 2

                text = f"{label}: {conf:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, thickness)
                ty = startY - 10 if startY - 10 > 10 else startY + 20
                cv2.putText(frame, text, (startX, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Improved phone heuristic detection (contour + morphology + shape checks)
    phone_boxes = []
    gray_small = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_small, (5, 5), 0)

    # Close gaps and remove small noise to get cleaner contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.adaptiveThreshold(closed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    edges = cv2.Canny(closed, 50, 150)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000 or area > 80000:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w_rect, h_rect), angle = rect
        if w_rect <= 0 or h_rect <= 0:
            continue

        # normalize aspect ratio to be >=1
        ar = max(w_rect, h_rect) / min(w_rect, h_rect)

        # phones are elongated rectangles: require reasonable aspect ratio
        if ar < 1.2 or ar > 6.0:
            continue

        # extent: how much of the bounding rect is filled by the contour
        box_area = w_rect * h_rect
        extent = area / box_area if box_area > 0 else 0
        if extent < 0.35 or extent > 0.95:
            continue

        # polygonal approximation (prefer 4-6 vertex shapes)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) < 4 or len(approx) > 8:
            # allow slightly noisy rectangles but filter out very irregular shapes
            continue

        # edge density inside the candidate box (phones have visible edges)
        box_pts = cv2.boxPoints(rect).astype(int)
        x_coords = box_pts[:, 0]
        y_coords = box_pts[:, 1]
        x1, x2 = max(0, x_coords.min()), min(frame.shape[1], x_coords.max())
        y1, y2 = max(0, y_coords.min()), min(frame.shape[0], y_coords.max())
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        roi_edges = edges[y1:y2, x1:x2]
        edge_density = cv2.countNonZero(roi_edges) / float((x2 - x1) * (y2 - y1))
        if edge_density < 0.004:  # tweakable threshold
            continue

        # avoid overlapping face or detected bottle boxes
        overlap_face = False
        for (fx, fy, fw, fh) in faces:
            if x1 < fx + fw and fx < x2 and y1 < fy + fh and fy < y2:
                overlap_face = True
                break
        if overlap_face:
            continue

        overlap_bottle = False
        for (bx1, by1, bx2, by2) in bottle_boxes:
            if x1 < bx2 and bx1 < x2 and y1 < by2 and by1 < y2:
                overlap_bottle = True
                break
        if overlap_bottle:
            continue

        # passed checks -> draw rotated rect and label
        cv2.drawContours(frame, [box_pts], 0, (0, 255, 255), 3)
        cv2.putText(frame, "Phone", (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        phone_boxes.append((x1, y1, x2 - x1, y2 - y1))

    # Wall detection via edges + Hough lines (detect strong vertical/horizontal lines)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges2 = cv2.Canny(gray_blur, 50, 150)
    lines = cv2.HoughLinesP(edges2, 1, (3.14159 / 180), threshold=80, minLineLength=80, maxLineGap=10)
    wall_count = 0
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = x2 - x1
            dy = y2 - y1
            # treat near-vertical or near-horizontal as walls
            if abs(dx) < 20 or abs(dy) < 20:
                wall_count += 1
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # FPS smoothing
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / (now - prev)) if (now - prev) > 0 else fps
    prev = now

    # Overlay info
    info = f"Faces: {len(faces)}  Objects: {len(objects)}  Phones: {len(phone_boxes)}  Walls: {wall_count}  FPS: {fps:.1f}"
    cv2.putText(frame, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Live Face/Object/Wall Detection", frame)
    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord("q")):
        break
    if k == ord("s"):
        cv2.imwrite("capture.png", frame)

cap.release()
cv2.destroyAllWindows()