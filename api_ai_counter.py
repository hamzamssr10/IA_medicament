import os
import re
import cv2
import sys
import time
import json
import threading
import requests
import numpy as np
import pytesseract


from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from rapidfuzz import process, fuzz
from collections import defaultdict

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EAST_MODEL_PATH = os.path.join(BASE_DIR, "frozen_east_text_detection.pb")

IP = os.getenv("IP", "127.0.0.1")

SNAPSHOT_API_URL = f"http://{IP}/api/v2/fridge/snapshot"

API_HOST = 0.0.0.0 #os.getenv("API_HOST", "0.0.0.0")
API_PORT = 8000 #int(os.getenv("API_PORT", "8000"))

BACKEND_TIMEOUT_SEC = 5
BACKEND_RETRY_COUNT = 3
BACKEND_RETRY_DELAY_SEC = 1

CAMERA_INDEXES = {
    "cam1": 0,
    "cam2": 1,
    "cam3": 2,
    "cam4": 3,
}

MEDICAMENTS = [
    "doliprane",
    "smecta",
    "meteospasmyl",
    "profenid",
    "febrex",
    "compresses steriles",
    "vita c1000",
]

DISPLAY_NAMES = {
    "doliprane": "doliprane",
    "smecta": "smecta",
    "meteospasmyl": "meteospasmyl",
    "profenid": "profenid",
    "febrex": "febrex",
    "compresses steriles": "compresses steriles",
    "vita c1000": "vita c",
}

FUZZY_THRESHOLD = 60
MIN_CROP_H = 8
MIN_CROP_W = 20
MIN_EXPANDED_SCORE = 72
EAST_INPUT_SIZE = (1024, 1024)

CROP_TOP_BOTTOM_PERCENT = 12

VERBOSE = False

# =========================================================
# APP / GLOBALS
# =========================================================
app = FastAPI(title="Medication Counter API Final POST Safe")
REQUEST_LOCK = threading.Lock()
DETECTOR = None

# =========================================================
# CHECKS
# =========================================================
if not os.path.exists(EAST_MODEL_PATH):
    raise FileNotFoundError(f"Fichier EAST introuvable : {EAST_MODEL_PATH}")

cv2.setNumThreads(1)


def log(msg: str) -> None:
    if VERBOSE:
        print(msg)


# =========================================================
# CROP IMAGE
# =========================================================
def crop_top_bottom_percent(frame, percent=12):
    if frame is None:
        return None

    h, w = frame.shape[:2]
    crop_px = int(h * percent / 100)

    y1 = crop_px
    y2 = h - crop_px

    if y2 <= y1:
        return frame

    return frame[y1:y2, 0:w]


# =========================================================
# STARTUP
# =========================================================
@app.on_event("startup")
def startup_event():
    global DETECTOR
    DETECTOR = create_text_detector()
    print("[INFO] EAST model loaded once at startup")
    print(f"[INFO] Backend URL: {SNAPSHOT_API_URL}")


# =========================================================
# POST BACKEND
# =========================================================
def format_medications_payload(global_counts):
    medications = []

    for med, count in sorted(global_counts.items()):
        display_name = DISPLAY_NAMES.get(med, med)
        medications.append({
            "name": display_name,
            "count": int(count)
        })

    return {
        "medications": medications
    }


def send_snapshot_to_backend(global_counts):
    payload = format_medications_payload(global_counts)
    last_error = None

    for attempt in range(1, BACKEND_RETRY_COUNT + 1):
        try:
            response = requests.post(
                SNAPSHOT_API_URL,
                json=payload,
                timeout=BACKEND_TIMEOUT_SEC
            )

            response.raise_for_status()

            return {
                "success": True,
                "sent_payload": payload,
                "backend_status_code": response.status_code,
                "backend_response": response.text,
                "attempt": attempt,
                "error": None
            }

        except requests.RequestException as e:
            last_error = str(e)
            print(
                f"[WARNING] Backend POST failed "
                f"{attempt}/{BACKEND_RETRY_COUNT}: {last_error}"
            )

            if attempt < BACKEND_RETRY_COUNT:
                time.sleep(BACKEND_RETRY_DELAY_SEC)

    return {
        "success": False,
        "sent_payload": payload,
        "backend_status_code": None,
        "backend_response": None,
        "attempt": BACKEND_RETRY_COUNT,
        "error": last_error
    }


# =========================================================
# TEXT / MATCHING
# =========================================================
def clean_ocr(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_label(text):
    return clean_ocr(text).replace(" ", "")


NORMALIZED_MED_MAP = {normalize_label(m): m for m in MEDICAMENTS}
NORMALIZED_KEYS = list(NORMALIZED_MED_MAP.keys())


def token_is_useful(token):
    token = clean_ocr(token)
    return len(token) >= 4


def match_medicament(text, seuil=FUZZY_THRESHOLD):
    text = clean_ocr(text)

    if len(text) < 3:
        return None, 0

    best_match = None
    best_score = 0

    result = process.extractOne(
        text,
        MEDICAMENTS,
        scorer=fuzz.token_sort_ratio
    )

    if result is not None:
        match, score, _ = result
        best_match = match
        best_score = score

    normalized_text = normalize_label(text)

    result2 = process.extractOne(
        normalized_text,
        NORMALIZED_KEYS,
        scorer=fuzz.ratio
    )

    if result2 is not None:
        match2, score2, _ = result2
        if score2 > best_score:
            best_match = NORMALIZED_MED_MAP[match2]
            best_score = score2

    if best_match is None or best_score < seuil:
        return None, best_score

    if abs(len(normalize_label(text)) - len(normalize_label(best_match))) > max(
        4, int(len(normalize_label(best_match)) * 0.7)
    ):
        return None, best_score

    return best_match, best_score


# =========================================================
# EAST DETECTOR
# =========================================================
def create_text_detector():
    detector = cv2.dnn_TextDetectionModel_EAST(EAST_MODEL_PATH)
    detector.setConfidenceThreshold(0.15)
    detector.setNMSThreshold(0.2)
    detector.setInputParams(
        1.0,
        EAST_INPUT_SIZE,
        (123.68, 116.78, 103.94),
        True
    )
    return detector


# =========================================================
# USB CAPTURE
# =========================================================
def capture_one_usb_camera(camera_name, camera_index):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        return {
            "camera": camera_name,
            "status": "error",
            "frame": None,
            "message": f"Impossible d'ouvrir la caméra USB index {camera_index}"
        }

    frame = None

    for _ in range(2):
        ret, tmp = cap.read()
        if ret and tmp is not None:
            frame = tmp

    cap.release()

    if frame is None:
        return {
            "camera": camera_name,
            "status": "error",
            "frame": None,
            "message": f"Impossible de lire une frame depuis {camera_name}"
        }

    frame = crop_top_bottom_percent(frame, CROP_TOP_BOTTOM_PERCENT)

    return {
        "camera": camera_name,
        "status": "ok",
        "frame": frame,
        "message": f"capture réussie + crop {CROP_TOP_BOTTOM_PERCENT}% haut/bas"
    }


def capture_all_usb_cameras():
    results = {}

    for cam_name, cam_index in CAMERA_INDEXES.items():
        results[cam_name] = capture_one_usb_camera(cam_name, cam_index)

    return results


def get_frames_usb_only():
    capture_results = capture_all_usb_cameras()

    usb_ok = all(
        v["status"] == "ok" and v["frame"] is not None
        for v in capture_results.values()
    )

    if not usb_ok:
        return None, "usb", capture_results

    frames_dict = {
        cam_name: capture_results[cam_name]["frame"]
        for cam_name in CAMERA_INDEXES.keys()
    }

    return frames_dict, "usb", capture_results


# =========================================================
# CONCATENATION
# =========================================================
def pad_to_same_height(img1, img2):
    h1, _ = img1.shape[:2]
    h2, _ = img2.shape[:2]

    if h1 == h2:
        return img1, img2

    max_h = max(h1, h2)

    def pad(img, target_h):
        h, w = img.shape[:2]
        pad_bottom = target_h - h
        return cv2.copyMakeBorder(
            img,
            0,
            pad_bottom,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

    return pad(img1, max_h), pad(img2, max_h)


def pad_to_same_width(img1, img2):
    _, w1 = img1.shape[:2]
    _, w2 = img2.shape[:2]

    if w1 == w2:
        return img1, img2

    max_w = max(w1, w2)

    def pad(img, target_w):
        h, w = img.shape[:2]
        pad_right = target_w - w
        return cv2.copyMakeBorder(
            img,
            0,
            0,
            0,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

    return pad(img1, max_w), pad(img2, max_w)


def build_composite_from_frames(frames_dict):
    cam1 = frames_dict["cam1"]
    cam2 = frames_dict["cam2"]
    cam3 = frames_dict["cam3"]
    cam4 = frames_dict["cam4"]

    top_left, top_right = pad_to_same_height(cam1, cam2)
    top_row = np.hstack((top_left, top_right))

    bottom_left, bottom_right = pad_to_same_height(cam3, cam4)
    bottom_row = np.hstack((bottom_left, bottom_right))

    top_row, bottom_row = pad_to_same_width(top_row, bottom_row)
    composite = np.vstack((top_row, bottom_row))

    h1, w1 = top_left.shape[:2]
    h2, w2 = top_right.shape[:2]
    h3, w3 = bottom_left.shape[:2]
    h4, w4 = bottom_right.shape[:2]

    layout = {
        "cam1": {"x1": 0, "y1": 0, "x2": w1, "y2": h1},
        "cam2": {"x1": w1, "y1": 0, "x2": w1 + w2, "y2": h2},
        "cam3": {"x1": 0, "y1": h1, "x2": w3, "y2": h1 + h3},
        "cam4": {"x1": w3, "y1": h1, "x2": w3 + w4, "y2": h1 + h4},
    }

    return composite, layout


def get_camera_from_point(x, y, layout):
    for cam_name, zone in layout.items():
        if zone["x1"] <= x < zone["x2"] and zone["y1"] <= y < zone["y2"]:
            return cam_name
    return "unknown"


# =========================================================
# CROP GEOMETRY
# =========================================================
def crop_rotated_box(image, box, pad=6):
    box = np.array(box, dtype="float32")

    rect = np.zeros((4, 2), dtype="float32")

    s = box.sum(axis=1)
    rect[0] = box[np.argmin(s)]
    rect[2] = box[np.argmax(s)]

    diff = np.diff(box, axis=1)
    rect[1] = box[np.argmin(diff)]
    rect[3] = box[np.argmax(diff)]

    center = np.mean(rect, axis=0)

    padded_rect = []

    for p in rect:
        v = p - center
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        padded_rect.append(p + v * pad)

    rect = np.array(padded_rect, dtype="float32")
    tl, tr, br, bl = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    if maxWidth <= 0 or maxHeight <= 0:
        return None

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def crop_expanded_rect(image, box, expand_x=0.18, expand_y=0.45):
    pts = np.array(box, dtype=np.float32)

    x_min = np.min(pts[:, 0])
    y_min = np.min(pts[:, 1])
    x_max = np.max(pts[:, 0])
    y_max = np.max(pts[:, 1])

    w = x_max - x_min
    h = y_max - y_min

    x1 = int(max(0, x_min - expand_x * w))
    y1 = int(max(0, y_min - expand_y * h))
    x2 = int(min(image.shape[1], x_max + expand_x * w))
    y2 = int(min(image.shape[0], y_max + expand_y * h))

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


# =========================================================
# OCR PREPROCESS
# =========================================================
def preprocess_for_ocr_simple(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]

    if h < 20 or w < 80:
        scale = 3
    elif h < 35 or w < 130:
        scale = 2
    else:
        scale = 1

    if scale != 1:
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh


def preprocess_for_ocr_enhanced(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]

    if h < 20 or w < 80:
        scale = 3
    else:
        scale = 2

    gray = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return otsu


# =========================================================
# OCR
# =========================================================
def run_ocr_best(cropped):
    proc1 = preprocess_for_ocr_simple(cropped)
    proc2 = preprocess_for_ocr_enhanced(cropped)

    best_text = ""
    best_med = None
    best_score = 0

    text = pytesseract.image_to_string(
        proc1,
        lang="eng+fra",
        config="--oem 3 --psm 6"
    )

    clean_text = clean_ocr(text)

    if len(clean_text) >= 3:
        med, score = match_medicament(clean_text)
        best_text, best_med, best_score = clean_text, med, score

        if best_score >= 92:
            return best_text, best_med, best_score

    text = pytesseract.image_to_string(
        proc2,
        lang="eng+fra",
        config="--oem 3 --psm 6"
    )

    clean_text = clean_ocr(text)

    if len(clean_text) >= 3:
        med, score = match_medicament(clean_text)

        if score > best_score:
            best_text, best_med, best_score = clean_text, med, score

        if best_score >= 92:
            return best_text, best_med, best_score

    if best_med is None:
        text = pytesseract.image_to_string(
            proc2,
            lang="eng+fra",
            config="--oem 3 --psm 7"
        )

        clean_text = clean_ocr(text)

        if len(clean_text) >= 3:
            med, score = match_medicament(clean_text)

            if score > best_score:
                best_text, best_med, best_score = clean_text, med, score

    return best_text, best_med, best_score


# =========================================================
# DUPLICATION CONTROL
# =========================================================
def is_far_enough(key, box, detected_positions, threshold=100):
    cx = np.mean([p[0] for p in box])
    cy = np.mean([p[1] for p in box])

    positions = detected_positions.get(key, [])

    for px, py in positions:
        dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)

        if dist < threshold:
            return False

    return True


# =========================================================
# PROCESS IMAGE
# =========================================================
def process_composite_image(image, detector, layout):
    result = {
        "camera": "composite",
        "status": "ok",
        "counts": {},
        "counts_by_camera": {},
        "detections": []
    }

    if image is None:
        result["status"] = "error"
        result["message"] = "Image composite invalide"
        return result

    if detector is None:
        result["status"] = "error"
        result["message"] = "Détecteur EAST non initialisé"
        return result

    try:
        boxes, _ = detector.detect(image)
    except cv2.error as e:
        result["status"] = "error"
        result["message"] = f"Erreur EAST sur image composite: {str(e)}"
        return result

    med_count = defaultdict(int)
    med_count_by_camera = defaultdict(lambda: defaultdict(int))
    detected_positions = {}

    if boxes is None or len(boxes) == 0:
        result["message"] = "Aucune zone de texte détectée"
        return result

    for i, box in enumerate(boxes):
        best_text = ""
        best_med = None
        best_score = 0
        best_crop = ""

        cx = float(np.mean([p[0] for p in box]))
        cy = float(np.mean([p[1] for p in box]))
        source_camera = get_camera_from_point(cx, cy, layout)

        cropped_rotated = crop_rotated_box(image, box, pad=6)

        if cropped_rotated is not None:
            h, w = cropped_rotated.shape[:2]

            if h >= MIN_CROP_H and w >= MIN_CROP_W:
                text, med, score = run_ocr_best(cropped_rotated)

                if score > best_score:
                    best_text = text
                    best_med = med
                    best_score = score
                    best_crop = "rotated"

        if best_score < 80:
            cropped_expanded = crop_expanded_rect(
                image,
                box,
                expand_x=0.18,
                expand_y=0.45
            )

            if cropped_expanded is not None:
                h, w = cropped_expanded.shape[:2]

                if h >= MIN_CROP_H and w >= MIN_CROP_W:
                    text, med, score = run_ocr_best(cropped_expanded)

                    if score > best_score and score >= MIN_EXPANDED_SCORE:
                        best_text = text
                        best_med = med
                        best_score = score
                        best_crop = "expanded"

        if len(best_text) < 3:
            continue

        found_in_box = set()

        if best_med:
            found_in_box.add(best_med)

        if best_med is None:
            for word_text in best_text.split():
                if not token_is_useful(word_text):
                    continue

                med_word, _ = match_medicament(word_text)

                if med_word:
                    found_in_box.add(med_word)

        counted_here = []

        for med in found_in_box:
            key = f"{source_camera}::{med}"

            if is_far_enough(key, box, detected_positions, threshold=100):
                med_count[med] += 1
                med_count_by_camera[source_camera][med] += 1

                if key not in detected_positions:
                    detected_positions[key] = []

                detected_positions[key].append((cx, cy))
                counted_here.append(med)

        result["detections"].append({
            "box_id": i,
            "source_camera": source_camera,
            "ocr_text": best_text,
            "matched_medicaments": list(found_in_box),
            "counted": counted_here,
            "best_score": float(best_score),
            "best_crop": best_crop,
            "center": [cx, cy]
        })

    result["counts"] = dict(med_count)
    result["counts_by_camera"] = {
        cam: dict(vals)
        for cam, vals in med_count_by_camera.items()
    }

    return result


# =========================================================
# API ROUTES
# =========================================================
@app.get("/")
def root():
    return {
        "message": "Medication Counter API is running",
        "send_to": SNAPSHOT_API_URL,
        "capture_endpoint": f"http://127.0.0.1:{API_PORT}/capture_and_process",
        "crop_top_bottom_percent": CROP_TOP_BOTTOM_PERCENT,
        "backend_timeout_sec": BACKEND_TIMEOUT_SEC,
        "backend_retry_count": BACKEND_RETRY_COUNT
    }


@app.get("/health")
def health():
    if REQUEST_LOCK.locked():
        return {"status": "busy"}

    return {"status": "ok"}


@app.post("/restart")
def restart_api():
    print("[INFO] Restart demandé depuis /restart")
    os.execv(sys.executable, [sys.executable] + sys.argv)


@app.post("/capture_and_process")
def capture_and_process():
    acquired = REQUEST_LOCK.acquire(blocking=False)

    if not acquired:
        return JSONResponse(
            status_code=429,
            content={
                "status": "busy",
                "message": "Une requête est déjà en cours de traitement. Réessayez après quelques secondes."
            }
        )

    try:
        now_dt = datetime.now()

        frames_dict, source_used, source_results = get_frames_usb_only()

        if frames_dict is None:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Impossible de récupérer les images depuis les caméras USB",
                    "source_used": source_used,
                    "source_results": {
                        k: {
                            "status": v["status"],
                            "message": v["message"]
                        }
                        for k, v in source_results.items()
                    }
                }
            )

        composite_image, layout = build_composite_from_frames(frames_dict)
        result = process_composite_image(composite_image, DETECTOR, layout)

        send_result = send_snapshot_to_backend(result.get("counts", {}))

        if not send_result["success"]:
            return JSONResponse(
                status_code=207,
                content={
                    "status": "partial_success",
                    "message": "OCR terminé, mais l'envoi vers le backend a échoué",
                    "date": now_dt.strftime("%d/%m/%Y %H:%M"),
                    "sent_to": SNAPSHOT_API_URL,
                    "payload": send_result["sent_payload"],
                    "backend_error": send_result["error"],
                    "ocr_result": result
                }
            )

        return JSONResponse(
            content={
                "status": "success",
                "date": now_dt.strftime("%d/%m/%Y %H:%M"),
                "sent_to": SNAPSHOT_API_URL,
                "payload": send_result["sent_payload"],
                "backend_status_code": send_result["backend_status_code"],
                "backend_response": send_result["backend_response"],
                "ocr_result": result
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Erreur interne: {str(e)}"
            }
        )

    finally:
        if REQUEST_LOCK.locked():
            REQUEST_LOCK.release()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    uvicorn.run(
        "api_ai_counter:app",
        host=API_HOST,
        port=API_PORT,
        reload=False
    )