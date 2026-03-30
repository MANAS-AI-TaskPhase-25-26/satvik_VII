import cv2 
import numpy as np 
from ultralytics import YOLO 

# ====================================================================
# 1. CONFIGURATION & INITIALIZATION
# ====================================================================

model = YOLO('yolov8n.pt') 

video_path = '/Users/satviksingh/Documents/code/ manas_git/GitHub/satvik_VII/working_directory/neural net problems/opencvtask/Volleyball.mp4'
cap = cv2.VideoCapture(video_path)

# --- HSV COLOR TUNING ---
team1_lower = np.array([9, 101, 0])
team1_upper = np.array([73, 255, 255])

team2_lower = np.array([34, 26, 0])
team2_upper = np.array([163, 233, 117])

ball_lower = np.array([6, 80, 137])
ball_upper = np.array([66, 185, 255])

# --- ALLOWED ZONES (GEOFENCING) ---
team1_allowed_poly = np.array([[91, 378], [899, 372], [765, 115], [224, 117]], np.int32)
team2_allowed_poly = np.array([[914, 582], [79, 584], [214, 263], [772, 262]], np.int32)

# --- BALL TRACKING ---
last_ball_pos = None

# ====================================================================
# 2. MAIN PROCESSING LOOP
# ====================================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1000, 600))
    team1_players = 0
    team2_players = 0

    # -----------------------------------------------------
    # A. YOLO PERSON TRACKER
    # -----------------------------------------------------
    results = model.track(frame, classes=[0], conf=0.3, persist=True, tracker="botsort.yaml", verbose=False)

    # Build player exclusion mask
    player_exclusion_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0])
            pad = 15
            px1, py1 = max(0, px1 - pad), max(0, py1 - pad)
            px2, py2 = min(frame.shape[1], px2 + pad), min(frame.shape[0], py2 + pad)
            cv2.rectangle(player_exclusion_mask, (px1, py1), (px2, py2), 255, -1)

    # -----------------------------------------------------
    # B. SIMPLIFIED BALL DETECTION (Color + Circularity only)
    # -----------------------------------------------------
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ball_mask = cv2.inRange(hsv_frame, ball_lower, ball_upper)

    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)

    # Remove player areas
    ball_mask = cv2.bitwise_and(ball_mask, cv2.bitwise_not(player_exclusion_mask))

    # Find contours and pick best candidate
    b_cnts, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_ball = None
    best_score = -1

    for c in b_cnts:
        area = cv2.contourArea(c)
        if not (15 < area < 400):
            continue

        bx, by, bw, bh = cv2.boundingRect(c)

        # Aspect ratio check
        if not (0.5 < float(bw) / bh < 1.5):
            continue

        # Circularity check
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.65:
            continue

        cx, cy = bx + bw // 2, by + bh // 2

        # Score: circularity + proximity to last position
        score = circularity * 50
        if last_ball_pos is not None:
            dist = np.hypot(cx - last_ball_pos[0], cy - last_ball_pos[1])
            score += max(0, 50 - dist / 6)
        else:
            score += 25

        if score > best_score:
            best_score = score
            best_ball = (cx, cy, bx, by, bw, bh)

    if best_ball:
        cx, cy, bx, by, bw, bh = best_ball
        last_ball_pos = (cx, cy)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.putText(frame, "Ball", (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    # -----------------------------------------------------
    # C. CLASSIFY TEAMS
    # -----------------------------------------------------
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)

            feet_x, feet_y = int((x1 + x2) / 2), y2

            player_crop = frame[y1:y2, x1:x2]
            if player_crop.size == 0:
                continue
            hsv_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)

            t1_pixels = cv2.countNonZero(cv2.inRange(hsv_crop, team1_lower, team1_upper))
            t2_pixels = cv2.countNonZero(cv2.inRange(hsv_crop, team2_lower, team2_upper))

            min_color_pixels = 50

            if t1_pixels > t2_pixels and t1_pixels > min_color_pixels:
                if cv2.pointPolygonTest(team1_allowed_poly, (feet_x, feet_y), False) >= 0:
                    team1_players += 1
                    color, label = (0, 0, 255), "Team 1"
                else:
                    color, label = (0, 165, 255), "T1 (Out of Zone)"
            elif t2_pixels > t1_pixels and t2_pixels > min_color_pixels:
                if cv2.pointPolygonTest(team2_allowed_poly, (feet_x, feet_y), False) >= 0:
                    team2_players += 1
                    color, label = (255, 0, 0), "Team 2"
                else:
                    color, label = (0, 165, 255), "T2 (Out of Zone)"
            else:
                color, label = (200, 200, 200), "Other"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # -----------------------------------------------------
    # D. UI OVERLAY
    # -----------------------------------------------------
    cv2.polylines(frame, [team1_allowed_poly], isClosed=True, color=(0, 0, 255), thickness=1)
    cv2.polylines(frame, [team2_allowed_poly], isClosed=True, color=(255, 0, 0), thickness=1)

    cv2.putText(frame, f"Team 1 Players: {team1_players}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Team 2 Players: {team2_players}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Volleyball Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
