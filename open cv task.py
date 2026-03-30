import cv2 
import numpy as np 
from ultralytics import YOLO 

# ====================================================================
# 1. CONFIGURATION & INITIALIZATION
# ====================================================================

# Load YOLOv8 Nano: Lightweight and fast, perfect for real-time video.
model = YOLO('yolov8n.pt') 

video_path = 'Volleyball.mp4' # Ensure your path is correct
cap = cv2.VideoCapture(video_path)

# --- HSV COLOR TUNING ---
# These define the 'color range' for team jerseys and the ball.
# HSV (Hue, Saturation, Value) is better than RGB for color detection
# because it separates color (Hue) from brightness (Value).
team1_lower = np.array([9, 101, 0])
team1_upper = np.array([73, 255, 255])

team2_lower = np.array([34, 26, 0])
team2_upper = np.array([163, 233, 117])

ball_lower = np.array([6, 80, 137])
ball_upper = np.array([66, 185, 255])

# --- ALLOWED ZONES (GEOFENCING) ---
# Defines the 4-point polygons representing each team's side of the court.
# We use this to verify if a detected player is actually in their valid area.
team1_allowed_poly = np.array([[91, 378], [899, 372], [765, 115], [224, 117]], np.int32)
team2_allowed_poly = np.array([[914, 582], [79, 584], [214, 263], [772, 262]], np.int32)

# --- MOTION DETECTION SETUP ---
prev_gray = None          # Stores the previous frame to calculate movement
MOTION_THRESHOLD = 25     # Sensitivity: how much a pixel must change to be "moving"
MIN_MOTION_PIXELS = 8     # Noise filter: minimum movement area to consider it a ball

# --- SINGLE BALL LOGIC ---
# Used to track the ball across frames and filter out stationary yellow objects.
last_ball_pos = None   # Stores (x, y) coordinates of the ball from the previous frame

# ====================================================================
# 2. MAIN PROCESSING LOOP
# ====================================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Standardize size for consistent coordinate math across different videos
    frame = cv2.resize(frame, (1000, 600))
    
    team1_players = 0
    team2_players = 0

    # Convert to grayscale for motion and contour detection (faster than color)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------------------
    # A. MOTION MASK
    # Logic: The ball is usually moving fast. By subtracting the last frame
    # from the current one, we highlight only the moving parts.
    # -----------------------------------------------------
    if prev_gray is None:
        motion_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    else:
        # Calculate absolute difference between current and previous frame
        frame_diff = cv2.absdiff(gray, prev_gray)
        _, motion_mask = cv2.threshold(frame_diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        # Dilate fills in small holes in the moving objects
        motion_kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv2.dilate(motion_mask, motion_kernel, iterations=2)

    prev_gray = gray.copy() # Update for next loop iteration

    # -----------------------------------------------------
    # B. YOLO PERSON TRACKER
    # Detecting players. 'persist=True' maintains ID tracking across frames.
    # -----------------------------------------------------
    results = model.track(frame, classes=[0], conf=0.3, persist=True, tracker="botsort.yaml", verbose=False)

    # Create an 'exclusion mask': We don't want to find the ball INSIDE a person
    # (prevents jerseys or shoes being mistaken for the ball).
    player_exclusion_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0])
            pad = 15 # Add a small buffer around the person
            px1 = max(0, px1 - pad)
            py1 = max(0, py1 - pad)
            px2 = min(frame.shape[1], px2 + pad)
            py2 = min(frame.shape[0], py2 + pad)
            cv2.rectangle(player_exclusion_mask, (px1, py1), (px2, py2), 255, -1)

    # -----------------------------------------------------
    # C. HSV BALL MASK
    # Logic: Filter the frame for the ball's color, then remove areas 
    # where players are standing or things aren't moving.
    # -----------------------------------------------------
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ball_mask = cv2.inRange(hsv_frame, ball_lower, ball_upper)
    
    # Clean up noise (Morphology): Removes tiny dots and fills small gaps
    kernel = np.ones((3, 3), np.uint8)
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)
    
    # Subtract the player boxes from the ball mask
    ball_mask = cv2.bitwise_and(ball_mask, cv2.bitwise_not(player_exclusion_mask))

    # Find the outlines (contours) of all yellow blobs left over
    b_cnts, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # -----------------------------------------------------
    # D. COLLECT & SCORE CANDIDATES
    # Since there might be multiple yellow moving objects, we score them
    # based on Circularity, Motion, and Proximity to the last ball position.
    # -----------------------------------------------------
    candidates = []

    for c in b_cnts:
        area = cv2.contourArea(c)
        if not (15 < area < 400): # Ball must be within a specific size range
            continue

        bx, by, bw, bh = cv2.boundingRect(c)

        # Filter 1: Aspect Ratio (A ball should be roughly square-shaped)
        aspect_ratio = float(bw) / bh
        if not (0.5 < aspect_ratio < 1.5):
            continue

        # Filter 2: Circularity (How close the shape is to a perfect circle)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.65: # 1.0 is a perfect circle
            continue

        # Filter 3: Motion (Does this yellow blob have movement inside it?)
        candidate_motion = motion_mask[by:by + bh, bx:bx + bw]
        moving_pixels = cv2.countNonZero(candidate_motion)
        if moving_pixels < MIN_MOTION_PIXELS:
            continue

        # Calculate Center
        cx, cy = bx + bw // 2, by + bh // 2

        # --- THE SCORING ENGINE ---
        # 1. More circular = more points
        circularity_score = circularity * 50
        # 2. More motion = more points
        motion_score = min(moving_pixels, 50)
        # 3. Proximity: If it's near where the ball was 0.03s ago, it's likely the ball
        if last_ball_pos is not None:
            dist = np.hypot(cx - last_ball_pos[0], cy - last_ball_pos[1])
            proximity_score = max(0, 50 - dist / 6)
        else:
            proximity_score = 25 

        total_score = circularity_score + motion_score + proximity_score
        candidates.append((total_score, cx, cy, bx, by, bw, bh))

    # -----------------------------------------------------
    # E. PICK THE SINGLE BEST CANDIDATE
    # -----------------------------------------------------
    if candidates:
        # Sort by total_score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0]
        _, cx, cy, bx, by, bw, bh = best

        last_ball_pos = (cx, cy) # Update tracker

        # Draw green box around the ball
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.putText(frame, "Ball", (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    # -----------------------------------------------------
    # F. CLASSIFY TEAMS
    # For every person YOLO found, decide which team they are on.
    # -----------------------------------------------------
    for result in results:
        if result.boxes is None: continue
            
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1) 
            
            # The 'feet' location is best for checking if someone is inside a zone
            feet_x, feet_y = int((x1 + x2) / 2), y2
            
            # Look only at the pixels inside the person's bounding box
            player_crop = frame[y1:y2, x1:x2]
            if player_crop.size == 0: continue
            hsv_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)
            
            # Count pixels matching Team 1 color vs Team 2 color
            t1_pixels = cv2.countNonZero(cv2.inRange(hsv_crop, team1_lower, team1_upper))
            t2_pixels = cv2.countNonZero(cv2.inRange(hsv_crop, team2_lower, team2_upper))
            
            min_color_pixels = 50 
            
            # Logic: If color matches AND feet are inside the assigned polygon zone
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

            # Draw visual feedback for players
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # -----------------------------------------------------
    # G. VISUAL DEBUGGING & UI
    # -----------------------------------------------------
    # Draw the team zones (polygons)
    cv2.polylines(frame, [team1_allowed_poly], isClosed=True, color=(0, 0, 255), thickness=1)
    cv2.polylines(frame, [team2_allowed_poly], isClosed=True, color=(255, 0, 0), thickness=1)

    # Display live counts
    cv2.putText(frame, f"Team 1 Players: {team1_players}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Team 2 Players: {team2_players}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Hybrid Volleyball Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()