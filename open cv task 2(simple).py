import cv2 
import numpy as np 
from ultralytics import YOLO 

model = YOLO('yolov8n.pt') 

video_path = '/Users/satviksingh/Documents/code/ manas_git/GitHub/satvik_VII/working_directory/neural net problems/opencvtask/Volleyball.mp4'
cap = cv2.VideoCapture(video_path)

# ====================================================================
# VIDEO WRITER SETUP (SAVE OUTPUT TO FILE)
# ====================================================================
# Get the framerate of the original video so the output plays at normal speed
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: 
    fps = 30 # Fallback just in case

# Define the codec and create VideoWriter object. 
# We use 'mp4v' for standard .mp4, and MUST match the 1000x600 size we resize to below.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = '/Users/satviksingh/Documents/code/ manas_git/GitHub/satvik_VII/working_directory/neural net problems/opencvtask/Tracked_Output.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (1000, 600))

# HSV MASK TO INDENTIFY BALL
ball_lower = np.array([6, 80, 137])
ball_upper = np.array([66, 185, 255])

# DEFINING ZONE TO INDENTIFY TEAMS 
team1_allowed_poly = np.array([
    [171, 372],
    [819, 370],
    [916, 579],
    [78, 584],
], np.int32)

team2_allowed_poly = np.array([
    [171, 370],
    [818, 366],
    [760, 236],
    [227, 235],
], np.int32)

# MOTION DETECTION SETUP 
prev_gray = None
MOTION_THRESHOLD = 25
MIN_MOTION_PIXELS = 8
last_ball_pos = None   

# --- BALL TRAIL SETUP ---
ball_trail = []       
MAX_TRAIL_AGE = 20    

# ====================================================================
# MAIN PROCESSING LOOP
# ====================================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1000, 600))
    
    team1_players = 0
    team2_players = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------------------
    # A. MOTION MASK
    # -----------------------------------------------------
    if prev_gray is None:
        motion_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    else:
        frame_diff = cv2.absdiff(gray, prev_gray)
        _, motion_mask = cv2.threshold(frame_diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        motion_kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv2.dilate(motion_mask, motion_kernel, iterations=2)

    prev_gray = gray.copy()

    # -----------------------------------------------------
    # B. YOLO PERSON TRACKER
    # -----------------------------------------------------
    results = model.track(frame, classes=[0], conf=0.3, persist=True, tracker="botsort.yaml", verbose=False)

    player_exclusion_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0])
            pad = 15
            px1 = max(0, px1 - pad)
            py1 = max(0, py1 - pad)
            px2 = min(frame.shape[1], px2 + pad)
            py2 = min(frame.shape[0], py2 + pad)
            cv2.rectangle(player_exclusion_mask, (px1, py1), (px2, py2), 255, -1)

    # -----------------------------------------------------
    # C. HSV BALL MASK
    # ----------------------------------------------------- 
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
    ball_mask = cv2.inRange(hsv_frame, ball_lower, ball_upper)
    
    kernel = np.ones((3, 3), np.uint8)
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)
    ball_mask = cv2.bitwise_and(ball_mask, cv2.bitwise_not(player_exclusion_mask))

    b_cnts, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # -----------------------------------------------------
    # D. COLLECT ALL CANDIDATES THAT PASS FILTERS
    # -----------------------------------------------------
    candidates = []

    for c in b_cnts:
        area = cv2.contourArea(c)
        if not (15 < area < 400):
            continue

        bx, by, bw, bh = cv2.boundingRect(c)

        aspect_ratio = float(bw) / bh
        if not (0.5 < aspect_ratio < 1.5):
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.65:
            continue

        candidate_motion = motion_mask[by:by + bh, bx:bx + bw]
        moving_pixels = cv2.countNonZero(candidate_motion)
        if moving_pixels < MIN_MOTION_PIXELS:
            continue

        cx = bx + bw // 2
        cy = by + bh // 2

        circularity_score = circularity * 50
        motion_score = min(moving_pixels, 50) 

        if last_ball_pos is not None:
            dist = np.hypot(cx - last_ball_pos[0], cy - last_ball_pos[1])
            proximity_score = max(0, 50 - dist / 6)
        else:
            proximity_score = 25 

        total_score = circularity_score + motion_score + proximity_score
        candidates.append((total_score, cx, cy, bx, by, bw, bh))

    # -----------------------------------------------------
    # E. PICK CANDIDATE & DRAW FADING TRAIL
    # -----------------------------------------------------
    for pt in ball_trail:
        pt['age'] += 1
    
    ball_trail = [pt for pt in ball_trail if pt['age'] < MAX_TRAIL_AGE]

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0]
        _, cx, cy, bx, by, bw, bh = best

        last_ball_pos = (cx, cy)
        
        radius = int(max(bw, bh) / 1.5)
        
        ball_trail.append({'pos': (cx, cy), 'radius': radius, 'age': 0})
        
        cv2.putText(frame, "Ball", (bx, by - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for pt in ball_trail:
        curr_cx, curr_cy = pt['pos']
        r = pt['radius']
        age = pt['age']
        
        fade_factor = 1.0 - (age / MAX_TRAIL_AGE)
        intensity = int(255 * fade_factor)
        color = (0, intensity, 0)
        
        cv2.circle(frame, (curr_cx, curr_cy), r, color, 2)
        
        if age == 0:
            cv2.circle(frame, (curr_cx, curr_cy), 3, (0, 255, 0), -1)

    # -----------------------------------------------------
    # F. CLASSIFY TEAMS (PURE GEOFENCING)
    # -----------------------------------------------------
    for result in results:
        if result.boxes is None:
            continue
            
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1) 
            
            feet_x = int((x1 + x2) / 2)
            feet_y = y2
            
            in_team1 = cv2.pointPolygonTest(team1_allowed_poly, (feet_x, feet_y), False) >= 0
            in_team2 = cv2.pointPolygonTest(team2_allowed_poly, (feet_x, feet_y), False) >= 0
            
            if in_team1:
                team1_players += 1
                color = (255, 0, 0) 
                label = "T1"
            elif in_team2:
                team2_players += 1
                color = (0, 255, 255) 
                label = "T2"
            else:
                color = (20, 20, 20) 
                label = "Out"

            cv2.circle(frame, (feet_x, feet_y), 3, (0, 0, 0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # -----------------------------------------------------
    # G. VISUAL DEBUGGING & UI & SAVING
    # -----------------------------------------------------
    cv2.polylines(frame, [team1_allowed_poly], isClosed=True, color=(255, 0, 0), thickness=1)
    cv2.polylines(frame, [team2_allowed_poly], isClosed=True, color=(0, 255, 255), thickness=1)

    cv2.putText(frame, f"Team 1 Players: {team1_players}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f"Team 2 Players: {team2_players}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # >>> WRITE THE PROCESSED FRAME TO OUR OUTPUT VIDEO <<<
    out.write(frame)

    cv2.imshow("Hybrid Volleyball Tracker", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up both the input reader and output writer
cap.release()
out.release()
cv2.destroyAllWindows()
