import numpy as np
from collections import deque
import time

hazard_buffer = deque(maxlen=5)
sb_buffer = deque(maxlen=5)
Kp = 0.005  

slowing_end_time = 0 
WAIT_DURATION = 4.0 

def get_control_commands(mask):
    global slowing_end_time
    
    h, w = mask.shape
    linear_v, angular_z = 0.22, 0.0
    status = "NAVIGATING"
    current_time = time.time() # Current system time in seconds

    # 1. ROI & DETECTION LOGIC
    roi_top, roi_bottom = int(h * 0.5), int(h * 0.9)
    roi_left, roi_right = int(w * 0.15), int(w * 0.85)
    safety_zone = mask[roi_top:roi_bottom, roi_left:roi_right]
    
    # Identify non-road hazards and speed breakers
    blocking_pixels = np.sum((safety_zone != 3) & (safety_zone != 4))
    hazard_buffer.append(blocking_pixels)
    
    current_sb = (mask[int(h*0.5):int(h*0.8), :] == 5).sum()
    sb_buffer.append(current_sb)

    # Calculate ratios for decision making
    avg_blocked = sum(hazard_buffer) / len(hazard_buffer)
    avg_sb = sum(sb_buffer) / len(sb_buffer)
    hazard_ratio = avg_blocked / safety_zone.size


    # Priority 1: Emergency Stop (Always overrides slowing)
    if hazard_ratio > 0.25:
        linear_v = 0.0
        status = "EMERGENCY STOP"
    
    # Priority 2: Slowing (Checks if we are still within the 4-second window)
    elif current_time < slowing_end_time:
        linear_v = 0.11
        status = "SLOWING"
        
    # Priority 3: Normal Navigation
    else:
        linear_v = 0.22
        status = "NAVIGATING"

    # --- 3. YAW CONTROL ---
    look_ahead_row = int(h * 0.75)
    road_pixels = np.where(mask[look_ahead_row, :] == 3)[0]
    target_x = np.mean(road_pixels) if len(road_pixels) > 0 else w // 2
    angular_z = -(target_x - (w / 2)) * Kp
    
    return linear_v, angular_z, int(target_x), look_ahead_row, status