import numpy as np
from collections import deque
import time

hazard_buffer = deque(maxlen=5)
sb_buffer = deque(maxlen=5)
Kp = 0.005  
slowing_end_time = 0 
SLOW_DURATION_SECONDS = 3.0 

def get_control_commands(mask):
    global slowing_end_time
    
    h, w = mask.shape
    linear_v, angular_z = 0.22, 0.0
    status = "NAVIGATING"
    current_time = time.time()

    # 1. ROI & Detection (Keeping your logic)
    roi_top, roi_bottom = int(h * 0.5), int(h * 0.9)
    roi_left, roi_right = int(w * 0.15), int(w * 0.85)
    safety_zone = mask[roi_top:roi_bottom, roi_left:roi_right]
    
    blocking_pixels = np.sum((safety_zone != 3) & (safety_zone != 4))
    hazard_buffer.append(blocking_pixels)
    
    current_sb = (mask[int(h*0.5):int(h*0.8), :] == 5).sum()
    sb_buffer.append(current_sb)

    avg_blocked = sum(hazard_buffer) / len(hazard_buffer)
    avg_sb = sum(sb_buffer) / len(sb_buffer)
    hazard_ratio = avg_blocked / safety_zone.size

    # 2. IMPROVED STATE MACHINE
    
    # Update slowing counter based on SB detection
    if avg_sb > 600:
        slowing_end_time = current_time + SLOW_DURATION_SECONDS

    # Priority 1: Emergency Stop
    if hazard_ratio > 0.25:
        linear_v = 0.0
        status = "EMERGENCY STOP"
    
    # Priority 2: Slowing (Check if current time is still before the end time)
    elif current_time < slowing_end_time:
        linear_v = 0.11
        status = "SLOWING"
        
    # Priority 3: Normal Navigation
    else:
        linear_v = 0.22
        status = "NAVIGATING"

    # 3. YAW CONTROL (Keeping your logic)
    look_ahead_row = int(h * 0.75)
    road_pixels = np.where(mask[look_ahead_row, :] == 3)[0]
    target_x = np.mean(road_pixels) if len(road_pixels) > 0 else w // 2
    angular_z = -(target_x - (w / 2)) * Kp
    
    return linear_v, angular_z, int(target_x), look_ahead_row, status