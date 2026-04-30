import numpy as np

def get_control_commands(mask, Kp=0.005):
    h, w = mask.shape
    linear_v, angular_z = 0.22, 0.0
    
    # --- 1. Define the ROI (The "Safety Corridor") ---
    # We only care about obstacles in the bottom 60% of the image 
    # and the middle 40% of the width (directly in front of us).
    roi_top, roi_bottom = int(h * 0.6), h
    roi_left, roi_right = int(w * 0.3), int(w * 0.7)
    
    safety_zone = mask[roi_top:roi_bottom, roi_left:roi_right]
    
    # --- 2. Check for Obstacles ONLY in the Safety Zone ---
    # 1: Human, 2: Obstacle
    hazard_pixels = np.sum((safety_zone == 1) | (safety_zone == 2))
    
    if hazard_pixels > 1000: # Threshold for objects directly in front
        linear_v = 0.0
        status = "EMERGENCY STOP (Path Blocked)"
    else:
        # --- 3. Speed Breaker Check (Can be wider/higher than the hazard zone) ---
        if (mask[int(h*0.5):, :] == 5).sum() > 200:
            linear_v = 0.11
            status = "SLOWING"
        else:
            status = "NAVIGATING"

    # --- 4. Yaw Control (Unchanged) ---
    road_pixels = np.where(mask[int(h * 0.75), :] == 3)[0]
    target_x = np.mean(road_pixels) if len(road_pixels) > 0 else w // 2
    angular_z = -(target_x - (w / 2)) * Kp
    
    return linear_v, angular_z, int(target_x), int(h * 0.75), status
