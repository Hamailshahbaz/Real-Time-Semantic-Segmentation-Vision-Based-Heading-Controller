import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from src.utils import load_calibration, undistort_frame
from src.navigation import get_control_commands
from BiSeNet.model.build_BiSeNet import BiSeNet

# --- 1. SETUP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load K and D using your modular utility
K, D = load_calibration('data/camera_calibration.npz')

# Define the transform (Must match training exactly)
target_size = (360, 640)
transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 2. LOAD MODEL ---
model = BiSeNet(num_classes=6, context_path='resnet18').to(device)
model.load_state_dict(torch.load('weights/bisenet_v2_final_best.pth', map_location=device))
model.eval()

# --- 3. RUN VIDEO ---
video_path = 'data/t2.mp4' # Update this to your local or relative path
cap = cv2.VideoCapture(video_path)

# Prepare Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('final_navigation_demo.mp4', fourcc, 30.0, (1280, 360))

print("Starting Video Generation...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    # Step A: Undistort (Using your modular function)
    undistorted = undistort_frame(frame, K, D)
    
    # Step B: Segment
    # Convert OpenCV BGR to PIL RGB for the transformer
    pil_img = Image.fromarray(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        # Handle BiSeNet auxiliary heads
        preds = torch.argmax(output[0] if isinstance(output, tuple) else output, dim=1)
        preds = preds.squeeze(0).cpu().numpy()

    # Step C: Get Control Commands
    v, w, tx, ty, status = get_control_commands(preds)

    # Step D: Visuals
    # Create an empty RGB mask
    mask_color = np.zeros((360, 640, 3), dtype=np.uint8)
    mask_color[preds == 3] = [255, 0, 0]    # Road -> Red
    mask_color[preds == 1] = [255, 255, 0]  # Human -> Yellow
    mask_color[preds == 2] = [128, 0, 128]  # Obstacle -> Purple
    mask_color[preds == 5] = [0, 255, 0]    # Speed Breaker -> Green
    mask_color[preds == 4] = [0, 0, 255]    # Sidewalk -> Blue
    
    # Resize raw frame to match mask size for hstack
    resized_rgb = cv2.resize(undistorted, (640, 360))
    
    # Logic-based text color
    if v == 0:
        text_color = (0, 0, 255)   # BGR Red
    elif v == 0.11:
        text_color = (0, 255, 255) # BGR Yellow
    else:
        text_color = (0, 255, 0)   # BGR Green

    # UI Overlays
    cv2.line(resized_rgb, (320, 360), (tx, ty), (0, 255, 0), 3)
    cv2.putText(resized_rgb, f"V: {v:.2f} W: {w:.2f}", (30, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    cv2.putText(resized_rgb, status, (30, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # Side-by-side View
    combined = np.hstack((resized_rgb, mask_color))
    out.write(combined)

cap.release()
out.release()
print("Success! final_navigation_demo.mp4 has been created.")
