from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

class LabeledRoadDataset(Dataset):
    def __init__(self, root_dir, target_size=(288, 512)):
        self.root_dir = root_dir
        # Exactly as her: Identify images first
        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg') and '_mask' not in f])
        self.target_size = target_size 

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base = os.path.splitext(img_name)[0]  # Get the 'base' name
        mask_name = base + "_mask.png"        # Append '_mask.png' to find the pair

        img_path = os.path.join(self.root_dir, img_name)
        mask_path = os.path.join(self.root_dir, mask_name)

        # Skip/Error if missing pair:
        if not os.path.exists(mask_path):
            # If the specific augmented mask doesn't exist, try the base version
            # This handles Roboflow cases where frame_001_aug.jpg uses frame_001_mask.png
            base_prefix = base.split('_')[0]
            mask_name = base_prefix + "_mask.png"
            mask_path = os.path.join(self.root_dir, mask_name)
            
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing mask for: {img_name}")

        # READ IMAGE
        img_file = Image.open(img_path).convert("RGB")
        # READ MASK
        mask_file = Image.open(mask_path).convert("L") 

        # PRECISION FLIP
        # Only flip the mask if the image itself is a flipped version
        if "flip" in img_name.lower():
            mask_file = mask_file.transpose(Image.FLIP_LEFT_RIGHT)

        # RESIZE 
        img_resized = img_file.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        mask_resized = mask_file.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)

        # Final conversion to Tensors
        img_tensor = self.img_transform(img_resized)
        mask_np = np.array(mask_resized)
        
        # Clean labels to stay within [0, 5] range for BiSeNet
        mask_np[mask_np > 5] = 0 
        mask_tensor = torch.from_numpy(mask_np).long()

        return img_tensor, mask_tensor