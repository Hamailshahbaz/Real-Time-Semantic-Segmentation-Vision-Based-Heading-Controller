from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

class LabeledRoadDataset(Dataset):
    def __init__(self, root_dir, target_size=(360, 640)):
        self.root_dir = root_dir
        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg')])
        self.target_size = target_size 

        self.img_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Add this
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.NEAREST)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load Image
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img_file = Image.open(img_path).convert("RGB")

        # Find matching mask (e.g., frame_0000.jpg -> frame_0000_mask.png)
        mask_prefix = img_name.split('.')[0]
        mask_name = None
        for f in os.listdir(self.root_dir):
            if f.startswith(mask_prefix) and f.endswith('_mask.png'):
                mask_name = f
                break
        
        if mask_name is None:
            raise FileNotFoundError(f"Mask not found for {img_name}")

        mask_path = os.path.join(self.root_dir, mask_name)
        mask_file = Image.open(mask_path).convert("L") 

        # Apply transforms
        img_tensor = self.img_transform(img_file)
        mask_resized = self.mask_transform(mask_file)
        
        # Convert to numpy and sanitize indices
        mask_np = np.array(mask_resized)
      
        mask_tensor = torch.from_numpy(mask_np).long()

        return img_tensor, mask_tensor
