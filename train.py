import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchmetrics import JaccardIndex
from tqdm import tqdm
import os

# Import your modular components
from src.dataset import LabeledRoadDataset
from BiSeNet.model.build_BiSeNet import BiSeNet

def train_model():
    # 1. Hyperparameters & Configuration
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    DATA_PATH = '/kaggle/input/datasets/hamailshahbaz/labeled-frames/train'
    SAVE_PATH = 'weights/bisenet_best_miou.pth'
    
    os.makedirs('weights', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Data Preparation
    full_dataset = LabeledRoadDataset(root_dir=DATA_PATH)
    
    # Split into Train (90%) and Val (10%)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Dataset Split: {len(train_dataset)} Train | {len(val_dataset)} Val")

    # 3. Model, Optimizer, Loss, and Metrics
    model = BiSeNet(num_classes=6, context_path='resnet18').to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize mIoU Metric
    iou_metric = JaccardIndex(task="multiclass", num_classes=6).to(device)

    best_miou = 0.0

    # 4. Training & Validation Loop
    for epoch in range(NUM_EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss, train_corrects, train_pixels = 0.0, 0, 0
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train")
        
        for images, masks in train_loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            main_out = outputs[0] if isinstance(outputs, tuple) else outputs
            
            loss = criterion(main_out, masks)
            loss.backward()
            optimizer.step()
            
            # Quick Accuracy check for progress bar
            preds = torch.argmax(main_out, dim=1)
            train_corrects += torch.sum(preds == masks).item()
            train_pixels += torch.numel(masks)
            train_loss += loss.item()
            
            current_acc = (train_corrects / train_pixels) * 100
            train_loop.set_postfix(loss=loss.item(), acc=f"{current_acc:.2f}%")

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, val_corrects, val_pixels = 0.0, 0, 0
        iou_metric.reset() # Reset for each epoch's fresh evaluation
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Val", leave=False)
            for images, masks in val_loop:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                main_out = outputs[0] if isinstance(outputs, tuple) else outputs
                
                v_loss = criterion(main_out, masks)
                val_loss += v_loss.item()
                
                preds = torch.argmax(main_out, dim=1)
                val_corrects += torch.sum(preds == masks).item()
                val_pixels += torch.numel(masks)
                
                # Update mIoU Metric
                iou_metric.update(preds, masks)

        # 5. Epoch Summary & Saving
        epoch_train_acc = (train_corrects / train_pixels) * 100
        epoch_val_acc = (val_corrects / val_pixels) * 100
        epoch_miou = iou_metric.compute().item() * 100 
        
        print(f"\nSummary Epoch {epoch+1}:")
        print(f"  [Train] Acc: {epoch_train_acc:.2f}%")
        print(f"  [Val]   Acc: {epoch_val_acc:.2f}% | mIoU: {epoch_miou:.2f}%")

        # Save Best Model based on mIoU
        if epoch_miou > best_miou:
            best_miou = epoch_miou
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"*** New Best mIoU: {best_miou:.2f}% - Weights Saved to {SAVE_PATH} ***")

if __name__ == "__main__":
    train_model()