import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split # Added random_split
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
    SAVE_PATH = 'weights/bisenet_v2_final_best.pth'
    
    os.makedirs('weights', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Data Preparation with Train-Val Split
    full_dataset = LabeledRoadDataset(root_dir=DATA_PATH)
    total_size = len(full_dataset)
    
    # 80% Training, 20% Validation
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Dataset loaded: {total_size} images total.")
    print(f"Training on: {len(train_dataset)} | Validating on: {len(val_dataset)}")

    # 3. Model, Optimizer, and Loss
    model = BiSeNet(num_classes=6, context_path='resnet18').to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0 # Track best Val accuracy, not Train

    # 4. Training & Validation Loop
    for epoch in range(NUM_EPOCHS):
        # --- TRAINING ---
        model.train()
        train_corrects = 0
        train_pixels = 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            main_out = outputs[0] if isinstance(outputs, tuple) else outputs
            
            loss = criterion(main_out, masks)
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(main_out, dim=1)
            train_corrects += torch.sum(preds == masks).item()
            train_pixels += torch.numel(masks)
            
            current_train_acc = (train_corrects / train_pixels) * 100
            loop.set_postfix(loss=loss.item(), acc=f"{current_train_acc:.2f}%")

        # --- VALIDATION ---
        model.eval()
        val_corrects = 0
        val_pixels = 0
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Val", leave=False)
            for images, masks in val_loop:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                main_out = outputs[0] if isinstance(outputs, tuple) else outputs
                
                preds = torch.argmax(main_out, dim=1)
                val_corrects += torch.sum(preds == masks).item()
                val_pixels += torch.numel(masks)

        # Epoch Summary
        epoch_train_acc = (train_corrects / train_pixels) * 100
        epoch_val_acc = (val_corrects / val_pixels) * 100
        
        print(f"Epoch {epoch+1} Summary: Train Acc: {epoch_train_acc:.2f}% | Val Acc: {epoch_val_acc:.2f}%")

        # Save Best Weights based on Validation Performance
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"*** New Best Val Acc: {best_val_acc:.2f}% - Weights Saved ***")

if __name__ == "__main__":
    train_model()
