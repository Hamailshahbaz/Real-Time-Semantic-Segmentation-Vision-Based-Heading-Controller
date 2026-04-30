import torch
import torch.optim as optim
from torch.utils.data import DataLoader
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

    # 2. Data Preparation
    dataset = LabeledRoadDataset(root_dir=DATA_PATH)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset loaded: {len(dataset)} images found.")

    # 3. Model, Optimizer, and Loss
    # Using 6 classes to account for 0-5 (including speed breaker)
    model = BiSeNet(num_classes=6, context_path='resnet18').to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0

    # 4. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_pixels = 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            # BiSeNet often returns a tuple (main_out, feat_out1, feat_out2)
            main_out = outputs[0] if isinstance(outputs, tuple) else outputs
            
            loss = criterion(main_out, masks)
            loss.backward()
            optimizer.step()
            
            # Accuracy Calculation
            preds = torch.argmax(main_out, dim=1)
            running_corrects += torch.sum(preds == masks).item()
            total_pixels += torch.numel(masks)
            running_loss += loss.item()
            
            current_acc = (running_corrects / total_pixels) * 100
            loop.set_postfix(loss=loss.item(), acc=f"{current_acc:.2f}%")

        # Epoch Summary
        epoch_acc = (running_corrects / total_pixels) * 100
        print(f"Epoch {epoch+1} Complete. Accuracy: {epoch_acc:.2f}%")

        # Save Best Weights
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved new best weights with {best_acc:.2f}% accuracy.")

if __name__ == "__main__":
    train_model()
