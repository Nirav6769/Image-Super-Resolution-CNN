import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from model import SRCNN

# 1. Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 2. Collect images
train_images = glob.glob("Data/train/*.*")
if not train_images:
    print("Please add image files into Data/train/ before running!")
    exit()

def get_y_channel(img_path):
    img = Image.open(img_path).convert('YCbCr')
    y, _, _ = img.split()
    return y

print(f"Training on {len(train_images)} images using {device}...")

epochs = 15
for epoch in range(epochs):
    epoch_loss = 0.0
    for img_path in train_images:
        y_img = get_y_channel(img_path)
        w, h = y_img.size
        # Crop slightly to keep dimensions clean
        w, h = (w // 3) * 3, (h // 3) * 3
        hr = y_img.crop((0, 0, w, h))
        
        # Create low-res and bicubic-upsampled image
        lr = hr.resize((w // 3, h // 3), Image.BICUBIC)
        bicubic = lr.resize((w, h), Image.BICUBIC)
        
        # Convert to tensors
        x = torch.from_numpy(np.array(bicubic, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        y = torch.from_numpy(np.array(hr, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_images):.6f}")

# 3. Save weights in Results/
os.makedirs("Results", exist_ok=True)
torch.save(model.state_dict(), "Results/srcnn.pth")
print("Training complete! Model saved to Results/srcnn.pth")