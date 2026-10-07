import os
import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from model import SRCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SRCNN().to(device)
model.load_state_dict(torch.load("Results/srcnn.pth", map_location=device))
model.eval()

# Pick test image
test_images = glob.glob("Data/test/*.*")
if not test_images:
    print("Please add an image to Data/test/!")
    exit()

img_path = test_images[0]
full_img = Image.open(img_path).convert('YCbCr')
y, cb, cr = full_img.split()

# Prepare ground truth HR and LR input
w, h = (y.size[0] // 3) * 3, (y.size[1] // 3) * 3
hr_y = y.crop((0, 0, w, h))
lr_y = hr_y.resize((w // 3, h // 3), Image.BICUBIC)
bicubic_y = lr_y.resize((w, h), Image.BICUBIC)

# Predict with SRCNN
x = torch.from_numpy(np.array(bicubic_y, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
with torch.no_grad():
    pred_y = model(x).clamp(0.0, 1.0).squeeze().cpu().numpy()

# Calculate PSNR
hr_np = np.array(hr_y, dtype=np.float32) / 255.0
bicubic_np = np.array(bicubic_y, dtype=np.float32) / 255.0

psnr_bicubic = psnr(hr_np, bicubic_np, data_range=1.0)
psnr_srcnn = psnr(hr_np, pred_y, data_range=1.0)

print(f"Bicubic PSNR: {psnr_bicubic:.2f} dB")
print(f"SRCNN PSNR:   {psnr_srcnn:.2f} dB")

# Reconstruct RGB for visualization
pred_y_img = Image.fromarray(np.uint8(pred_y * 255))
out_img = Image.merge('YCbCr', [pred_y_img, cb.crop((0, 0, w, h)), cr.crop((0, 0, w, h))]).convert('RGB')
bicubic_rgb = Image.open(img_path).resize((w // 3, h // 3), Image.BICUBIC).resize((w, h), Image.BICUBIC)

# Save result plot to Results/
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title(f"Bicubic (PSNR: {psnr_bicubic:.2f}dB)")
plt.imshow(bicubic_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title(f"SRCNN (PSNR: {psnr_srcnn:.2f}dB)")
plt.imshow(out_img)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Original Ground Truth")
plt.imshow(full_img.convert('RGB').crop((0, 0, w, h)))
plt.axis("off")

plt.tight_layout()
plt.savefig("Results/comparison.png")
print("Saved comparison plot to Results/comparison.png")
plt.show()