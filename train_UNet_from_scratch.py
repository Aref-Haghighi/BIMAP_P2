import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
import matplotlib.pyplot as plt
from czifile import imread
from read_roi import read_roi_zip
from skimage.draw import polygon2mask
from skimage import measure
import cv2
import csv

# ------------ Settings ------------
czi_files = [
    "WT_NADA_RADA_HADA_NHS_40min_ROI1_SIM.czi",
    "WT_NADA_RADA_HADA_NHS_40min_ROI2_SIM.czi",
    "WT_NADA_RADA_HADA_NHS_40min_ROI3_SIM.czi",
]
roi_files = [
    "RoiSet_Contour_bacteria_ROI1.zip",
    "RoiSet_Contour_bacteria_ROI2.zip",
    "RoiSet_Contour_bacteria_ROI3.zip",
]

# ------------ Helper Functions ------------
def czi_to_rgb(filename):
    """Convert .czi to RGB normalized image"""
    raw = imread(filename)
    img = np.squeeze(raw)[0:3, :, :]  # first 3 channels
    def normalize(ch): return (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
    rgb = np.stack([normalize(img[2]), normalize(img[1]), normalize(img[0])], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    return rgb

def roi_to_mask(filename, shape):
    """Convert ROI zip to binary mask"""
    rois = read_roi_zip(filename)
    mask = np.zeros(shape, dtype=np.uint8)
    for roi in rois.values():
        x, y = roi['x'], roi['y']
        poly_mask = polygon2mask(shape, np.column_stack((y, x)))
        mask[poly_mask] = 1
    return mask

# ------------ Dataset & Augmentation ------------
class BacteriaDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        # To tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).long()
        return image, mask

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3)
])

# ------------ Prepare Data ------------
images, masks = [], []
for czi, roi in zip(czi_files, roi_files):
    img = czi_to_rgb(czi)
    mask = roi_to_mask(roi, img.shape[:2])
    images.append(img)
    masks.append(mask)

dataset = BacteriaDataset(images, masks, transform=augment)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# ------------ U-Net Model ------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_classes=2):
        super().__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.enc1 = CBR(in_channels, 32)
        self.enc2 = CBR(32, 64)
        self.enc3 = CBR(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = CBR(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = CBR(64, 32)
        self.head = nn.Conv2d(32, out_classes, 1)
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.bottleneck(self.pool(x3))
        y3 = self.dec2(torch.cat([self.up2(x4), x3], dim=1))
        y2 = self.dec1(torch.cat([self.up1(y3), x2], dim=1))
        y1 = self.dec0(torch.cat([self.up0(y2), x1], dim=1))
        return self.head(y1)

# ------------ Training ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

num_epochs = 50
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        loss = criterion(logits, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "bacteria_unet.pth")

# ------------ Inference & Visualization with Borders ------------
model.eval()

def draw_borders(image, mask, color):
    """Draws colored border (R, G, B) for given binary mask."""
    borders = measure.find_contours(mask, 0.5)
    bordered_img = image.copy()
    for contour in borders:
        contour = np.round(contour).astype(int)
        for y, x in contour:
            if 0 <= y < bordered_img.shape[0] and 0 <= x < bordered_img.shape[1]:
                bordered_img[y, x] = color
    return bordered_img

def infer_and_visualize(image, mask, model, device, out_prefix):
    img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    # ----- Metrics -----
    intersection = np.logical_and(pred_mask, mask).sum()
    union = np.logical_or(pred_mask, mask).sum()
    dice = 2 * intersection / (pred_mask.sum() + mask.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    print(f"[{out_prefix}] Dice: {dice:.3f} | IoU: {iou:.3f}")

    # ----- Save metrics to CSV -----
    csv_file = "metrics_results.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Image", "Dice", "IoU"])
        writer.writerow([out_prefix, dice, iou])

    # ----- Border Visualization -----
    bordered = image.copy()
    bordered = draw_borders(bordered, pred_mask, (255, 255, 0))  # yellow border = prediction
    bordered = draw_borders(bordered, mask, (0, 0, 255))         # blue border = ground truth

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1); plt.imshow(image); plt.title('Original'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(pred_mask, cmap='gray'); plt.title('Predicted'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(bordered); plt.title('Borders (Yellow=Pred, Blue=GT)'); plt.axis('off')
    plt.tight_layout(); plt.show()

    cv2.imwrite(f"{out_prefix}_borders.png", cv2.cvtColor(bordered, cv2.COLOR_RGB2BGR))
    return pred_mask

# ------------ Visualize on Training Images ------------
for idx in range(len(images)):
    infer_and_visualize(images[idx], masks[idx], model, device, f"train_{idx+1}")
