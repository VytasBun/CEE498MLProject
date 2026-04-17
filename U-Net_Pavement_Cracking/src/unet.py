import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import os
from PIL import Image
from torchvision import transforms
import json
import numpy as np


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.bottleneck = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))
    
class PavementDataset(Dataset):
    def __init__(self, dataset_root, list_txt, transform=None):
        self.dataset_root = dataset_root
        self.transform = transform
        self.pairs = []

        with open(list_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_path = os.path.join(dataset_root, parts[0])
                    mask_path = os.path.join(dataset_root, parts[1])
                    self.pairs.append((img_path, mask_path))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        img_path, mask_path = self.pairs[index]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

def train(model, train_loader, test_loader, device, threshold=1e-5, patience=3):
    os.makedirs('./U-Net_Pavement_Cracking/results/', exist_ok=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    history = []
    no_improvement_epochs = 0
    best_test_loss = 1
    epoch = 0

    print(f"Starting Training on {device}...")

    while True:
        epoch += 1
        model.train()
        train_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'loss': 0.0}

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = (outputs > 0.5).float()
            train_metrics['loss'] += loss.item()
            train_metrics['tp'] += torch.sum((preds == 1) & (masks == 1)).item()
            train_metrics['tn'] += torch.sum((preds == 0) & (masks == 0)).item()
            train_metrics['fp'] += torch.sum((preds == 1) & (masks == 0)).item()
            train_metrics['fn'] += torch.sum((preds == 0) & (masks == 1)).item()

        model.eval()
        test_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'loss': 0.0}

        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)

                preds = (outputs > 0.5).float()
                test_metrics['loss'] += loss.item()
                test_metrics['tp'] += torch.sum((preds == 1) & (masks == 1)).item()
                test_metrics['tn'] += torch.sum((preds == 0) & (masks == 0)).item()
                test_metrics['fp'] += torch.sum((preds == 1) & (masks == 0)).item()
                test_metrics['fn'] += torch.sum((preds == 0) & (masks == 1)).item()

        def calc_stats(m, loader):
            total_px = m['tp'] + m['tn'] + m['fp'] + m['fn']
            acc = (m['tp'] + m['tn']) / (total_px + 1e-9)
            prec = m['tp'] / (m['tp'] + m['fp'] + 1e-9)
            rec = m['tp'] / (m['tp'] + m['fn'] + 1e-9)
            f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
            return acc, prec, rec, f1, m['loss'] / len(loader)

        tr_acc, tr_prec, tr_rec, tr_f1, tr_loss = calc_stats(train_metrics, train_loader)
        te_acc, te_prec, te_rec, te_f1, te_loss = calc_stats(test_metrics, test_loader)
        epoch_data = {
            'epoch': epoch,
            'train': {'loss': tr_loss, 'acc': tr_acc, 'f1': tr_f1, 'tp': train_metrics['tp'], 'tn': train_metrics['tn'], 'fp': train_metrics['fp'], 'fn': train_metrics['fn']},
            'test':  {'loss': te_loss, 'acc': te_acc, 'f1': te_f1, 'tp': test_metrics['tp'], 'tn': test_metrics['tn'], 'fp': test_metrics['fp'], 'fn': test_metrics['fn']}
        }
        history.append(epoch_data)

        print(f"Epoch {epoch} | Train Loss: {tr_loss:.4f} | Test Acc: {te_acc:.5f} | Test F1: {te_f1:.5f}")
        if te_loss < best_test_loss:
            best_test_loss = te_loss
            no_improvement_epochs = 0
            torch.save(model.state_dict(), 'best_unet_model.pth')
            print("Saved Model")
        else:
            no_improvement_epochs += 1
            print(f"No Improvements in {no_improvement_epochs} epochs")
        if no_improvement_epochs >= patience:
            print(f"Early stopping triggered")
            break
        with open('./U-Net_Pavement_Cracking/results/unet_analytics.json', 'w') as f:
            json.dump(history, f, indent=4)

        if epoch >= 40: break

    return history

def inference(model_path, image_path, save_path, device):
    model = UNet(in_channels=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((360, 640)), 
        transforms.ToTensor()
    ])

    input_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
    
    pred_mask = (output.squeeze() > 0.5).cpu().numpy().astype(np.uint8) * 255
    result_img = Image.fromarray(pred_mask)
    result_img.save(save_path)

if __name__ == "__main__":
    
    MODEL_PATH = './best_unet_model.pth' 
    INPUT_IMAGE = './CRACK500/testcrop/20160222_164141_1281_361.jpg' 
    OUTPUT_PATH = './U-Net_Pavement_Cracking/results/prediction_01.png'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference(model_path=MODEL_PATH, image_path=INPUT_IMAGE, save_path=OUTPUT_PATH, device=device)
    
    print(f"Success! Prediction saved to {OUTPUT_PATH}")