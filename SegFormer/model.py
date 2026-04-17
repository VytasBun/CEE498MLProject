import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import os
from PIL import Image
from torchvision import transforms
import json
import numpy as np
from transformers import SegformerImageProcessor
from transformers import SegformerForSemanticSegmentation
    
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
    os.makedirs('./SegFormer/results/', exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

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
            images = images.to(device)
            masks = (masks * 255).to(device).squeeze(1).long()
            masks = torch.where(masks > 0, 1, 0)

            outputs = model(images)
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(upsampled_logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = upsampled_logits.argmax(dim=1)
            train_metrics['loss'] += loss.item()
            train_metrics['tp'] += torch.sum((preds == 1) & (masks == 1)).item()
            train_metrics['tn'] += torch.sum((preds == 0) & (masks == 0)).item()
            train_metrics['fp'] += torch.sum((preds == 1) & (masks == 0)).item()
            train_metrics['fn'] += torch.sum((preds == 0) & (masks == 1)).item()

        model.eval()
        test_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'loss': 0.0}

        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), (masks * 255).to(device).squeeze(1).long()
                outputs = model(images)
                masks = torch.where(masks > 0, 1, 0)
                upsampled_logits = nn.functional.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(upsampled_logits, masks)

                preds = upsampled_logits.argmax(dim=1)
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
            model.save_pretrained('./SegFormer/results/best_model')
            processor.save_pretrained('./SegFormer/results/best_model')
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


def inference(model, image_path, processor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Upscale logits to original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred_mask = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy()
    
    return pred_mask

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_DIR = './SegFormer/results/best_model'
    INPUT_IMG = 'CRACK500/testcrop/20160222_080933_361_641.jpg'
    OUTPUT_IMG = './SegFormer/results/prediction_mask.png'
    processor = SegformerImageProcessor.from_pretrained(MODEL_DIR)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    pred_mask = inference(model, INPUT_IMG, processor, device)
    mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))

    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    mask_img.save(OUTPUT_IMG)