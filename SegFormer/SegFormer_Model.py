import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
from PIL import Image
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import json

class PavementDataset(Dataset):
    def __init__(self, root, list_txt):
        self.root = root
        self.samples = []

        with open(list_txt, "r") as f:
            for line in f:
                img, mask = line.strip().split()
                self.samples.append((img, mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(os.path.join(self.root, img_path)).convert("RGB")
        mask = Image.open(os.path.join(self.root, mask_path)).convert("L")

        mask = np.array(mask)
        mask = (mask > 0).astype(np.int64)  # binary clean

        return image, mask


def collate_fn(batch, processor):
    images = [item[0] for item in batch]
    masks  = [item[1] for item in batch]

    inputs = processor(
        images=images,
        segmentation_maps=masks,
        return_tensors="pt"
    )

    return inputs

def compute_metrics(preds, labels):
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    return acc, prec, rec, f1

def train(model, processor, train_loader, test_loader, device, patience=5):
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    best_loss = float("inf")
    no_improve = 0
    history = []

    os.makedirs("./SegFormer/results", exist_ok=True)

    for epoch in range(1, 41):
        model.train()
        train_loss = 0
        all_preds, all_labels = [], []

        for inputs in train_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            labels = inputs["labels"].cpu()
            upsampled = torch.nn.functional.interpolate(
            outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            preds = upsampled.argmax(dim=1).detach().cpu()

            all_preds.append(preds)
            all_labels.append(labels)

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        tr_acc, tr_prec, tr_rec, tr_f1 = compute_metrics(preds, labels)

        model.eval()
        test_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs in test_loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs)
                test_loss += outputs.loss.item()

                labels = inputs["labels"].cpu()
                upsampled = torch.nn.functional.interpolate(
                outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                preds = upsampled.argmax(dim=1).detach().cpu()

                all_preds.append(preds)
                all_labels.append(labels)

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        te_acc, te_prec, te_rec, te_f1 = compute_metrics(preds, labels)

        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "test_loss": test_loss / len(test_loader),
            "test_acc": te_acc,
            "test_f1": te_f1
        }
        history.append(epoch_data)

        print(f"Epoch {epoch} | Acc: {te_acc:.4f} | F1: {te_f1:.4f}")

        if test_loss < best_loss:
            best_loss = test_loss
            no_improve = 0

            model.save_pretrained("./SegFormer/results/best_model")
            processor.save_pretrained("./SegFormer/results/best_model")
            print("Saved best model")
        else:
            no_improve += 1
            print(f"No improvement: {no_improve}")

        if no_improve >= patience:
            print("Early stopping triggered")
            break

        with open("./SegFormer/results/history.json", "w") as f:
            json.dump(history, f, indent=4)

    return history

def inference_SegFormer(model, processor, image, device, threshold=None):
    model.eval()

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False
    )

    probs = torch.softmax(logits, dim=1)

    if threshold is None:
        pred = probs.argmax(dim=1)
    else:
        pred = (probs[:, 1, :, :] > threshold)

    return pred.squeeze(0).cpu().numpy().astype(np.uint8)

def run_inference(model_dir, image_path, output_path, threshold=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = SegformerImageProcessor.from_pretrained(model_dir)
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir).to(device)

    image = Image.open(image_path).convert("RGB")

    pred_mask = inference_SegFormer(model, processor, image, device, threshold)

    mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mask_img.save(output_path)

    print(f"Saved → {output_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = "nvidia/mit-b0"

    processor = SegformerImageProcessor.from_pretrained(checkpoint)
    model = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint,
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(device)

    train_ds = PavementDataset("./CRACK500", "./CRACK500/train.txt")
    test_ds = PavementDataset("./CRACK500", "./CRACK500/test.txt")

    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, processor)
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=4,
        collate_fn=lambda x: collate_fn(x, processor)
    )

    train(model, processor, train_loader, test_loader, device)


if __name__ == "__main__":
    main()