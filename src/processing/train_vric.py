import argparse
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Add project root to python path to allow imports from src
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.processing.benchmark_vric import VRICDataset, extract_features_loader, eval_query

class VRICTrainDataset(Dataset):
    """
    Dataset class for training on VRIC images with vehicle ID classification.
    """
    def __init__(self, img_dir, file_list_path, transform=None, subset_classes=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.samples = []
        
        if not file_list_path.exists():
            raise FileNotFoundError(f"Train label file not found: {file_list_path}")
            
        all_samples = []
        with open(file_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    img_name, vehicle_id = parts[0], int(parts[1])
                    all_samples.append((img_name, vehicle_id))
                    
        # Filter classes if a subset is specified (for quick CPU training)
        if subset_classes and subset_classes > 0:
            print(f"Filtering dataset to the top {subset_classes} most frequent vehicle classes...")
            counter = Counter([s[1] for s in all_samples])
            top_classes = [item[0] for item in counter.most_common(subset_classes)]
            top_classes_set = set(top_classes)
            all_samples = [s for s in all_samples if s[1] in top_classes_set]
            
        # Map vehicle_ids to continuous labels [0, num_classes-1]
        unique_ids = sorted(list(set([s[1] for s in all_samples])))
        self.id_to_label = {vid: label for label, vid in enumerate(unique_ids)}
        self.num_classes = len(unique_ids)
        
        for img_name, vid in all_samples:
            self.samples.append((img_name, self.id_to_label[vid]))
            
        print(f"Dataset prepared: {len(self.samples)} images across {self.num_classes} unique vehicles.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = self.img_dir / img_name
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

class ReIDModel(nn.Module):
    """
    ResNet-50 with a custom FC classification head.
    Outputs both L2-normalized features (for Re-ID retrieval) and classification logits.
    """
    def __init__(self, num_classes):
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights
            self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        except ImportError:
            self.backbone = models.resnet50(pretrained=True)
            
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove default classifier
        
        # Classification layer
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        
        # L2 normalize features
        norm = torch.norm(features, p=2, dim=1, keepdim=True)
        normalized_features = features / torch.clamp(norm, min=1e-12)
        
        # Classification logits
        logits = self.fc(features)
        
        return normalized_features, logits

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-50 on the VRIC Dataset for Vehicle Re-ID")
    parser.add_argument("--data_dir", type=str, default="data/vric", help="Path to VRIC dataset directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--subset", type=int, default=100, help="Number of classes to train on (default: 100 for fast CPU demo)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    vric_dir = Path(args.data_dir).resolve()
    if not vric_dir.exists():
        print(f"Error: Dataset directory {vric_dir} does not exist. Please run the download script first.")
        sys.exit(1)

    # Preprocessing transforms (adding data augmentation for training)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Standard evaluation transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("Loading Training Dataset...")
    train_dataset = VRICTrainDataset(
        img_dir=vric_dir / "train_images",
        file_list_path=vric_dir / "vric_train.txt",
        transform=train_transform,
        subset_classes=args.subset
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Instantiate model
    model = ReIDModel(num_classes=train_dataset.num_classes)
    
    # Freeze lower backbone layers to speed up training on CPU and prevent overfitting
    print("Freezing lower ResNet layers (keeping layer4 and FC head trainable)...")
    for name, param in model.backbone.named_parameters():
        if "layer4" not in name:
            param.requires_grad = False
            
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # Optimize only parameters that require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for imgs, labels in progress_bar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            _, logits = model(imgs)
            loss = criterion(logits, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            
            # Calculate accuracy
            _, preds = torch.max(logits, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_predictions += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{(correct_predictions.double() / total_predictions * 100):.2f}%"
            })
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = (correct_predictions.double() / len(train_dataset)) * 100
        print(f"Epoch {epoch} complete - Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Save fine-tuned checkpoint
    checkpoint_path = vric_dir / "veri_resnet50_finetuned.pth"
    print(f"Saving fine-tuned model checkpoint to {checkpoint_path}...")
    torch.save(model.state_dict(), checkpoint_path)

    # Quick evaluation on the Re-ID test sets to show improvement
    print("\nRunning Re-ID evaluation with the fine-tuned model...")
    model.eval()
    
    probe_dataset = VRICDataset(
        img_dir=vric_dir / "probe_images",
        file_list_path=vric_dir / "vric_probe.txt",
        transform=val_transform
    )
    probe_loader = DataLoader(probe_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    gallery_dataset = VRICDataset(
        img_dir=vric_dir / "gallery_images",
        file_list_path=vric_dir / "vric_gallery.txt",
        transform=val_transform
    )
    gallery_loader = DataLoader(gallery_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Temporary wrapper to extract using model's backbone
    class TempExtractor:
        def __init__(self, m, dev):
            self.model = m.backbone
            self.device = dev
            
    temp_extractor = TempExtractor(model, device)
    
    print("Extracting test features...")
    q_feats, q_pids, q_cids, _ = extract_features_loader(probe_loader, temp_extractor)
    g_feats, g_pids, g_cids, _ = extract_features_loader(gallery_loader, temp_extractor)

    print("Evaluating Re-ID metrics (mAP, Rank-1, Rank-5)...")
    all_ap = []
    all_cmc = []
    
    for idx in range(len(q_feats)):
        q_feat = q_feats[idx]
        q_pid = q_pids[idx]
        q_cid = q_cids[idx]
        
        ap, cmc = eval_query(q_feat, q_pid, q_cid, g_feats, g_pids, g_cids)
        all_ap.append(ap)
        all_cmc.append(cmc)
        
    mAP = np.mean(all_ap)
    cmc_avg = np.mean(np.array(all_cmc), axis=0)
    
    print("\n" + "="*45)
    print("      FINE-TUNED MODEL VRIC BENCHMARK RESULTS")
    print("="*45)
    print(f"Mean Average Precision (mAP): {mAP * 100:.2f}%")
    print(f"Rank-1 Accuracy:             {cmc_avg[0] * 100:.2f}%")
    print(f"Rank-5 Accuracy:             {cmc_avg[4] * 100:.2f}%")
    print(f"Rank-10 Accuracy:            {cmc_avg[9] * 100:.2f}%")
    print("="*45)
    
if __name__ == "__main__":
    main()
