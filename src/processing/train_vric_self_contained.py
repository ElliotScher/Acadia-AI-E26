import os
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

# ==========================================
# 1. DATASET DEFINITIONS
# ==========================================

class VRICDataset(Dataset):
    """
    Dataset class for evaluation sets (Probe and Gallery).
    """
    def __init__(self, img_dir, file_list_path, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.samples = []
        with open(file_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    img_name, vehicle_id, cam_id = parts[0], int(parts[1]), int(parts[2])
                    self.samples.append((img_name, vehicle_id, cam_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, vehicle_id, cam_id = self.samples[idx]
        img_path = self.img_dir / img_name
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, vehicle_id, cam_id, img_name


class VRICTrainDataset(Dataset):
    """
    Dataset class for training classification.
    """
    def __init__(self, img_dir, file_list_path, transform=None, subset_classes=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.samples = []
        
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
                    
        if subset_classes and subset_classes > 0:
            print(f"Filtering training set to top {subset_classes} vehicle classes...")
            counter = Counter([s[1] for s in all_samples])
            top_classes = [item[0] for item in counter.most_common(subset_classes)]
            top_classes_set = set(top_classes)
            all_samples = [s for s in all_samples if s[1] in top_classes_set]
            
        unique_ids = sorted(list(set([s[1] for s in all_samples])))
        self.id_to_label = {vid: label for label, vid in enumerate(unique_ids)}
        self.num_classes = len(unique_ids)
        
        for img_name, vid in all_samples:
            self.samples.append((img_name, self.id_to_label[vid]))
            
        print(f"Loaded {len(self.samples)} train images across {self.num_classes} vehicle classes.")

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

# ==========================================
# 2. MODEL DEFINITIONS
# ==========================================

class ReIDModel(nn.Module):
    """
    ResNet-50 model with custom FC classification head.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        norm = torch.norm(features, p=2, dim=1, keepdim=True)
        normalized_features = features / torch.clamp(norm, min=1e-12)
        logits = self.fc(features)
        return normalized_features, logits

class TripletLoss(nn.Module):
    """
    Triplet loss with batch hard mining.
    Reference: Hermans et al. "In Defense of the Triplet Loss for Person Re-Identification".
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance matrix
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # Find positives and negatives mask
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        
        # Hardest positive: maximum distance to sample with same label
        dist_ap, _ = torch.max(dist * mask.float(), dim=1)
        
        # Hardest negative: minimum distance to sample with different label
        dist_an, _ = torch.min(dist + 1e5 * mask.float(), dim=1)
        
        # Compute loss: max(0, dist_ap - dist_an + margin)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class RandomIdentitySampler(torch.utils.data.sampler.Sampler):
    """
    Randomly samples N identities, then K instances for each identity.
    Total batch size = N * K.
    Guarantees sufficient positive samples in each batch for triplet mining.
    """
    def __init__(self, data_source, batch_size, num_instances=4):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        
        # Group image indices by vehicle ID
        self.index_dic = {}
        for index, (_, label) in enumerate(self.data_source.samples):
            if label not in self.index_dic:
                self.index_dic[label] = []
            self.index_dic[label].append(index)
            
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = []
        import random
        # We sample indices to cover the length of the dataset
        while len(indices) < len(self.data_source):
            random.shuffle(self.pids)
            for pid in self.pids:
                t = self.index_dic[pid]
                if len(t) < self.num_instances:
                    t = np.random.choice(t, size=self.num_instances, replace=True)
                else:
                    t = random.sample(t, self.num_instances)
                indices.extend(t)
                if len(indices) >= len(self.data_source):
                    break
        return iter(indices[:len(self.data_source)])

    def __len__(self):
        return len(self.data_source)

# ==========================================
# 3. EVALUATION FUNCTIONS
# ==========================================

def extract_features_loader(loader, model, device):
    feats_list, pids_list, cids_list = [], [], []
    
    # Extract backbone reference, handling DataParallel wrapping
    backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
    backbone.eval()
    
    with torch.no_grad():
        for imgs, pids, cids, _ in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(device)
            # Use ResNet backbone output directly
            feats = backbone(imgs)
            
            # L2 Normalize
            feats = feats.cpu().numpy()
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            feats = feats / norms
            
            feats_list.append(feats)
            pids_list.extend(pids.numpy())
            cids_list.extend(cids.numpy())
            
    return np.vstack(feats_list), np.array(pids_list), np.array(cids_list)

def eval_query(q_feat, q_pid, q_cid, gallery_feats, gallery_pids, gallery_cids):
    sim = np.dot(gallery_feats, q_feat)
    indices = np.argsort(sim)[::-1]
    
    # Filter same identity on same camera
    keep = ~((gallery_pids == q_pid) & (gallery_cids == q_cid))
    indices = indices[keep[indices]]
    
    if len(indices) == 0:
        return 0.0, np.zeros(10)
        
    matches = (gallery_pids[indices] == q_pid).astype(np.int32)
    total_g_matches = np.sum(matches)
    if total_g_matches == 0:
        return 0.0, np.zeros(10)
        
    cmc = np.zeros(10)
    for r in range(10):
        if np.any(matches[:r+1]):
            cmc[r] = 1.0
            
    raw_cumsum = np.cumsum(matches)
    ranks = np.arange(1, len(matches) + 1)
    precision = raw_cumsum / ranks
    ap = np.sum(precision * matches) / total_g_matches
    
    return ap, cmc

# ==========================================
# 4. TRAINING & BENCHMARK RUNNER
# ==========================================

def run_training(data_dir="./vric", epochs=5, batch_size=64, lr=0.001, subset=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    vric_dir = Path(data_dir).resolve()
    
    # Preprocessing with advanced data augmentation (added RandomErasing for Re-ID accuracy)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, value='random')
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading Dataset...")
    train_dataset = VRICTrainDataset(
        img_dir=vric_dir / "train_images",
        file_list_path=vric_dir / "vric_train.txt",
        transform=train_transform,
        subset_classes=subset
    )
    # Use RandomIdentitySampler to generate batches of PK (e.g. 16 identities, 4 images each = batch size 64)
    # This is essential for batch-hard triplet loss mining.
    sampler = RandomIdentitySampler(train_dataset, batch_size=batch_size, num_instances=4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)

    # Initialize model
    model = ReIDModel(num_classes=train_dataset.num_classes)
    
    # Freeze lower backbone layers initially
    print("Freezing lower ResNet layers (layer4 and classification head remain trainable)...")
    for name, param in model.backbone.named_parameters():
        if "layer4" not in name:
            param.requires_grad = False
            
    # Wrap model with DataParallel if multiple GPUs are available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel training.")
        model = nn.DataParallel(model)
        
    model.to(device)
    
    # Combined Cross-Entropy with Label Smoothing and Batch Hard Triplet Loss
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_tri = TripletLoss(margin=0.3)
    
    # Optimizer for active parameters
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # Added learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

    # Training
    backbone_unfrozen = False
    
    for epoch in range(1, epochs + 1):
        # Gradual unfreezing: unfreeze entire backbone after epoch 2 (at start of epoch 3)
        if epoch >= 3 and not backbone_unfrozen:
            print("\n>>> Unfreezing the entire ResNet backbone for fine-tuning...")
            actual_model = model.module if isinstance(model, nn.DataParallel) else model
            for param in actual_model.backbone.parameters():
                param.requires_grad = True
                
            # Recreate optimizer: low learning rate for backbone, standard for classification head
            optimizer = optim.Adam([
                {'params': actual_model.backbone.parameters(), 'lr': lr * 0.1},
                {'params': actual_model.fc.parameters(), 'lr': lr}
            ])
            # Recreate scheduler to hook onto the new optimizer
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
            backbone_unfrozen = True
            
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            feats, logits = model(imgs)
            
            # Combine Cross-Entropy and Triplet losses
            loss_ce = criterion_ce(logits, labels)
            loss_tri = criterion_tri(feats, labels)
            loss = loss_ce + loss_tri
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(logits, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'loss_ce': f"{loss_ce.item():.4f}",
                'loss_tri': f"{loss_tri.item():.4f}",
                'acc': f"{(correct.double() / total * 100):.2f}%"
            })
            
        print(f"Epoch {epoch} loss: {running_loss/len(train_dataset):.4f}, Acc: {(correct.double()/len(train_dataset)*100):.2f}%")
        scheduler.step()

    # Save fine-tuned checkpoint
    checkpoint_path = "./veri_resnet50_finetuned.pth"
    print(f"Saving checkpoint to {checkpoint_path}...")
    # Un-wrap DataParallel before saving to keep keys clean and compatible
    state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_to_save, checkpoint_path)

    # Re-ID Evaluation
    print("\nRunning Re-ID evaluation...")
    probe_dataset = VRICDataset(vric_dir / "probe_images", vric_dir / "vric_probe.txt", val_transform)
    probe_loader = DataLoader(probe_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    gallery_dataset = VRICDataset(vric_dir / "gallery_images", vric_dir / "vric_gallery.txt", val_transform)
    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    q_feats, q_pids, q_cids = extract_features_loader(probe_loader, model, device)
    g_feats, g_pids, g_cids = extract_features_loader(gallery_loader, model, device)

    all_ap, all_cmc = [], []
    for idx in range(len(q_feats)):
        ap, cmc = eval_query(q_feats[idx], q_pids[idx], q_cids[idx], g_feats, g_pids, g_cids)
        all_ap.append(ap)
        all_cmc.append(cmc)
        
    mAP = np.mean(all_ap)
    cmc_avg = np.mean(np.array(all_cmc), axis=0)
    
    print("\n" + "="*45)
    print("      KAGGLE FINE-TUNED MODEL RESULTS")
    print("="*45)
    print(f"Mean Average Precision (mAP): {mAP * 100:.2f}%")
    print(f"Rank-1 Accuracy:             {cmc_avg[0] * 100:.2f}%")
    print(f"Rank-5 Accuracy:             {cmc_avg[4] * 100:.2f}%")
    print(f"Rank-10 Accuracy:            {cmc_avg[9] * 100:.2f}%")
    print("="*45)

# ==========================================
# 5. EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # By default, run training on the full dataset (subset=0) for 5 epochs
    # Change these arguments as needed when calling the script
    run_training(data_dir="./vric", epochs=20, batch_size=64, lr=0.001, subset=0)
