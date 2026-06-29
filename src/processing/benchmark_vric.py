import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Add project root to python path to allow imports from src
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.processing.entityprofiling import EntityFeatureExtractor

class VRICDataset(Dataset):
    """
    PySide/PyTorch compatible dataset for VRIC images.
    """
    def __init__(self, img_dir, file_list_path, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.samples = []
        
        if not file_list_path.exists():
            raise FileNotFoundError(f"Label file not found: {file_list_path}")
            
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
        
        # Load image via PIL
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"\nWarning: Failed to load image {img_path}: {e}")
            # Return dummy blank image if load fails
            img = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            img = self.transform(img)
            
        return img, vehicle_id, cam_id, img_name

def extract_features_loader(loader, extractor):
    """
    Runs batch feature extraction on a DataLoader using the model from EntityFeatureExtractor.
    """
    feats_list = []
    pids_list = []
    cids_list = []
    names_list = []
    
    extractor.model.eval()
    with torch.no_grad():
        for imgs, pids, cids, names in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(extractor.device)
            feats = extractor.model(imgs)
            
            # Move to CPU, convert to numpy
            feats = feats.cpu().numpy()
            
            # L2 Normalize the features (standard Re-ID practice)
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            feats = feats / norms
            
            feats_list.append(feats)
            pids_list.extend(pids.numpy())
            cids_list.extend(cids.numpy())
            names_list.extend(names)
            
    return np.vstack(feats_list), np.array(pids_list), np.array(cids_list), names_list

def eval_query(q_feat, q_pid, q_cid, gallery_feats, gallery_pids, gallery_cids):
    """
    Evaluates a single query against the gallery and returns AP and CMC metrics.
    """
    # Cosine similarity is the dot product of normalized feature vectors
    sim = np.dot(gallery_feats, q_feat)
    
    # Sort indices in descending order of similarity
    indices = np.argsort(sim)[::-1]
    
    # Filter out gallery images of the same vehicle captured by the same camera (standard protocol)
    keep = ~((gallery_pids == q_pid) & (gallery_cids == q_cid))
    indices = indices[keep[indices]]
    
    if len(indices) == 0:
        return 0.0, np.zeros(10)
        
    # Generate binary matches list (1 if match, 0 if distractor)
    matches = (gallery_pids[indices] == q_pid).astype(np.int32)
    
    total_g_matches = np.sum(matches)
    if total_g_matches == 0:
        return 0.0, np.zeros(10)
        
    # Calculate Rank accuracies (CMC) up to Rank-10
    cmc = np.zeros(10)
    for r in range(10):
        if np.any(matches[:r+1]):
            cmc[r] = 1.0
            
    # Calculate Average Precision (AP)
    raw_cumsum = np.cumsum(matches)
    ranks = np.arange(1, len(matches) + 1)
    precision = raw_cumsum / ranks
    
    ap = np.sum(precision * matches) / total_g_matches
    return ap, cmc

def main():
    parser = argparse.ArgumentParser(description="Benchmark EntityFeatureExtractor on the VRIC Dataset")
    parser.add_argument("--data_dir", type=str, default="data/vric", help="Path to VRIC dataset directory")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for feature extraction")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching of extracted features")
    args = parser.parse_args()

    vric_dir = Path(args.data_dir).resolve()
    if not vric_dir.exists():
        print(f"Error: Dataset directory {vric_dir} does not exist. Please run the download script first.")
        sys.exit(1)

    cache_file = vric_dir / "vric_features_cache.npz"
    
    if cache_file.exists() and not args.no_cache:
        print(f"Loading cached features from {cache_file}...")
        data = np.load(cache_file, allow_pickle=True)
        q_feats = data["q_feats"]
        q_pids = data["q_pids"]
        q_cids = data["q_cids"]
        g_feats = data["g_feats"]
        g_pids = data["g_pids"]
        g_cids = data["g_cids"]
    else:
        print("Initializing EntityFeatureExtractor...")
        extractor = EntityFeatureExtractor(use_gpu=args.gpu)
        
        print("Preparing Datasets and DataLoaders...")
        # Setup Probe (Query) dataset
        probe_dataset = VRICDataset(
            img_dir=vric_dir / "probe_images",
            file_list_path=vric_dir / "vric_probe.txt",
            transform=extractor.transform
        )
        probe_loader = DataLoader(probe_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        # Setup Gallery dataset
        gallery_dataset = VRICDataset(
            img_dir=vric_dir / "gallery_images",
            file_list_path=vric_dir / "vric_gallery.txt",
            transform=extractor.transform
        )
        gallery_loader = DataLoader(gallery_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        print(f"Extracting features for {len(probe_dataset)} probe images...")
        q_feats, q_pids, q_cids, _ = extract_features_loader(probe_loader, extractor)
        
        print(f"Extracting features for {len(gallery_dataset)} gallery images...")
        g_feats, g_pids, g_cids, _ = extract_features_loader(gallery_loader, extractor)
        
        if not args.no_cache:
            print(f"Caching extracted features to {cache_file}...")
            np.savez_compressed(
                cache_file,
                q_feats=q_feats,
                q_pids=q_pids,
                q_cids=q_cids,
                g_feats=g_feats,
                g_pids=g_pids,
                g_cids=g_cids
            )
            
    print("Evaluating Re-ID metrics (mAP, Rank-1, Rank-5)...")
    num_queries = len(q_feats)
    all_ap = []
    all_cmc = []
    
    for idx in tqdm(range(num_queries), desc="Computing similarities"):
        q_feat = q_feats[idx]
        q_pid = q_pids[idx]
        q_cid = q_cids[idx]
        
        ap, cmc = eval_query(q_feat, q_pid, q_cid, g_feats, g_pids, g_cids)
        all_ap.append(ap)
        all_cmc.append(cmc)
        
    mAP = np.mean(all_ap)
    cmc_avg = np.mean(np.array(all_cmc), axis=0)
    
    print("\n" + "="*40)
    print("           VRIC BENCHMARK RESULTS")
    print("="*40)
    print(f"Mean Average Precision (mAP): {mAP * 100:.2f}%")
    print(f"Rank-1 Accuracy:             {cmc_avg[0] * 100:.2f}%")
    print(f"Rank-5 Accuracy:             {cmc_avg[4] * 100:.2f}%")
    print(f"Rank-10 Accuracy:            {cmc_avg[9] * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
