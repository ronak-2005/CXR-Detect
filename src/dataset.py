import random
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

CLASSES      = ["NORMAL", "PNEUMONIA"]
CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}

def get_transforms(img_size, split, cfg):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    
    if split == "train":
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),     
            transforms.RandomCrop(img_size),             
            transforms.RandomHorizontalFlip(p=cfg["augmentation"]["random_horizontal_flip"]),      
            transforms.RandomRotation(cfg["augmentation"]["random_rotation"]),
            transforms.ColorJitter(                      
                brightness=cfg["augmentation"]["brightness"],
                contrast=cfg["augmentation"]["contrast"],
            ),
            transforms.ToTensor(),                       
            transforms.Normalize(mean, std),             
            transforms.RandomErasing(p=cfg["augmentation"]["random_erasing"])             
        ])
    
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),  
        transforms.ToTensor(),                    
        transforms.Normalize(mean, std)           
    ])

class XRayDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root      = Path(root)
        self.transform = transform
        self.samples   = self._load_samples()
    
    def _load_samples(self):
     samples = []
     exts = {".jpg", ".jpeg", ".png"}
     for cls in CLASSES:
        folder = self.root / cls
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        for file in folder.iterdir():
            if file.suffix.lower() in exts:
                samples.append((file, CLASS_TO_IDX[cls]))
     return samples
    
    def class_counts(self):
     counts = [0] * len(CLASSES)
     for _, label in self.samples:
        counts[label] += 1
     return counts
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label, str(path)
    
def build_sampler(dataset):
    counts = dataset.class_counts()     
    total = sum(counts)
    num_classes = len(counts)

    class_weights = [total / (num_classes * c) for c in counts]

    sample_weights = []
    for _, label in dataset.samples:
        sample_weights.append(class_weights[label])

    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def build_loaders(cfg):

    img_size = cfg["data"]["img_size"]
    root     = Path(cfg["data"]["root"])
    batch_size = cfg["training"]["batch_size"]
    workers    = cfg["data"]["num_workers"]

    train_ds = XRayDataset(root / "train", transform=get_transforms(img_size, "train", cfg))
    val_ds   = XRayDataset(root / "val",   transform=get_transforms(img_size, "val", cfg))
    test_ds  = XRayDataset(root / "test",  transform=get_transforms(img_size, "test", cfg))

    sampler = build_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


def get_tta_transforms(img_size):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return [
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([
            transforms.Resize((img_size + 20, img_size + 20)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=(-10, -10)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    ]