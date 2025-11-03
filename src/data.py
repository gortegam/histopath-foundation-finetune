import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_transforms(size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf

class ImageFolderWithPaths(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for c in self.classes:
            cdir = os.path.join(root, c)
            if not os.path.isdir(cdir): 
                continue
            for fname in os.listdir(cdir):
                if fname.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff')):
                    self.samples.append((os.path.join(cdir, fname), self.class_to_idx[c]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, path

def make_loaders(data_dir: str, batch_size: int = 64, size: int = 224, num_workers: int = 0):
    train_tf, eval_tf = build_transforms(size)
    train_ds = ImageFolderWithPaths(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds   = ImageFolderWithPaths(os.path.join(data_dir, "val"),   transform=eval_tf)
    test_ds  = ImageFolderWithPaths(os.path.join(data_dir, "test"),  transform=eval_tf)

    # single-process loading to avoid /dev/shm pressure
    common = dict(num_workers=0, pin_memory=False, persistent_workers=False)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **common)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **common)
    test_ld  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **common)
    return train_ld, val_ld, test_ld, train_ds.classes

