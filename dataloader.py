from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageFile
import numpy as np
from glob import glob
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CardiacSegmentation(Dataset):
    def __init__(self, root, transforms=None):
        self.img_paths = sorted(glob(f'{root}/images/*.png'))
        self.mask_paths = sorted(glob(f'{root}/labels/*.png'))
        self.transforms = transforms
        self.num_classes = 2
        assert len(self.img_paths) == len(self.mask_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img, mask = self.get_img_mask(self.img_paths[idx], self.mask_paths[idx])
        if self.transforms:
            img, mask = self.apply_transforms(img, mask)

        return img, (mask / 255).int()

    def get_img_mask(self, img_path, mask_path):
        return self.read(img_path, mask_path)

    def read(self, img_path, mask_path):
        return np.array(Image.open(img_path).convert('RGB')), np.array(Image.open(mask_path).convert('L'))

    def apply_transforms(self, img, mask):
        transformed = self.transforms(image=img, mask=mask)
        return transformed['image'], transformed['mask']

def get_loaders(root, transforms, batch_size, num_workers, split = [0.9, 0.05, 0.05]):
    assert sum(split) == 1., 'Sum of the split must be exactly 1'

    dataset = CardiacSegmentation(root=root, transforms=transforms)
    num_classes = dataset.num_classes

    total_size = len(dataset)
    trn_size = int(total_size * split[0])
    val_size = int(total_size * split[2])
    test_size = total_size - trn_size - val_size

    trn_data, val_data, test_data = random_split(dataset, [trn_size, val_size, test_size])

    print(f'\nThere are {len(trn_data)} images in the train set')
    print(f'\nThere are {len(val_data)} images in the val set')
    print(f'\nThere are {len(test_data)} images in the test set')

    trn_loader = DataLoader(trn_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=num_workers)

    return trn_loader, val_loader, test_loader, num_classes