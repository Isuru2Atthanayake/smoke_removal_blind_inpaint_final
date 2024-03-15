import torch
import random
import string

from torch.autograd import Variable
from torch.utils import data
import os
import os
# print("Current working directory:", os.getcwd())
from PIL import Image
from torchvision import transforms
from utils.config import get_cfg_defaults

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def unnormalize_batch(batch, mean_, std_, div_factor=1.0):
    """
    Unnormalize batch
    :param batch: input tensor with shape
     (batch_size, nbr_channels, height, width)
    :param div_factor: normalizing factor before data whitening
    :return: unnormalized data, tensor with shape
     (batch_size, nbr_channels, height, width)
    """
    # normalize using dataset mean and std
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = mean_[0]
    mean[:, 1, :, :] = mean_[1]
    mean[:, 2, :, :] = mean_[2]
    std[:, 0, :, :] = std_[0]
    std[:, 1, :, :] = std_[1]
    std[:, 2, :, :] = std_[2]
    batch = torch.div(batch, div_factor)

    batch *= Variable(std)
    batch = torch.add(batch, Variable(mean))
    return batch


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def linear_scaling(x):
    return (x * 255.) / 127.5 - 1.


def linear_unscaling(x):
    return (x + 1.) * 127.5 / 255.

class RaindropDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data = self._make_dataset()

    def __getitem__(self, index):
        img_path, label_path = self.data[index]
        # print("Image path:", img_path)
        # print("Label path:", label_path)

        img = self._read_img(img_path)
        label = self._read_img(label_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.data)

    def _make_dataset(self): #checks for the train and mask folders in the root directory
        dataset = []
        # train_dir = os.path.join(self.root, "train")
        # mask_dir = os.path.join(train_dir, "mask")
        train_dir = os.path.join(self.root, "train")
        mask_dir = os.path.join(train_dir, "mask")

        # print("Train directory:", train_dir)
        # print("Mask directory:", mask_dir)

        # Check if the necessary directories exist
        if not os.path.exists(train_dir) or not os.path.exists(mask_dir):
            print("Error: 'train' or 'mask' directories not found.")
            print("Error: 'train' or 'mask' directories not found.")
            print(f"train_dir exists: {os.path.exists(train_dir)}")
            print(f"mask_dir exists: {os.path.exists(mask_dir)}")
            return dataset  # Return empty dataset

        # Collect image paths and their corresponding label paths
        for subdir in os.listdir(train_dir):
            if subdir == "mask":
                continue  # Skip the mask directory
            subdir_path = os.path.join(train_dir, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    if filename.endswith('.png'):
                        img_path = os.path.join(subdir_path, filename)
                        label_path = os.path.join(mask_dir, filename)
                        dataset.append((img_path, label_path))

        return dataset

    def _read_img(self, im_path):
        return Image.open(im_path).convert("RGB")


if __name__ == '__main__': # the following code is executed when the file is run directly
    cfg = get_cfg_defaults()
    transform = transforms.Compose([
        transforms.Resize(cfg.DATASET.SIZE),
        transforms.CenterCrop(cfg.DATASET.SIZE),
        transforms.ToTensor()
    ])
    # dataset = RaindropDataset("datasets/smoke_dataset", transform=transform, target_transform=transform)
    dataset = RaindropDataset("datasets/smoke_dataset", transform=transform, target_transform=transform)
    if len(dataset) > 0:
        img, y = dataset.__getitem__(3)
        print("Image size:", img.size())
        print("Label size:", y.size())
    else:
        print("No data found. Exiting.")