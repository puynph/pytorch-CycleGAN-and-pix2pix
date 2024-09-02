"""Dataset class template

Custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class FundusDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""


    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  
        self.B_size = len(self.B_paths)  
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        # self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        # self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.transform_A = get_transform(self.opt)
        self.transform_B = get_transform(self.opt)

        self.num_augmentations = 7


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        # Calculate the index for the image and the augmentation
        img_idx = index // self.num_augmentations
        aug_idx = index % self.num_augmentations

        # Get paths for images A and B
        A_path = self.A_paths[img_idx % self.A_size]
        index_B = img_idx % self.B_size
        B_path = self.B_paths[index_B]

        # Load images A and B
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # Apply the same augmentation to both A and B
        if aug_idx == 0:  # Original images
            A = self.transform_A(A_img)
            B = self.transform_B(B_img)
        elif aug_idx == 1:  # Horizontal flip
            A = self.transform_A(A_img.transpose(Image.FLIP_LEFT_RIGHT))
            B = self.transform_B(B_img.transpose(Image.FLIP_LEFT_RIGHT))
        elif aug_idx == 2:  # Vertical flip
            A = self.transform_A(A_img.transpose(Image.FLIP_TOP_BOTTOM))
            B = self.transform_B(B_img.transpose(Image.FLIP_TOP_BOTTOM))
        elif aug_idx == 3:  # Rotate 5 degrees clockwise
            A = self.transform_A(A_img.rotate(-5))
            B = self.transform_B(B_img.rotate(-5))
        elif aug_idx == 4:  # Rotate 10 degrees clockwise
            A = self.transform_A(A_img.rotate(-10))
            B = self.transform_B(B_img.rotate(-10))
        elif aug_idx == 5:  # Rotate 5 degrees counterclockwise
            A = self.transform_A(A_img.rotate(5))
            B = self.transform_B(B_img.rotate(5))
        elif aug_idx == 6:  # Rotate 10 degrees counterclockwise
            A = self.transform_A(A_img.rotate(10))
            B = self.transform_B(B_img.rotate(10))
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images multiplied by the number of augmentations."""
        return max(self.A_size, self.B_size) * self.num_augmentations
