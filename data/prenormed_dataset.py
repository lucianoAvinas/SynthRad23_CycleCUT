import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import tifffile as tiff
from PIL import Image
import random
import torchvision.transforms as transforms
from torch import from_numpy
import numpy as np
import util.util as util

import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset

#from skimage import color

class prenormedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

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
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        # print(self.A_paths)
        # print(self.B_size)
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
         
        # assert(input_nc == 1); assert(output_nc == 1)
        # print(self.A_paths)
        
        self.transform_A = get_transform(self.opt, data_id="A")
        self.transform_B = get_transform(self.opt, data_id="B")

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        # A_img = tiff.imread(A_path)
        # B_img = tiff.imread(B_path)
        A_img = np.array(np.load(A_path))[:,:,:]
        B_img = np.array(np.load(B_path))[:,:,:]

        # is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        # modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        # transform = get_transform(modified_opt)
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        #C = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        #print(C["A"])
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


class ACast(object):
    def __init__(self, no_flip):
        self.no_flip = no_flip

    def __repr__(self):
        return "16 bit kV Image cast to normalized [-1,1] Tensor"

    def __call__(self, pic):
        # img = np.minimum(np.array(pic, np.int16), 2**12)
        # 2**12 for CT, 2**14 for MRI
        #img = np.minimum(np.array(pic, np.float64), 1)
        img = np.minimum(np.array(pic, np.float64), 1)
        # img = np.maximum(np.array(pic, np.int16), 0)
        # img = img - 1024
        # img = pic/(2**6)
        #img = img/3
        if not self.no_flip and random.random() < 0.5:
           img = np.flip(img,2)

        # im = np.array(img)
        # lab = color.rgb2lab(im).astype(np.float32)
        # # lab = lab/(2**12)
        # img1 = transforms.ToTensor()(lab)
        # img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img1)
        # return img

        #img = img.view(pic.size[1], pic.size[0], 3)
        # img = img.transpose(0, 1).transpose(0, 2).contiguous()
        # print(img.shape)
        # img = img.transpose(2, 0, 1)
        # print(img.shape)
        # img = img/(1.5)
        # img = pic/(2**12)
        #temp = from_numpy(np.array(pic, np.int32)).float()/(2**12)
        # print(pic.img())
        #img = (img-0.7)
        img = from_numpy(img.copy())
        return img.float()


class BCast(object):
    def __init__(self, no_flip):
        self.no_flip = no_flip

    def __repr__(self):
        return "8 bit water/fat Image cast to normalized [-1,1] Tensor"

    def __call__(self, pic):
        # img = np.minimum(np.array(pic, np.int16), -2**14)
        # img = np.maximum(np.array(pic, np.int16), 0)
        #img = pic+1024
        img = np.minimum(np.array(pic, np.float64), 1)
        # img = img - 64
        if not self.no_flip and random.random() < 0.5:
           img = np.flip(img,2)
        
        # im = np.array(img)
        # lab = color.rgb2lab(im).astype(np.float32)
        # # lab = lab/(2**8)
        # img1 = transforms.ToTensor()(lab)
        # img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img1)
        # return img

        #img = img.view(pic.size[1], pic.size[0], 3)
        # img = img.transpose(0, 2)
        #img = img.transpose(0, 1).transpose(0, 2).contiguous()
        # print(img.size)
        # img = img.transpose(2, 0, 1)
        # print(img.shape)
        # img = img.transpose(2, 0, 1)
        # print(img.shape)
        #img = img/(2**12)
        
        #temp = from_numpy(np.array(pic, np.int32)).float()/(2**12)
        #img = (img-0.5)/0.5
        img = from_numpy(img.copy())
        return img.float()


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, data_id, params=None, method=Image.BICUBIC, convert=True):
    transform_list = []
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        #transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        #transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))
        pass

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        #transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))
        pass

    #if not opt.no_flip:
    #    if params is None:
    #        transform_list.append(transforms.RandomHorizontalFlip())
    #    elif params['flip']:
    #        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        if data_id == "A":
            transform_list += [ACast(opt.no_flip)]
        else:
            transform_list += [BCast(opt.no_flip)]

        #transform_list += [transforms.Normalize((0.5,), (0.5,))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

