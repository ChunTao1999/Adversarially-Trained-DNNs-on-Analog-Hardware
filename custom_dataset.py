import torch
import torch.utils.data as data
from   PIL import Image
import os
import os.path
import numpy as np
import pdb

class CUSTOM_DATA(data.Dataset):
    """
    Args:
        root  (string)   : Root directory of dataset.
        split (string)   : One of {'train', 'test'} to be loaded.
        transform (callable, optional): A function/transform that  takes in a PIL image and
                                        returns a transformed version. Ex: ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
                                               target and transforms it.
    """
    filename = ""

    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root             = os.path.expanduser(root)
        self.split            = split
        self.transform        = transform
        self.target_transform = target_transform
        self.split_list       = {'train': 'custom_train.npy',
                                 'test' : 'custom_test.npy'}

        if self.split not in self.split_list:
           raise ValueError('Wrong split entered! Please use split="train" '
                            'or split="test"')

        self.filename = os.path.join(self.root, self.split_list[self.split])

        # Load the dataset
        dataset     = np.load(self.filename, allow_pickle=True)
        self.data   = dataset.item()['data']
        self.label  = dataset.item()['label']
        self.target = dataset.item()['target']


    def __getitem__(self, index):
        """
        Args   : index (int): Index
        Returns: tuple: (image, target) 
        """
        
        img, target, label = self.data[index], self.target[index], self.label[index]

        # Doing this to be consistent with all other datasets and return PIL Image
        img = np.transpose(img, (1, 2, 0))
        #img = Image.fromarray((img * 255).astype(np.uint8))
        #img = np.transpose(img, (1, 2, 0))


        if self.transform is not None:
           img = self.transform(img)

        if self.target_transform is not None:
           target = self.target_transform(target)

        return img, target, label

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def get_mean_std(self):
        mean = []
        std  = []
        for i in range(self.data.shape[1]):
            ch_data = self.data[:, i, :, :].ravel()
            mean.append(np.mean(ch_data))
            std.append (np.std (ch_data))
        return torch.FloatTensor(mean), torch.FloatTensor(std)


class CUSTOM_CIFAR(data.Dataset):
    """
    Args:
        root  (string)   : Root directory of dataset.
        split (string)   : One of {'train', 'test'} to be loaded.
        transform (callable, optional): A function/transform that  takes in a PIL image and
                                        returns a transformed version. Ex: ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
                                               target and transforms it.
    """
    filename = ""

    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root             = os.path.expanduser(root)
        self.split            = split
        self.transform        = transform
        self.target_transform = target_transform
        self.split_list       = {'train': 'custom_train.npy',
                                 'test' : 'custom_test.npy'}

        if self.split not in self.split_list:
           raise ValueError('Wrong split entered! Please use split="train" '
                            'or split="test"')

        self.filename = os.path.join(self.root, self.split_list[self.split])

        # Load the dataset
        dataset     = np.load(self.filename, allow_pickle=True)
        self.data   = dataset.item()['data']
        self.label  = dataset.item()['label']
        self.target = dataset.item()['target']


    def __getitem__(self, index):
        """
        Args   : index (int): Index
        Returns: tuple: (image, target) 
        """
        
        img, target, label = self.data[index], self.target[index], self.label[index]

        # Doing this to be consistent with all other datasets and return PIL Image
        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray((img * 255).astype(np.uint8))
        #img = np.transpose(img, (1, 2, 0))


        if self.transform is not None:
           img = self.transform(img)

        if self.target_transform is not None:
           target = self.target_transform(target)

        return img, target, label

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def get_mean_std(self):
        mean = []
        std  = []
        for i in range(self.data.shape[1]):
            ch_data = self.data[:, i, :, :].ravel()
            mean.append(np.mean(ch_data))
            std.append (np.std (ch_data))
        return torch.FloatTensor(mean), torch.FloatTensor(std)

class CUSTOM_CIFAR_LABEL_ONLY(data.Dataset):
    """
    Args:
        root  (string)   : Root directory of dataset.
        split (string)   : One of {'train', 'test'} to be loaded.
        transform (callable, optional): A function/transform that  takes in a PIL image and
                                        returns a transformed version. Ex: ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
                                               target and transforms it.
    """
    filename = ""

    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root             = os.path.expanduser(root)
        self.split            = split
        self.transform        = transform
        self.target_transform = target_transform
        self.split_list       = {'train': 'custom_train.npy',
                                 'test' : 'custom_test.npy'}

        if self.split not in self.split_list:
           raise ValueError('Wrong split entered! Please use split="train" '
                            'or split="test"')

        self.filename = os.path.join(self.root, self.split_list[self.split])

        # Load the dataset
        dataset     = np.load(self.filename, allow_pickle=True)
        self.data   = dataset.item()['data']
        self.label  = dataset.item()['label']
        #self.target = dataset.item()['target']


    def __getitem__(self, index):
        """
        Args   : index (int): Index
        Returns: tuple: (image, target) 
        """
        
        #img, #target, label = self.data[index], self.target[index], self.label[index]
        img, label = self.data[index], self.label[index]

        # Doing this to be consistent with all other datasets and return PIL Image
        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray((img * 255).astype(np.uint8))
        #img = np.transpose(img, (1, 2, 0))

        if self.transform is not None:
           img = self.transform(img)

        if self.target_transform is not None:
           target = self.target_transform(target)

        return img, label

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def get_mean_std(self):
        mean = []
        std  = []
        for i in range(self.data.shape[1]):
            ch_data = self.data[:, i, :, :].ravel()
            mean.append(np.mean(ch_data))
            std.append (np.std (ch_data))
        return torch.FloatTensor(mean), torch.FloatTensor(std)



