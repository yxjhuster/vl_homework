import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(imdb):
    # DONE: classes: list of classes
    # DONE: class_to_idx: dictionary with keys=classes and values=class index
    # If you did Task 0, you should know how to set these values from the imdb
    num_classes = imdb.num_classes
    classes = imdb.classes
    class_to_idx = {}
    for i in range(num_classes):
        class_to_idx[classes[i]] = i
    return classes, class_to_idx


def make_dataset(imdb, class_to_idx):
    # DONE: return list of (image path, list(+ve class indices)) tuples
    # You will be using this in IMDBDataset
    dataset_list = []
    idx_list = imdb._image_index
    roi = imdb.gt_roidb()
    for i, idx in enumerate(idx_list):
        image_path = imdb.image_path_from_index(idx)
        gt_classes = (roi[i]['gt_classes'] - 1).tolist()
        dataset_list.append((image_path, gt_classes))
    return dataset_list


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        # DONE: Define model
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11),
                      stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2),
                         dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5),
                      stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2),
                         dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True))

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1)))

    def forward(self, x):
        # DONE: Define forward pass
        x = self.features(x)
        x = self.classifier(x)
        return x


class LocalizerAlexNetHighres(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetHighres, self).__init__()
        # DONE: Ignore for now until instructed
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11),
                      stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2),
                         dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5),
                      stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2),
                         dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, x):
        # DONE: Ignore for now until instructed
        x = self.features(x)
        x = self.classifier(x)

        return x


def localizer_alexnet(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    # DONE: Initialize weights correctly based on whethet it is pretrained or not

    def init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant(m.bias.data, 0)

    model.apply(init_weights)

    if pretrained:
        weights = model_zoo.load_url(model_urls['alexnet'])
        model_pretrained = model.state_dict()

        for i in model_pretrained.keys():
            if 'features' in i:
                model_pretrained[i] = weights[i]
        model.load_state_dict(model_pretrained)

    return model


def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetHighres(**kwargs)
    # TODO: Ignore for now until instructed

    def init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant(m.bias.data, 0)
    model.apply(init_weights)

    if pretrained:
        weights = model_zoo.load_url(model_urls['alexnet'])
        model_pretrained = model.state_dict()

        for i in model_pretrained.keys():
            if 'features' in i:
                model_pretrained[i] = weights[i]
        model.load_state_dict(model_pretrained)

    return model


class IMDBDataset(data.Dataset):
    """A dataloader that reads imagesfrom imdbs
    Args:
        imdb (object): IMDB from fast-rcnn repository
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, list(+ve class indices)) tuples
    """

    def __init__(self, imdb, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(imdb)
        imgs = make_dataset(imdb, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images, what's going on?"))
        self.imdb = imdb
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a binary vector with 1s
                                   for +ve classes and 0s for -ve classes
                                   (it can be a numpy array)
        """
        # DONE: Write this function, look at the imagenet code for inspiration
        path, class_list = self.imgs[index]
        img = Image.open(path)
        target = np.zeros(len(self.classes), dtype=np.int)
        target[np.asarray(self.imgs[index][1])] = 1

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
