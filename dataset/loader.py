from torch.utils import data
from PIL import Image
import torch
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader

def find_roi_mask(image_path):
    folder = os.path.dirname(image_path)
    base_name = os.path.basename(image_path)
    for f in os.listdir(folder):
        if f != base_name and f.lower().endswith(('.jpg', '.jpeg', '.png', '.dcm')):
            return os.path.join(folder, f)
    return None

def apply_roi_mask(image_path):
    image = Image.open(image_path).convert("RGB")
    mask_path = find_roi_mask(image_path)
    if mask_path is None:
        return image

    mask = Image.open(mask_path).convert("L")
    # Redimensiona a máscara para o tamanho da imagem
    mask = mask.resize(image.size, resample=Image.NEAREST)
    
    mask = np.array(mask)
    mask = (mask > 0).astype(np.uint8)
    image_array = np.array(image)
    masked_image_array = image_array * mask[:, :, None]
    return Image.fromarray(masked_image_array)

def my_collate_fn(batch):
    images, labels, meta_data, uids = zip(*batch)
    images = torch.stack(images, 0)          # empilha imagens
    return images, list(labels), list(meta_data), list(uids)


class MyDataset (data.Dataset):
    def __init__(self, imgs_path, labels, meta_data=None, transform=None):
        """
        The constructor gets the images path and their respectively labels and meta-data (if applicable).
        In addition, you can specify some transform operation to be carry out on the images.

        It's important to note the images must match with the labels (and meta-data if applicable). For example, the
        imgs_path[x]'s label must take place on labels[x].

        Parameters:
        :param imgs_path (list): a list of string containing the image paths
        :param labels (list) a list of labels for each image
        :param meta_data (list): a list of meta-data regarding each image. If None, there is no information.
        Defaul is None.
        :param transform (torchvision.transforms.Compose): transform operations to be carry out on the images
        """

        super().__init__()
        self.imgs_path = imgs_path
        self.labels = labels 
        self.meta_data = meta_data 

        # if transform is None, we need to ensure that the PIL image will be transformed to tensor, otherwise we'll get
        # an exception
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        """ This method just returns the dataset size """
        return len(self.imgs_path)


    def __getitem__(self, item):
        """
        It gets the image, labels and meta-data (if applicable) according to the index informed in `item`.
        It also performs the transform on the image.

        :param item (int): an index in the interval [0, ..., len(img_paths)-1]
        :return (tuple): a tuple containing the image, its label and meta-data (if applicable)
        """

        # Cut using ROI
        image = apply_roi_mask(self.imgs_path[item])

        # Applying the transformations
        image = self.transform(image)


        if self.meta_data is None:
            meta_data = []
            img_id = None
        else:
            meta_data = self.meta_data[item]
            img_id = meta_data['SeriesInstanceUID']


        if self.labels is None:
            labels = []
        else:
            labels = self.labels[item]

        return image, labels, meta_data, img_id


def get_data_loader (imgs_path, labels, meta_data=None, transform=None, batch_size=5, shuf=True, num_workers=4,
                     pin_memory=False):
    """
    This function gets a list og images path, their labels and meta-data (if applicable) and returns a DataLoader
    for these files. You also can set some transformations using torchvision.transforms in order to perform data
    augmentation. Lastly, params is a dictionary that you can set the following parameters:
    batch_size (int): the batch size for the dataset. If it's not informed the default is 30
    shuf (bool): set it true if wanna shuffe the dataset. If it's not informed the default is True
    num_workers (int): the number thread in CPU to load the dataset. If it's not informed the default is 0 (which


    :param imgs_path (list): a list of string containing the images path
    :param labels (list): a list of labels for each image
    :param meta_data (list, optional): a list of meta-data regarding each image. If it's None, it means there's
    no meta-data. Default is None
    :param transform (torchvision.transforms, optional): use the torchvision.transforms.compose to perform the data
    augmentation for the dataset. Alternatively, you can use the jedy.pytorch.utils.augmentation to perform the
    augmentation. If it's None, none augmentation will be perform. Default is None
    :param batch_size (int): the batch size. If the key is not informed or params = None, the default value will be 30
    :param shuf (bool): if you'd like to shuffle the dataset. If the key is not informed or params = None, the default
    value will be True
    :param num_workers (int): the number of threads to be used in CPU. If the key is not informed or params = None, the
    default value will be  4
    :param pin_memory (bool): set it to True to Pytorch preload the images on GPU. If the key is not informed or
    params = None, the default value will be True
    :return (torch.utils.data.DataLoader): a dataloader with the dataset and the chose params
    """

    dt = MyDataset(imgs_path, labels, meta_data, transform)
    dl = DataLoader (dataset=dt, batch_size=batch_size, shuffle=shuf, num_workers=num_workers,
                          pin_memory=pin_memory, collate_fn=my_collate_fn)
    return dl
