import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch, cv2
import torchvision
from torch.utils import data
from PIL import Image
import matplotlib.pyplot as plt

normMean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
normStd = np.array((0.229, 0.224, 0.225), dtype=np.float32)

def standardization(image):
    image = ((image / 255) - normMean) / normStd
    return image

class ISPRSDataset(data.Dataset):
    def __init__(
            self,
            images_dir,
            masks_dir,
            dtype = None,
            crop_size = None,
            max_iters=None,
            augmentation=None,
            preprocessing=True,
    ):

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.gens_num = len(self.images_dir)
        self.img_ids = os.listdir(self.images_dir)
        self.lenth = len(self.img_ids)
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(self.images_dir, name)
            label_file = os.path.join(self.masks_dir, name)

            self.files.append({
                "img": img_file,
                "label": label_file
            })

    def __getitem__(self, index):
        datafiles = self.files[index]
        # read data
        image = Image.open(datafiles["img"]).convert('RGB')
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = Image.open(datafiles["label"]).convert('L')

        # resize
        if self.crop_size != None:
            image = image.resize(self.crop_size, Image.BICUBIC)
            mask = mask.resize(self.crop_size, Image.NEAREST)


        image = np.asarray(image)
        mask = np.asarray(mask)

        # plt.subplot(221)
        # plt.imshow(image)
        # plt.subplot(222)
        # plt.imshow(mask)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # plt.subplot(223)
        # plt.imshow(image)
        # plt.subplot(224)
        # plt.imshow(mask)
        # plt.show()

        image = np.asarray(image, np.float32)
        mask = np.asarray(mask, np.float32)
        # apply preprocessing
        if self.preprocessing:
            # sample = self.preprocessing(image=image, mask=mask)
            # image, mask = sample['image'], sample['mask']
            image=standardization(image)

        image = image.transpose((2, 0, 1))
        # print(len(self.img_ids))

        return image.copy(), mask.copy(), datafiles["img"]

    def __len__(self):
        return len(self.img_ids)
        #return self.lenth

class ISPRSDataset_val(data.Dataset):
    def __init__(
            self,
            images_dir,
            masks_dir,
            crop_size = None,
            max_iters=None,
            augmentation=None,
            preprocessing=True,
    ):

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.gens_num = len(self.images_dir)
        self.img_ids = os.listdir(self.images_dir)
        self.lenth = len(self.img_ids)
        # if not max_iters==None:
        #     self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(self.images_dir, name)
            label_file = os.path.join(self.masks_dir, name)

            self.files.append({
                "img": img_file,
                "label": label_file
            })

    def __getitem__(self, index):
        datafiles = self.files[index]
        # read data
        image = Image.open(datafiles["img"]).convert('RGB')
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = Image.open(datafiles["label"]).convert('L')

        # resize
        if self.crop_size != None:
            image = image.resize(self.crop_size, Image.BICUBIC)
            mask = mask.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        mask = np.asarray(mask, np.float32)
        # print(image.shape, mask.shape)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            # sample = self.preprocessing(image=image, mask=mask)
            # image, mask = sample['image'], sample['mask']
            image=standardization(image)

        image = image.transpose((2, 0, 1))
        # print(len(self.img_ids))

        return image.copy(), mask.copy(), datafiles["img"]

    def __len__(self):
        # return len(self.img_ids)
        return self.lenth


class GTA5DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name


if __name__ == '__main__':
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
