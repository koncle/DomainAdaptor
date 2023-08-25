# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# from einops import rearrange


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def Solarize(img, v):  # [0, 256]  All pixels above this greyscale level are inverted.
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]  number of bits to keep for each channel
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0,  1] grey -> original
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0, 1] black -> original
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0, 1]  black -> white
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0, 1, 2]  blured -> original -> sharpened
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def mixup(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img


def augment_list():
    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
        (Identity, 0., 1.0),

        (Equalize, 0, 1),  # 0
        (Invert, 0, 1),  # 1

        (Posterize, 0, 4),  # 2
        (Solarize, 0, 256),  # 3
        (SolarizeAdd, 0, 110),  # 4

        (AutoContrast, 0, 1),  # 5

        (Color, 0.1, 1.9),  # 7
        (Brightness, 0.1, 1.9),  # 8
        (Sharpness, 0.1, 1.9),  # 9

        (Contrast, 0.1, 1.9),  # 6
        # (CutoutAbs, 0, 40), # 12
        # (Rotate, 0, 30),  # 15
    ]
    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    def __init__(self, n=4, m=5):
        self.n = n
        self.m = m
        self.augment_list = augment_list()
        from torchvision import transforms
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, img, n, m):
        if n <= 0:
            return img
        idxes = [random.randint(0, len(self.augment_list) - 1) for i in range(n)]
        return self.post_transform(self.aug_img(idxes, img, m))

    def aug_img(self, idxes, img, m):
        for idx in idxes:
            op, minval, maxval = self.augment_list[idx]
            if m == -1:
                new_m = np.random.randint(0, 30)
            else:
                new_m = m
            val = (float(new_m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        return img

    def aug_batch(self, batch, n=5, m=4):
        imgs = []
        for img in batch:
            idxes = [random.randint(0, len(self.augment_list) - 1) for i in range(n)]
            img = self.aug_img(idxes, img, m)
            imgs.append(img)
        return imgs

    def aug_sequential(self, img, n=5, m=4):
        imgs = [img]
        idxes = [random.randint(0, len(self.augment_list) - 1) for i in range(n)]
        for idx in idxes:
            img = self.aug_img([idx], img, m)
            imgs.append(img)
        return imgs


class TestTimeAug(object):
    def __init__(self, args, randaug=False, jitter=False):
        self.args = args
        p = .1

        self.crop, self.flip, self.jitter, self.randaug = True, True, jitter, randaug

        self.resized_crop = transforms.RandomResizedCrop(args.img_size, scale=(args.min_scale, 1))
        self.rand_flip = transforms.RandomHorizontalFlip()
        self.randaug_op = RandAugment()
        self.jitter_op = transforms.ColorJitter(p, p, p, p)

        self.post_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def test_aug(self, image, bs=16):
        inputs = []

        for i in range(bs):
            img = image
            if self.flip:
                img = self.rand_flip(img)

            if self.jitter:
                img2 = self.jitter_op(img)
                img = Image.blend(img, img2, alpha=0.5)

            if self.randaug:
                img = self.randaug_op.aug_batch([img])[0]

            if self.crop:
                img = self.resized_crop(img)

            img = self.post_t(img)
            inputs.append(img)

        inputs = torch.stack(inputs, 0)
        return inputs


class Rotation(object):
    def tensor_rot_90(self, x):
        return x.flip(-1).transpose(-2, -1)

    def tensor_rot_180(self, x):
        return x.flip(-1).flip(-2)

    def tensor_rot_270(self, x):
        return x.transpose(-2, -1).flip(-1)

    def rotate_batch_with_labels(self, batch, labels):
        images = []
        for img, label in zip(batch, labels):
            if label == 1:
                img = self.tensor_rot_90(img)
            elif label == 2:
                img = self.tensor_rot_180(img)
            elif label == 3:
                img = self.tensor_rot_270(img)
            images.append(img.unsqueeze(0))
        return torch.cat(images)

    def __call__(self, img, rot_type='rand'): # rotate a single image
        data, label = self.rotate_batch(img.unsqueeze(0), rot_type)
        return data[0], label[0]

    def rotate_batch(self, batch, rot_type='rand'):
        if rot_type == 'rand':
            labels = torch.randint(4, (len(batch),), dtype=torch.long)
        elif rot_type == 'expand':
            labels = torch.cat([torch.zeros(len(batch), dtype=torch.long),
                                torch.zeros(len(batch), dtype=torch.long) + 1,
                                torch.zeros(len(batch), dtype=torch.long) + 2,
                                torch.zeros(len(batch), dtype=torch.long) + 3])
            batch = batch.repeat((4, 1, 1, 1))
        else:
            assert isinstance(rot_type, int)
            labels = torch.zeros((len(batch),), dtype=torch.long) + rot_type
        return self.rotate_batch_with_labels(batch, labels), labels.to(batch.device)
