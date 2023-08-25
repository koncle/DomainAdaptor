from random import random

import numpy as np
import torch
import torchvision


class JigsawDataset():
    def __init__(self, jig_classes=31, bias_whole_image=None):
        self.permutations = self.__retrieve_permutations(jig_classes)
        self.grid_size = 3
        self.bias_whole_image = bias_whole_image
        self.make_grid = lambda x: torchvision.utils.make_grid(x, self.grid_size, padding=0)

    def __call__(self, img):
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            tiles[n] = self.get_tile(img, n)

        order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0

        if order == 0:
            data = tiles
        else:
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]

        data = torch.stack(data, 0)
        return self.make_grid(data), int(order)
        # return {'jigsaw': self.make_grid(data), 'jigsaw_label': int(order)}

    def get_tile(self, img, n):
        w = int(img.shape[-1] / self.grid_size)
        y = int(n / self.grid_size)
        x = int(n % self.grid_size)
        tile = img[:, y * w:(y + 1) * w, x * w:(x + 1) * w]
        # tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        return tile

    def __retrieve_permutations(self, classes):
        all_perm = np.load('dataloader/jigsaw/permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1
        return all_perm
