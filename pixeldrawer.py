from DrawingInterface import DrawingInterface

import torch
import skimage
import skimage.io
import random
import ttools.modules
import argparse
import math
import torchvision
import torchvision.transforms as transforms
import numpy as np
import PIL.Image

to_tensor = transforms.ToTensor()
to_img = transforms.ToPILImage()

class PixelDrawer(DrawingInterface):
    num_rows = 45
    num_cols = 80
    do_mono = False
    pixels = []
    init_image = None

    def __init__(self, lr, pixel_size, scale, init_image=None):
        super(DrawingInterface, self).__init__()
        self.lr = lr
        self.init_image = init_image
        self.pixel_size = pixel_size
        self.scale = scale

    def load_model(self, config_path, checkpoint_path, device):
        if self.init_image:
            self.current = to_tensor(self.init_image).cuda()

        else:
            self.current = torch.rand(3, self.pixel_size[1], self.pixel_size[0]).cuda()

        self.current.requires_grad = True

        self.opts = [torch.optim.Adam([self.current], lr=self.lr)]

    def get_opts(self):
        return self.opts

    def rand_init(self, toksX, toksY):
        pass

    def init_from_tensor(self, init_tensor):
        pass

    def reapply_from_tensor(self, new_tensor):
        pass

    def get_z_from_tensor(self, ref_tensor):
        return None

    def get_num_resolutions(self):
        return 5

    def synth(self, cur_iteration):
        return torch.nn.functional.interpolate(
            self.current.unsqueeze(0), 
            (1, 3, self.pixel_size[1] * self.scale, self.pixel_size[0] * self.scale), 
            mode='nearest'
        )

    @torch.no_grad()
    def to_image(self):
        return to_img(self.current).resize(
            (self.pixel_size[0] * self.scale, self.pixel_size[1] * self.scale),
             PIL.Image.NEAREST
        )

    def clip_z(self):
        with torch.no_grad():
            self.current.clamp_(0.0, 1.0)

    def get_z(self):
        return None

    def get_z_copy(self):
        return None
