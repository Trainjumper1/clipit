import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from IPython import display
from PIL import Image
from CLIP import clip
from torchvision import transforms
from tqdm.notebook import tqdm
from torchvision.transforms import InterpolationMode as im
import kornia
import kornia.augmentation as K
from typing import cast, Dict, List, Optional, Tuple, Union

to_img = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

device = torch.device("cuda")

AUGMENT_COUNT = 40
NOISE_FAC = 0.05

class MyRandomPerspective(K.RandomPerspective):
    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, _, height, width = input.shape
        transform = cast(torch.Tensor, transform)
        return kornia.geometry.warp_perspective(
            input, transform, (height, width),
             mode=self.resample.name.lower(), align_corners=self.align_corners, padding_mode="border"
        )

class PixelDrawer:
    size = 100
    scale = 3

    def init(self):
        self.width = self.size
        self.height = self.size

        self.current = torch.rand(3, self.width, self.height).cuda()
        # pil_img = Image.open("lake.webp").convert('RGB').resize((self.size, self.size), Image.NEAREST)
        # self.current = to_tensor(pil_img).cuda()
        self.current.requires_grad = True

        self.prompt = "medieval city on fire landscape #pixelart"
                    
        self.clip_model, _ = clip.load("ViT-B/32", device=device)

        self.vsize = self.clip_model.visual.input_resolution

        prompt_in = clip.tokenize(self.prompt).to(device)

        with torch.no_grad():
            self.prompt_features = self.clip_model.encode_text(prompt_in)

        self.optimizer = torch.optim.Adam([self.current], lr=0.01)

        # self.zoom_augment = transforms.Compose([
        #   transforms.ColorJitter(hue=0.1, saturation=0.15, brightness=0.05, contrast=0.05),

        #   transforms.Resize(self.clip_model.visual.input_resolution),

        #   transforms.RandomApply([transforms.RandomResizedCrop(
        #       self.clip_model.visual.input_resolution,
        #       scale = (0.1, 0.4),
        #       ratio = (0.85, 1.17),
        #       interpolation = im.NEAREST
        #   )], p=0.9),

        #   transforms.RandomPerspective(
        #       distortion_scale=0.5, 
        #       p=0.7,
        #   ),
        # ])

        # n_s = 0.9
        # n_t = (1-n_s)/2

        # self.wide_augment = transforms.Compose([
        #     transforms.ColorJitter(hue=0.1, saturation=0.15, brightness=0.05, contrast=0.05),

        #     transforms.Resize(self.clip_model.visual.input_resolution),

        #     transforms.RandomAffine(
        #         degrees=0,
        #         translate=(n_t, n_t), 
        #         scale=(n_s, n_s),
        #     ),

        #     transforms.RandomPerspective(
        #         distortion_scale=0.2, 
        #         p=0.7,
        #     ),
        # ])

        augmentations = []
        augmentations.append(MyRandomPerspective(distortion_scale=0.40, p=0.7, return_transform=True))
        augmentations.append(K.RandomResizedCrop(size=(self.vsize,self.vsize), scale=(0.1,0.75),  ratio=(0.85,1.2), cropping_mode='resample', p=0.7, return_transform=True))
        augmentations.append(K.ColorJitter(hue=0.1, saturation=0.1, p=0.8, return_transform=True))
        self.zoom_augment = nn.Sequential(*augmentations)

        augmentations = []

        n_s = 0.95
        n_t = (1-n_s)/2
        augmentations.append(K.RandomAffine(degrees=0, translate=(n_t, n_t), scale=(n_s, n_s), p=1.0, return_transform=True))

        # augmentations.append(K.CenterCrop(size=(self.self.vsize,self.self.vsize), p=1.0, cropping_mode="resample", return_transform=True))
        augmentations.append(K.CenterCrop(size=self.vsize, cropping_mode='resample', p=1.0, return_transform=True))
        augmentations.append(K.RandomPerspective(distortion_scale=0.20, p=0.7, return_transform=True))
        augmentations.append(K.ColorJitter(hue=0.1, saturation=0.1, p=0.8, return_transform=True))
        self.wide_augment = nn.Sequential(*augmentations)


    def run(self):
        for i in tqdm(range(5000)):
            self.optimizer.zero_grad()

            batch1 = self.current.unsqueeze(0).expand(AUGMENT_COUNT, 3, self.width, self.height)
            batch1, _ = self.wide_augment(transforms.Resize(self.vsize)(batch1))

            batch2 = self.current.unsqueeze(0).expand(AUGMENT_COUNT, 3, self.width, self.height)
            batch2, _ = self.wide_augment(transforms.Resize(self.vsize)(batch2))

            batch = torch.cat((normalize(batch1), normalize(batch2)))

            # with torch.no_grad():
            #   facs = batch.new_empty([AUGMENT_COUNT * 2, 1, 1, 1]).uniform_(-NOISE_FAC, NOISE_FAC)
            
            # batch = batch + facs * torch.randn_like(batch)

            image_features = self.clip_model.encode_image(batch)

            cos = nn.CosineSimilarity(dim=1)
            loss = -torch.sum(cos(image_features, self.prompt_features))

            img_normed = F.normalize(image_features.unsqueeze(1), dim=2)
            prompt_normed = F.normalize(self.prompt_features.unsqueeze(0), dim=2)
            dists = img_normed.sub(prompt_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
            loss = dists.mean()
            
            if i % 20 == 0:
                print("{}: {}".format(i, loss.item()))
                self.display(self.current)

            loss.backward()
            self.optimizer.step()
            
            with torch.no_grad():
                self.current.clamp_(0.0, 1.0)
        
    def display(self, tensor):
      with torch.no_grad():
        pil_image = to_img(tensor.cpu())
        pil_image = pil_image.resize((self.width * self.scale, self.height * self.scale), Image.NEAREST)
        pil_image.save("out.png")
        display.display(display.Image("out.png"))

if __name__ == "__main__":
    pd = PixelDrawer()
    pd.init()
    pd.run()