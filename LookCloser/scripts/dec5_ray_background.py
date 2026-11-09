"""Small frozen ray-space background, fitted strictly outside train subjects."""
import torch
from torch import nn


class RayBackgroundNetwork(nn.Module):
    def __init__(self,origin_levels=1,direction_levels=5,width=64):
        super().__init__();self.origin_levels=origin_levels;self.direction_levels=direction_levels
        features=6+6*(origin_levels+direction_levels)
        self.layers=nn.Sequential(nn.Linear(features,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,3),nn.Sigmoid())

    def forward(self,origins,directions):
        features=[origins,directions]
        for x,levels in [(origins,self.origin_levels),(directions,self.direction_levels)]:
            for k in range(levels):
                angle=x*(torch.pi*(2**k));features.extend([angle.sin(),angle.cos()])
        return self.layers(torch.cat(features,-1))
