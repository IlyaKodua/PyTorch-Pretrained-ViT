from pytorch_pretrained_vit.model import*
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate

model_name = 'B_16_imagenet1k'
model = ViT_interp_weights(model_name, pretrained=True)



model.interp_weigts((300,128), (1,128))



