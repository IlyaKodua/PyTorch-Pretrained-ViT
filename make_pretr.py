from pytorch_pretrained_vit.model import*
import torch


model_name = 'B_16_imagenet1k'
model = ViT_interp_weights(model_name, pretrained=True)


model.re_dim_matrix((128,128), (1,128))


