import numpy as np
import torch
import monai
import torch.nn as nn

class Monai_model(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, backbone = 'swinunetr', organ_embedding = None):
        # encoding: rand_embedding or word_embedding
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'swinunetr':
            self.backbone = monai.networks.nets.SwinUNETR(
                img_size=(96,96,96),
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=48,
                drop_rate=0,
                attn_drop_rate=0,
                dropout_path_rate=0,
            )
        elif backbone == 'segresnet':
            self.backbone = monai.networks.nets.SegResNet(
                blocks_down=[1,2,2,4],
                blocks_up=[1,1,1],
                init_filters=16,
                in_channels=in_channels,
                out_channels=out_channels,
                dropout_prob=0.2,
            )
        elif backbone == 'unet':
            self.backbone = monai.networks.nets.UNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=(16,32,64,128,256),
                strides=(2,2,2,2),
                num_res_units=2,
                norm=monai.networks.layers.Norm.BATCH,
            )
        elif backbone == 'unetpp':
            self.backbone = monai.networks.nets.BasicUNetPlusPlus(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                features=(32,32,64,128,256,32),
                dropout=0,
                upsample="deconv"
            )
        elif backbone == "vnet":
            self.backbone = monai.networks.nets.VNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
            )
        else:
            raise Exception('{} backbone is not implemented in curretn version'.format(backbone))

        self.register_buffer("organ_embedding",organ_embedding)
        #self.organ_embedding = organ_embedding
    

    def forward(self, x_in):
        return self.backbone(x_in)
