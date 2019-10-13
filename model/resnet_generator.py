import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm_module import *
from .mask_regression import *
from .sync_batchnorm import SynchronizedBatchNorm2d


class ResnetGenerator64(nn.Module):
    def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3):
        super(ResnetGenerator64, self).__init__()
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, 128)

        num_w = 128+128
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4*4*16*ch))

        self.res2 = ResBlock(ch*16, ch*8, upsample=True, num_w=num_w)
        self.res3 = ResBlock(ch*8, ch*4, upsample=True, num_w=num_w)
        self.res4 = ResBlock(ch*4, ch*2, upsample=True, num_w=num_w)
        self.res5 = ResBlock(ch*2, ch*1, upsample=True, num_w=num_w)
        self.final = nn.Sequential(SynchronizedBatchNorm2d(ch),
                                   nn.ReLU(),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())

        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        self.mask_regress = MaskRegressNet(num_w)

        self.init_parameter()

    def forward(self, z, bbox, z_im=None, y=None):
        b, o = z.size(0), z.size(1)

        label_embedding = self.label_embedding(y)

        z = z.view(b * o, -1)
        label_embedding = label_embedding.view(b * o, -1)

        latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)

        w = self.mapping(latent_vector.view(b * o, -1))
        bbox = self.mask_regress(w, bbox)

        if z_im is None:
            z_im = torch.randn((b, 128), device=z.device)

        # 4x4
        x = self.fc(z_im).view(b, -1, 4, 4)
        # 8x8
        x = self.res2(x, w, bbox)
        # 16x16
        x = self.res3(x, w, bbox)
        # 32x32
        x = self.res4(x, w, bbox)
        # 64x64
        x = self.res5(x, w, bbox)
        x = self.final(x)
        return x

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)


class ResnetGenerator128(nn.Module):
    def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3):
        super(ResnetGenerator128, self).__init__()
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, 180)

        num_w = 128+180
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4*4*16*ch))

        self.res1 = ResBlock(ch*16, ch*16, upsample=True, num_w=num_w)
        self.res2 = ResBlock(ch*16, ch*8, upsample=True, num_w=num_w)
        self.res3 = ResBlock(ch*8, ch*4, upsample=True, num_w=num_w)
        self.res4 = ResBlock(ch*4, ch*2, upsample=True, num_w=num_w)
        self.res5 = ResBlock(ch*2, ch*1, upsample=True, num_w=num_w)
        self.final = nn.Sequential(SynchronizedBatchNorm2d(ch),
                                   nn.ReLU(),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())

        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        self.mask_regress = MaskRegressNet(num_w)
        self.init_parameter()

    def forward(self, z, bbox, z_im=None, y=None):
        b, o = z.size(0), z.size(1)
        if y.dim() == 3:
            _, _, num_label = y.size()
            label_embedding = []
            for idx in range(num_label):
                label_embedding.append(self.label_embedding[idx](y[:, :, idx]))
            label_embedding = torch.cat(label_embedding, dim=-1)
        else:
            label_embedding = self.label_embedding(y)

        z = z.view(b * o, -1)
        label_embedding = label_embedding.view(b * o, -1)

        latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)

        w = self.mapping(latent_vector.view(b * o, -1))
        # preprocess bbox
        bbox = self.mask_regress(w, bbox)

        if z_im is None:
            z_im = torch.randn((b, 128), device=z.device)

        # 4x4
        x = self.fc(z_im).view(b, -1, 4, 4)
        # 8x8
        x = self.res1(x, w, bbox)
        # 16x16
        x = self.res2(x, w, bbox)
        # 32x32
        x = self.res3(x, w, bbox)
        # 64x64
        x = self.res4(x, w, bbox)
        # 128x128
        x = self.res5(x, w, bbox)
        # to RGB
        x = self.final(x)
        return x

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1, upsample=False, num_w=128):
        super(ResBlock, self).__init__()
        self.upsample = upsample
        self.h_ch = h_ch if h_ch else out_ch
        self.conv1 = conv2d(in_ch, self.h_ch, ksize, pad=pad)
        self.conv2 = conv2d(self.h_ch, out_ch, ksize, pad=pad)
        self.b1 = SpatialAdaptiveSynBatchNorm2d(in_ch, num_w=num_w)
        self.b2 = SpatialAdaptiveSynBatchNorm2d(self.h_ch, num_w=num_w)
        self.learnable_sc = in_ch != out_ch or upsample
        if self.learnable_sc:
            self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
        self.activation = nn.ReLU()

    def residual(self, in_feat, w, bbox):
        x = in_feat
        x = self.b1(x, w, bbox)
        x = self.activation(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.b2(x, w, bbox)
        x = self.activation(x)
        x = self.conv2(x)
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            if self.upsample:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.c_sc(x)
        return x

    def forward(self, in_feat, w, bbox):
        return self.residual(in_feat, w, bbox) + self.shortcut(in_feat)


def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv
