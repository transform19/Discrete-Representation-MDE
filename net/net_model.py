# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch as t

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        #self.hdc = hdc(512, 512)
        self.vq = VectorQuantizer(channel=512, n_res_block=1, n_res_channel=32, embed_dim=128, n_embed=2048, decay=0.99)
        #self.dehdc = dehdc(256, 256)
        self.conv = double_conv(256, 512)
        #self.arm1 = AttentionRefinementModule(512,512)
        self.up1 = up(1024, 256)
        self.arm2 = AttentionRefinementModule(256,256)
        self.up2 = up(512, 128)
        self.arm3 = AttentionRefinementModule(128,128)
        self.up3 = up(256, 64)
        self.arm4 = AttentionRefinementModule(64,64)
        self.up4 = up_without(128, 64)
        self.ffm = FeatureFusionModule(1, 512)
        self.outc1 = outconv(64, n_classes)
        self.outc2 = outconv(2, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #x5 = self.hdc(x5)
        loss, z, perplexity, _ = self.vq(x5)
        #z = self.dehdc(z)
        z =  self.conv(z)
        #d1 = self.arm1(z)
        #cx1 = F.interpolate(d1, size=x.size()[-2:], mode='bilinear', align_corners=True)
        z = self.up1(z, x4)
        d2 = self.arm2(z)
        cx2 = F.interpolate(d2, size=x.size()[-2:], mode='bilinear', align_corners=True)
        z = self.up2(z, x3)
        d3 = self.arm3(z)
        cx3 = F.interpolate(d3, size=x.size()[-2:], mode='bilinear', align_corners=True)
        z = self.up3(z, x2)
        d4 = self.arm4(z)
        cx4 = F.interpolate(d4, size=x.size()[-2:], mode='bilinear', align_corners=True)
        z = self.up4(z, x1)
        cx = t.cat([cx2, cx3, cx4], dim = 1)
        h = self.ffm(z, cx)
        x = self.outc1(z)
        f_in = t.cat([x,h], dim=1)
        f_out = self.outc2(f_in)

        return loss, t.sigmoid(f_out)
