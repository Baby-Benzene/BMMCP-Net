import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# =========================================================================
# 0. 基础网络库
# =========================================================================
class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, d=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SEWeightModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_map = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv1(spatial_map))


class CARAFE(nn.Module):
    def __init__(self, channels, scale=2, k_up=5):
        super().__init__()
        self.scale = scale
        self.k_up = k_up
        self.comp = nn.Conv2d(channels, channels // 4, 1)
        self.encoder = nn.Conv2d(channels // 4, (scale ** 2) * (k_up ** 2), kernel_size=3, padding=1)
        self.pix_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        N, C, H, W = x.shape
        kernel = self.comp(x)
        kernel = self.encoder(kernel)
        kernel = self.pix_shuffle(kernel)
        kernel = F.softmax(kernel, dim=1)

        x_up = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x_unfold = F.unfold(x_up, kernel_size=self.k_up, padding=self.k_up // 2)
        x_unfold = x_unfold.view(N, C, self.k_up ** 2, H * self.scale, W * self.scale)
        kernel = kernel.unsqueeze(1)
        out = (x_unfold * kernel).sum(dim=2)
        return out


class Pos(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)

    def forward(self, x):
        re = torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        fusion = self.conv2(x)
        return self.sigmoid(self.conv1(re)) * fusion + fusion


class SupervisedAttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cls = nn.Conv2d(channels, 1, 1)
        self.conv_context = nn.Sequential(nn.Conv2d(2, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.conv_out = ConvBNReLU(channels, channels)

    def forward(self, x):
        mask = self.cls(x)
        fg = torch.sigmoid(mask)
        bg = 1.0 - fg
        return self.conv_out(x * self.conv_context(torch.cat([fg, bg], dim=1))), mask


class Decoder(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.sam4 = SupervisedAttentionModule(channels)
        self.sam3 = SupervisedAttentionModule(channels)
        self.sam2 = SupervisedAttentionModule(channels)

        self.up4 = CARAFE(channels)
        self.up3 = CARAFE(channels)
        self.up2 = CARAFE(channels)

        self.conv3 = ConvBNReLU(channels, channels)
        self.conv2 = ConvBNReLU(channels, channels)
        self.conv1 = ConvBNReLU(channels, channels)

    def forward(self, c1, c2, c3, c4):
        p4, m4 = self.sam4(c4)
        p3 = self.conv3(c3 + self.up4(p4))
        p3, m3 = self.sam3(p3)
        p2 = self.conv2(c2 + self.up3(p3))
        p2, m2 = self.sam2(p2)
        feat_out = self.conv1(c1 + self.up2(p2))
        return feat_out, m2, m3, m4


# =========================================================================
# 1. 多方向轮廓解耦与打乱融合模块 (DSFM)
# =========================================================================
class DirectionalShuffleFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super(DirectionalShuffleFusionModule, self).__init__()
        self.groups = groups
        hidden_dim = out_channels

        self.project = nn.Conv2d(in_channels, hidden_dim, kernel_size=1) if in_channels != hidden_dim else nn.Identity()

        self.conv_0 = nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=(1, 5), padding=(0, 2))
        self.conv_90 = nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=(5, 1), padding=(2, 0))
        self.conv_45 = nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=(5, 1), padding=(2, 0))
        self.conv_135 = nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=(1, 5), padding=(0, 2))

    def _channel_shuffle(self, x, groups):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        return torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)

    def _h_transform(self, x):
        shape = x.size()
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        return x.reshape(shape[0], shape[1], shape[2], shape[2] + shape[3] - 1)

    def _inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3] + 1)
        return x[..., 0: shape[3] - shape[2] + 1]

    def _v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        return x.reshape(shape[0], shape[1], shape[2], shape[2] + shape[3] - 1).permute(0, 1, 3, 2)

    def _inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3] + 1)
        return x[..., 0: shape[3] - shape[2] + 1].permute(0, 1, 3, 2)

    def forward(self, x):
        identity = self.project(x)
        feat_0 = self.conv_0(identity)
        feat_90 = self.conv_90(identity)
        feat_45 = self._inv_h_transform(self.conv_45(self._h_transform(identity)))
        feat_135 = self._inv_v_transform(self.conv_135(self._v_transform(identity)))

        out = torch.cat((feat_0, feat_90, feat_45, feat_135), dim=1)
        out = self._channel_shuffle(out, self.groups)
        return identity + out


# =========================================================================
# 2. 通用参数量相当的现代主干封装 (PVT-v2 / ConvNeXt-V2 / MiT)
# =========================================================================
class TimmEncoder(nn.Module):

    def __init__(self, model_name='pvt_v2_b1', pretrained=True):
        super().__init__()
        # 提取 4 个阶段下采样特征 (1/4, 1/8, 1/16, 1/32)
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        # 获取 4 个阶段对应的实际输出通道数
        dummy = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            feats = self.model(dummy)
        self.out_channels = [f.shape[1] for f in feats]

    def forward(self, x):
        return self.model(x)


# =========================================================================
# 3. StripPoolingFA (SPFA)
# =========================================================================
class StripPoolingFA(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.convs = nn.ModuleList([ConvBNReLU(c, out_channels) for c in in_channels])
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.pool_v = nn.AdaptiveAvgPool2d((None, 1))
        # 动态适配最高层特征通道
        self.conv_shuffle = ConvBNReLU(in_channels[-1], in_channels[-1], k=1, p=0)

    def forward(self, features):
        x4 = features[-1]
        _, _, H, W = x4.shape
        sh = self.pool_h(x4).expand(-1, -1, H, W)
        sv = self.pool_v(x4).expand(-1, -1, H, W)
        gl = self.conv_shuffle(sh + sv)

        enhanced_features = list(features[:-1]) + [x4 + gl]
        return [self.convs[i](feat) for i, feat in enumerate(enhanced_features)]


# =========================================================================
# 4.  (TFAM)
# =========================================================================
class TFAM(nn.Module):
    def __init__(self, in_d, out_d, use_dsfm=True):
        super(TFAM, self).__init__()
        self.use_dsfm = use_dsfm

        self.diff_conv = ConvBNReLU(in_d * 2, out_d, k=1, p=0)

        self.branch1 = ConvBNReLU(out_d, out_d, d=7, p=7)
        self.branch2 = ConvBNReLU(out_d, out_d, d=5, p=5)
        self.branch3 = ConvBNReLU(out_d, out_d, d=3, p=3)
        self.branch4 = ConvBNReLU(out_d, out_d)

        if self.use_dsfm:
            self.dsfm = DirectionalShuffleFusionModule(in_channels=out_d, out_channels=out_d)

        self.cls = nn.Conv2d(out_d, 1, kernel_size=1)
        self.conv_context = nn.Sequential(
            nn.Conv2d(2, out_d, kernel_size=1), nn.BatchNorm2d(out_d), nn.ReLU(inplace=True)
        )
        self.convreduce = nn.Sequential(
            nn.Conv2d(out_d * 2, out_d * 2, 1, padding=0), nn.BatchNorm2d(out_d * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_d * 2, out_d, 3, padding=1), nn.BatchNorm2d(out_d),
            nn.LeakyReLU(inplace=True),
        )
        self.sa = Pos(out_d)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_d, out_d, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_d), nn.ReLU(inplace=True),
            nn.Conv2d(out_d, out_d, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(out_d), nn.ReLU(inplace=True)
        )

        self.channel_attention = SEWeightModule(out_d, reduction=16)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x1, x2):
        base_diff = self.diff_conv(torch.cat([x1, x2], dim=1))

        t1 = self.branch1(base_diff)
        t2 = self.branch2(t1)
        t3 = self.branch3(t2)
        t4 = self.branch4(t3)
        d = base_diff + t4

        if self.use_dsfm:
            d = self.dsfm(d)

        mask = self.cls(d)
        mask_f = torch.sigmoid(mask)
        mask_b = 1.0 - mask_f

        context = self.conv_context(torch.cat([mask_f, mask_b], dim=1))

        diff_x = torch.abs(x1 - x2)
        c = torch.cat((diff_x, d), dim=1)
        c = self.convreduce(c)
        c = self.sa(c)
        out = self.conv2(c.mul(context) + c)

        att = self.channel_attention(out)
        att = self.spatial_attention(att)
        out = out + att

        return out


# =========================================================================
# 5.  (DualGate)
# =========================================================================
class DualGate(nn.Module):
    def __init__(self, in_c=128, out_c=64):
        super().__init__()
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_c, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, out_c // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c // 4, out_c, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feat_deep, sf_diff):
        s_mask = self.spatial_gate(feat_deep)
        c_mask = self.channel_gate(feat_deep)
        filtered_sf = sf_diff * (1.0 + s_mask) * c_mask
        return filtered_sf


# =========================================================================
# 6.  (BaseNet)
# =========================================================================
class BaseNet(nn.Module):
    def __init__(self, backbone='pvt_v2_b1', pretrained=True):
        super().__init__()
        # 1. 采用与原模型参数相近的主干
        self.encoder = TimmEncoder(model_name=backbone, pretrained=pretrained)

        # 2. 动态自适应映射到统一的 128 通道
        self.nfa = StripPoolingFA(in_channels=self.encoder.out_channels, out_channels=128)

        # 3. 时序差分增强
        self.tfam1 = TFAM(128, 128, use_dsfm=True)
        self.tfam2 = TFAM(128, 128, use_dsfm=True)
        self.tfam3 = TFAM(128, 128, use_dsfm=False)
        self.tfam4 = TFAM(128, 128, use_dsfm=False)

        # 4. 解码器
        self.decoder = Decoder(128)

        # 5. 浅层辅助流
        self.shallow_feat = nn.Sequential(
            ConvBNReLU(3, 32, k=3, s=1, p=1),
            ConvBNReLU(32, 64, k=3, s=2, p=1)
        )
        self.shallow_diff_conv = ConvBNReLU(64, 64, k=3, p=1)
        self.gate_conv = DualGate(in_c=128, out_c=64)

        # 6. 最终融合头
        self.final_fuse = nn.Sequential(
            ConvBNReLU(128 + 64, 64, k=3, p=1),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x1, x2):
        # 浅层差分
        sf_diff = torch.abs(self.shallow_feat(x1) - self.shallow_feat(x2))
        sf_diff = self.shallow_diff_conv(sf_diff)

        # 主干特征提取与通道投影
        fx1 = self.nfa(self.encoder(x1))
        fx2 = self.nfa(self.encoder(x2))

        # TFAM 差分强化
        c1 = self.tfam1(fx1[0], fx2[0])
        c2 = self.tfam2(fx1[1], fx2[1])
        c3 = self.tfam3(fx1[2], fx2[2])
        c4 = self.tfam4(fx1[3], fx2[3])

        # 解码
        feat_out, m2, m3, m4 = self.decoder(c1, c2, c3, c4)

        # 门控滤波融合
        feat_half = F.interpolate(feat_out, scale_factor=2, mode='bilinear', align_corners=False)
        filtered_sf = self.gate_conv(feat_half, sf_diff)

        out_fused = torch.cat([feat_half, filtered_sf], dim=1)
        out_final = self.final_fuse(out_fused)

        # 分辨率复原
        out_final = F.interpolate(out_final, scale_factor=2, mode='bilinear', align_corners=False)
        m2 = F.interpolate(m2, scale_factor=8, mode='bilinear', align_corners=False)
        m3 = F.interpolate(m3, scale_factor=16, mode='bilinear', align_corners=False)
        m4 = F.interpolate(m4, scale_factor=32, mode='bilinear', align_corners=False)

        return out_final, m2, m3, m4

