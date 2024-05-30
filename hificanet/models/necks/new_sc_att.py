import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM_improved_Layer(nn.Module):
    def __init__(self, C, H, W, reduction=16, spatial_kernel=7):
        super(CBAM_improved_Layer, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(C, C // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // reduction, C, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=5, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.mlp2 = nn.Sequential(
            nn.Linear(W * H + C, (W * H + C) // reduction),
            nn.ReLU(inplace=True),
            nn.Linear((W * H + C) // reduction, W * H + C)
        )

    def forward(self, x):
        batch_size, C, H, W = x.size()

        max_out1 = self.mlp(self.max_pool(x))
        avg_out1 = self.mlp(self.avg_pool(x))
        channel_express = max_out1 + avg_out1

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)

        max_out_mlp = self.conv1(max_out)
        max_out_mlp = self.relu(max_out_mlp)
        max_out_mlp = F.interpolate(max_out_mlp, scale_factor=5, mode='bilinear', align_corners=False)

        avg_out_mlp = self.conv1(avg_out)
        avg_out_mlp = self.relu(avg_out_mlp)
        avg_out_mlp = F.interpolate(avg_out_mlp, scale_factor=5, mode='bilinear', align_corners=False)

        spatial_express = max_out_mlp + avg_out_mlp

        combined_feature = torch.cat((spatial_express.view(batch_size, -1),
                                      channel_express.view(batch_size, -1)), dim=1)

        output_feature = self.mlp2(combined_feature)

        output_feature_c = output_feature[:, :C].view(batch_size, C, 1, 1)
        output_feature_hw = output_feature[:, C:].view(batch_size, 1, H, W)

        output_feature_c = self.sigmoid(output_feature_c)
        output_feature_hw = self.sigmoid(output_feature_hw)

        x = x * output_feature_c + x * output_feature_hw

        return x
