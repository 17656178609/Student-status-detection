import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.yolo.utils.tal import dist2bbox, make_anchors
import torchvision.models as models
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()

        # [4, 256, 32, 32] => [32, 32, 32, 32]
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w

        # [32, 32, 32, 32] => [32, 32, 32, 1]
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

        # [32, 32, 32, 1] => [32, 32, 64, 1]
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))

        # [32, 32, 64, 1] => [32, 32, 32, 1]
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # [32, 32, 32, 32]*[32, 32, 32, 1]*[32, 32, 1, 32]  => [32, 32, 32, 32]
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        # [32, 32, 32, 32] => [32, 32, 32, 32]
        x2 = self.conv3x3(group_x)

        # [32, 32, 32, 32] => [32, 1, 32]
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))

        # [32, 32, 32, 32] => [32, 32, 1024]
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw

        # [32, 32, 32, 32] => [32, 1, 32]
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))

        # [32, 32, 32, 32] => [32, 32, 1024]
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw

        # [32, 1, 32]*[32, 32, 1024] + [32, 1, 32]*[32, 32, 1024] => [32, 1, 32, 32]
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        # [32, 32, 32, 32]*[32, 1, 32, 32] => [4, 256, 32, 32]
        output = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        return output
    
class EMA1(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA1, self).__init__()
        self.groups = 1
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()

        # [4, 256, 32, 32] => [32, 32, 32, 32]
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w

        # [32, 32, 32, 32] => [32, 32, 32, 1]
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

        # [32, 32, 32, 1] => [32, 32, 64, 1]
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))

        # [32, 32, 64, 1] => [32, 32, 32, 1]
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # [32, 32, 32, 32]*[32, 32, 32, 1]*[32, 32, 1, 32]  => [4, 256, 32, 32]
        output = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()).reshape(b, c, h, w)
        return output
    

class EMA2(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA2, self).__init__()
        self.groups = 1
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        x1 = self.conv3x3(x)
        x2 = self.conv3x3(x)
        # [4, 256, 32, 32] => [4, 1024, 256]
        x11 = self.softmax(self.agp(x1).reshape(b, c, -1).permute(0, 2, 1))

        # [4, 256, 32, 32] => [4, 256, 1024]
        x12 = x2.reshape(b, c, -1)  

        # [4, 256, 32, 32] => [4, 1024, 256]
        x21 = self.softmax(self.agp(x2).reshape(b, c, -1).permute(0, 2, 1))

        # [4, 256, 32, 32] => [4, 256, 1024]
        x22 = x1.reshape(b , c , -1)  # b*g, c//g, hw

        # [4, 1024, 256]*[4, 256, 1024] + [4, 1024, 256]*[4, 256, 1024] => [4, 1024, 32, 32]
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, -1, h, w)

        # [4, 256, 32, 32]*[4, 1024, 32, 32] => [4, 256, 32, 32]
        output = (x * weights.sigmoid()).reshape(b, c, h, w)
        return output
    
def main():
    input_data = torch.randn(4, 256, 32, 32)
    ema_model = EMA(channels=256, factor=8)
    ema_model1 = EMA1(channels=256, factor=8)
    ema_model2 = EMA2(channels=256, factor=8)
    output_data = ema_model(input_data)
    output_data1 = ema_model1(input_data)
    output_data2 = ema_model2(input_data)
    print("output_data:",output_data.shape)
    print("output_data1:",output_data1.shape)
    print("output_data2:",output_data2.shape)
if __name__ == '__main__':
    main()