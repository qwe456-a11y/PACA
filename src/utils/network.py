import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.fc(x)
        return x * attention


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.channel_attention = SEBlock(in_channels, reduction)
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        x = self.channel_attention(x)
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        return x * spatial_attention
    
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.conv12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.conv18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.out_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv6(x))
        x3 = self.relu(self.conv12(x))
        x4 = self.relu(self.conv18(x))
        x5 = self.relu(self.pool_conv(self.pool(x)))
        x5 = nn.functional.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.out_conv(x)
        return x


class FIR_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=6):
        super(FIR_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Attention3 = CBAM(256)

        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Attention4 = SEBlock(512)

        # 用 ASPP 替换普通卷积
        self.ASPP = ASPP(in_channels=512, out_channels=512) 

        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128) 
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = self.Attention3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = self.Attention4(x4)

        # 用 ASPP 处理 x4
        x4 = self.ASPP(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # Decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)  # Fix: this should match expected input channels for Up_conv3

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, padding=1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)
    
# 瓶颈残差块（Bottleneck Residual Block）
class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(BottleneckResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # 添加跳跃连接
        return out

# 深度残差块（Deep Residual Block）增加更多层
class DeepResidualBlock(nn.Module):
    def __init__(self, channels):
        super(DeepResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # 添加跳跃连接
        return out

# 改进后的生成器
class Generator(nn.Module):
    def __init__(self, feature_extractor):
        super(Generator, self).__init__()
        self.feature_extractor = feature_extractor  

        # 前景和背景增强模块
        self.tensor1_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.tensor2_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # 图像优化模块（加入更复杂的残差块）
        self.refinement_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            DeepResidualBlock(64),  # 添加更多层的残差块
            DeepResidualBlock(64),
            BottleneckResidualBlock(64, 128, 64),  # 引入瓶颈残差块以提高特征提取能力
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # 使用Sigmoid将输出值限制在[0,1]范围内
        )

    # def forward(self, imgs_cut, texture_img, masks):

    #     mask = masks.float()  # mask：物体部分为白色（1），背景为黑色（0）

    #     # 提取特征
    #     features = self.feature_extractor(imgs_cut)
    #     tensor1 = self.tensor1_conv(features[:, 0:3, :, :]) * mask
    #     tensor2 = self.tensor2_conv(features[:, 3:6, :, :]) * mask

    #     # 融合前景和背景
    #     tensor3 = torch.clamp(texture_img * tensor1 + tensor2, min=0, max=1)

    #     # 细化生成图像
    #     refined_img = self.refinement_conv(tensor3)  # 通过深层残差块优化
    #     refined_img = refined_img * mask  # 屏蔽背景区域
    #     refined_img = torch.clamp(refined_img + tensor3, 0, 1)  # 残差加回并限制输出范围
    #     return refined_img, tensor1, tensor2
    
    def forward(self, imgs_cut, texture_img, masks, tensor1=None, tensor2=None):

        mask = masks.float()  # mask：物体部分为白色（1），背景为黑色（0）

        # 如果 tensor1 和 tensor2 没有提供，则计算它们
        if tensor1 is None or tensor2 is None:
            # 提取特征
            features = self.feature_extractor(imgs_cut)
            tensor1 = self.tensor1_conv(features[:, 0:3, :, :]) * mask
            tensor2 = self.tensor2_conv(features[:, 3:6, :, :]) * mask

        # 融合前景和背景
        tensor3 = torch.clamp(texture_img * tensor1 + tensor2, min=0, max=1)

        # 细化生成图像
        refined_img = self.refinement_conv(tensor3)  # 通过深层残差块优化
        refined_img = refined_img * mask  # 屏蔽背景区域
        refined_img = torch.clamp(refined_img + tensor3, 0, 1)  # 残差加回并限制输出范围
        return refined_img, tensor1, tensor2

# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),  # Apply ReLU here
#             nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(channels)
#         )

#     def forward(self, x):
#         residual = x
#         out = self.block(x)
#         out += residual  # Add skip connection
#         return out  # No ReLU here

# 定义生成器
# class Generator(nn.Module):
#     def __init__(self, feature_extractor):
#         super(Generator, self).__init__()
#         self.feature_extractor = feature_extractor  # AttU_Net

#         # 前景和背景增强模块
#         self.tensor1_conv = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()
#         )
#         self.tensor2_conv = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()
#         )

#         # 图像优化模块（加入残差块）
#         self.refinement_conv = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             ResidualBlock(64),
#             ResidualBlock(64),
#             nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()  # Add Sigmoid here to constrain output
#         )

#     def forward(self, imgs_cut, texture_img, masks):
#         # 提取特征
#         features = self.feature_extractor(imgs_cut)

#         mask = masks.float()  # mask：物体部分为白色（1），背景为黑色（0）

#         # 计算前景和背景
#         tensor1 = self.tensor1_conv(features[:, 0:3, :, :]) * mask
#         tensor2 = self.tensor2_conv(features[:, 3:6, :, :]) * mask

#         # 融合前景和背景
#         tensor3=torch.clamp(texture_img*tensor1+tensor2, min=0, max=1)

#         # 细化生成图像
#         refined_img = self.refinement_conv(tensor3)  # 残差优化
#         refined_img = refined_img * mask # 屏蔽背景区域
#         refined_img = torch.clamp(refined_img + tensor3, 0, 1)  # 残差加回并限制范围

#         return refined_img, tensor1, tensor2
