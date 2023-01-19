import torch
from torch import nn
import torch.nn.functional as F
import math
from PIL import Image
from utils import convert_image
import sys
# import cv2


class Generator(nn.Module):

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        super(Generator, self).__init__()

        assert type(scaling_factor) is int and scaling_factor in [2, 4, 8]

        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='PReLu')

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])

        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)

        n_upsample_blocks = int(math.log2(scaling_factor))
        self.upsample_blocks = nn.Sequential(
            *[InterpolateUpsampleBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i
              in range(n_upsample_blocks)])

        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_imgs):
        output = self.conv_block1(lr_imgs)
        residual = output
        output = self.residual_blocks(output)
        output = self.conv_block2(output)
        output = output + residual
        output = self.upsample_blocks(output)
        sr_imgs = self.conv_block3(output)

        return sr_imgs


class Discriminator(nn.Module):

    def __init__(self, kernel_size=3, n_channels=64, n_blocks=8, fc_size=1024):
        global out_channels
        super(Discriminator, self).__init__()

        in_channels = 3

        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (n_channels if i is 0 else in_channels * 2) if i % 2 is 0 else in_channels
            conv_blocks.append(
                ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=1 if i % 2 is 0 else 2, batch_norm=i is not 0, activation='LeakyReLu'))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # адаптивный пулинг нужен, чтобы выход с последнего сверточного слоя всегда соответствовал одной и той же размерности
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, imgs):
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)

        return logit


class ConvolutionalBlock(nn.Module):
    #  Сверточный блок, который содержит сверточный слой, batch-norm слой и активацию

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh', 'relu'}

        layers = list()

        # добавляем сверточный слой
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        # батч-нормализация
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # Функция активации
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'relu':
            layers.append(nn.ReLU())

        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        output = self.conv_block(input)

        return output


class InterpolateUpsampleBlock(nn.Module):

    def __init__(self, n_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(InterpolateUpsampleBlock, self).__init__()

        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.conv(F.interpolate(input, scale_factor=2, mode="bilinear", align_corners=True)), 0.2,
                            True)


class SubPixelConvolutionalBlock(nn.Module):

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        super(SubPixelConvolutionalBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output


class ResidualBlock(nn.Module):
    #  residual block - два сверточных блока со skip connection через них
    def __init__(self, kernel_size=3, n_channels=64):
        super(ResidualBlock, self).__init__()

        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')

        # The second convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):
        residual = input
        output = self.conv_block1(input)
        output = self.conv_block2(output)
        output = output + residual

        return output


def process_photo(photo_path, save_path):
    n_blocks = 16
    upscale_factor = 4
    srgan_checkpoint = "SRGAN_16blocks_4x.pth"

    model = Generator(n_blocks=n_blocks, scaling_factor=upscale_factor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(srgan_checkpoint, map_location=torch.device(device)))

    model.to(device)
    model.eval()

    photo = Image.open(photo_path, mode="r").convert('RGB')
    # photo = cv2.imread(photo_path)
    # photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)


    photo_result = model(convert_image(photo, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    photo_result = torch.clamp(photo_result.squeeze(0).cpu().detach(), -1, 1)
    photo_result = convert_image(photo_result, source='[-1, 1]', target='pil')
    photo_result.save(save_path)



if __name__ == "__main__":
     _, photo_path, save_path = sys.argv
     photo_path = str(photo_path)
     save_path = str(save_path)
     process_photo(photo_path, save_path)

