########################################################################################################################
#                                  DEEP NEURAL NETWORKS FOR AERODYNAMIC LOW PREDICTION                                 #
#                                             Student: Albert García Plaza                                             #
#                                              Supervisor: Eduardo Alonso                                              #
#                                               INM363 Individual project                                              #
#                              MSc in Artificial Intelligence - City, University of London                             #
#                                                                                                                      #
# Code implementation based on:                                                                                        #
#   - ResNet:   Based on https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278              #
#               Based on https://github.com/usuyama/pytorch-unet                                                       #
#   - LeNet:    Based on https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320       #
#   - AlexNet:  Based on https://medium.com/@kushajreal/training-alexnet-with-tips-and-checks-on-how-to-train-cnns-    #
#                           practical-cnns-in-pytorch-1-61daa679c74a                                                   #
########################################################################################################################

import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    """
    Performs a double Conv2D-ReLU in a row pass:
    :param in_channels: number of channels of the input
    :param out_channels: number of channels of the output.
    :return: the result from the double (convolutional layer - ReLU activation) functions.

    PyTorch built-in functions explanation:
    - torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding): applies a 2D convolution over an input signal
      composed of several input planes.
        - in_channels: number of channels in the input array.
        - out_channels: number of channels produced by the convolution.
        - kernel_size: size of the square convolving kernel.
        - padding: pixels added to both sides of the input array.
      Mathematically, it performs the operation y(N_i, Cout_j)=b(Cout_j)+Sum_{k=0}^{Cin-1}{w(Cout_j,k)}<·>x(Ni,k), where
      y is the output, N_i the batch size, Cout_j the number of channels of the output, b the bias, Cin the number of
      channels of the input, x the input array, and <·> the cross-correlation operator.

    - torch.nn.ReLU(inplace): applies the rectified unit function element-wise.
        - inplace: if True, modifies the input directly, without allocating any additional output. Allows a dramatically
          decrease of the memory usage, but sometimes the operation may block the gradient propagation returning error.
      Mathematically, it performs the operation y(x)=max{0, x}, where y es the output, x the input.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


class ResNet(nn.Module):
    def __init__(self):
        """
        Declaration of the network layers for the UNet Encoder-Decoder neural network based on ResNet architecture.

        PyTorch built-in functions explanation:
        - torch.nn.Maxpool2d(kernel_size): applies 2D max pooling over an input signal composed of several input planes.
            - kernel_size: size of the square window to take the max value over

        - torch.nn.Linear(in_features, out_features): applies a linear transformation to the incoming data.
            - in_features: size of each input sample.
            - out_features: size of each output sample.
          Mathematically, it performs the operation y(x)=x·A+b, where y es the output, x the input, A the weights matrix,
          and b the biases array.

        - torch.nn.Upsample(scale_factor, mode, align_corners): upsample a given multi-channel bidimensional (2D) data.
            - scale_factor: multiplier for spatial size.
            - mode: the upsampling algorithm.
            - align_corners: if True, the corner pixels of the input and output tensors are aligned, and thus preserving
              the values at those pixels.
          Mathematically, it performs the operation y[Hout, Wout]=x[s·Hin, s·Win], where y is the input with dimensions
          (Hout, Wout), and x the input with dimensions (Hin, Win), being H the height and W the width.
        """
        super(ResNet, self).__init__()  # inherit attributes from parent PyTorch's nn.Module class

        # Declaration of the (double) convolutional layers
        self.conv1 = double_conv(in_channels=3, out_channels=64)
        self.conv2 = double_conv(in_channels=64, out_channels=128)
        self.conv3 = double_conv(in_channels=128, out_channels=256)
        self.conv4 = double_conv(in_channels=256, out_channels=512)

        # Declaration of the Maxpool layer
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # Declaration of the Upsample layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_odd = nn.Upsample(size=(75, 75), mode='bilinear', align_corners=True)

        # Declaration of the deconvolutional layers
        self.deconv1 = double_conv(in_channels=1024, out_channels=256)
        self.deconv2 = double_conv(in_channels=512, out_channels=128)
        self.deconv3 = double_conv(in_channels=256, out_channels=62)
        self.deconv4 = nn.Conv2d(in_channels=62, out_channels=2, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Performs the forward pass through all network layers.
        :param x: input array with shape [bs, in_channels, Hin, Win]
        :return: the output value after performing all network layers operations.
        """
        # Encoder: convolutional downsapling
        x_conv1 = self.conv1(x)  # output shape [bs, 64, Hin, Win]
        x = self.maxpool(x_conv1)    # output shape [bs, 64, Hin/2, Win/2]
        x_conv2 = self.conv2(x)  # output shape [bs, 128, Hin/2, Win/2]
        x = self.maxpool(x_conv2)  # output shape [bs, 128, Hin/4, Win/4]
        x_conv3 = self.conv3(x)  # output shape [bs, 256, Hin/4, Win/4]
        x = self.maxpool(x_conv3)  # output shape [bs, 256, Hin/8, Win/8]
        x_conv4 = self.conv4(x)  # output shape [bs, 512, Hin/8, Win/8]
        x = self.maxpool(x_conv4)  # output shape [bs, 512, Hin/16, Win/16]

        # Decoder: convolutional upsampling adding residual block from the downsampling process
        x = self.upsample_odd(x)  # output shape [bs, 256, Hin/8, Win/8]
        x = torch.cat([x, x_conv4], dim=1)  # output shape [bs, 1024, Hin/8, Win/8]
        x = self.deconv1(x)  # output shape [bs, 256, Hin/8, Win/8]
        x = self.upsample(x)  # output shape [bs, 128, Hin/4, Win/4]
        x = torch.cat([x, x_conv3], dim=1)  # output shape [bs, 512, Hin/4, Win/4]
        x = self.deconv2(x)  # output shape [bs, 128, Hin/4, Win/4]
        x = self.upsample(x)  # output shape [bs, 128, Hin/2, Win/2]
        x = torch.cat([x, x_conv2], dim=1)  # output shape [bs, 256, Hin/2, Win/2]
        x = self.deconv3(x)  # output shape [bs, 62, Hin/2, Win/2]
        x = self.upsample(x)  # output shape [bs, 62, Hin, Win]

        return self.deconv4(x)  # output shape [bs, 2, Hin, Win]


class LeNet(nn.Module):
    def __init__(self):
        """
        Declaration of the network layers for the UNet Encoder-Decoder neural network based on LeNet architecture.

        PyTorch built-in functions explanation:
        - torch.nn.Tanh(): applies the hyperbolic tangent of the elements of input.

        - torch.nn.AvgPool2d(kernel_size): applies 2D average pooling over an input
          signal composed of several input planes.
            - kernel_size: size of the square window to take the averaged value over.
        """
        super(LeNet, self).__init__()  # inherit attributes from parent PyTorch's nn.Module class

        # Declaration of the convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),
        )

        # Declaration of the deconvolutional layers
        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=120, out_channels=16, kernel_size=5, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=16, out_channels=6, kernel_size=5, stride=1),
            nn.Upsample(size=(600, 600), mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=6, out_channels=2, kernel_size=5, stride=1, padding=2)
        )


    def forward(self, x):
        """
        Performs the forward pass through all network layers.
        :param x: input array with shape [bs, in_channels, Hin, Win]
        :return: the output value after performing all network layers operations.
        """
        # Encoder: convolutional downsapling
        x = self.conv(x)

        # Decoder: convolutional upsampling adding residual block from the downsampling process
        return self.deconv(x)
