from base64 import encode
from turtle import forward
from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        # p = self.dropout(p)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c + out_c, out_c)

        # self.dropout = nn.Dropout(p=0.5)


    def forward(self, inputs, skip):
        x = self.up(inputs)
        # concatenation on axis 1 bcs this is where the number of channels is located
        x = torch.cat([x, skip], axis=1)

        # adding an dropout layer - common reguralization technique
        # x = self.dropout(x)

        # print(x.shape)
        x = self.conv(x)

        return x


class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """Encoder """
        # input channels: 3 -> RGB image, 1 -> grayscale image
        # output channels: 64
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        

        """Bottleneck"""
        self.b = conv_block(512, 1024)
        
        """Decoder"""
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """Classifier"""
        # Classifier is used to output basically the segmentation map
        # We put input channel same as the output of the previous layer
        # and output channel equal to 1 since we want the binary mask 
        # In case, you want a multi-class Unet structure change 1 to the 
        # number of classes you want to have masks (in our case 3 since 
        # we have background, podocytes and gbm)
        self.outputs = nn.Conv2d(64, 3, kernel_size=1, padding=0)

    def forward(self, inputs):
        """Encoder"""
        # s -> skip connection
        # p -> pooling output
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """Bottleneck"""
        b = self.b(p4)
        #print(inputs.shape, s1.shape, s2.shape, s3.shape, s4.shape)
        #print(b.shape)

        """Decoder"""
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        #print(d1.shape, d2.shape, d3.shape, d4.shape)

        # print(d4.shape)
        outputs = self.outputs(d4)

        return outputs

        # Change to sigmoid if binary
        # Change to softmax if multiclass
        #return F.softmax(outputs, dim=1)


if __name__ == "__main__":

    # input size:
    #   batch size, channels, height, width
    x = torch.randn((1, 1, 512, 512))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet()
    model = model.to(device)

    x = x.to(device)
    #print(x.shape)
    y = model(x)
    
    # print(model)
    # print("\n\n-------------------------")
    # print("Output shape: ", y.shape)
