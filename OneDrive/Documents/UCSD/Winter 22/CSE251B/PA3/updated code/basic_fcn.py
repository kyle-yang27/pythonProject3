import torch.nn as nn
from torchvision import models

   
class FCN(nn.Module):


    def __init__(self, n_class):
        super().__init__()

        
        self.n_class = n_class
        self.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3, dilation=0)
        self.bnd1    = nn.BatchNorm2d(64)
        self.conv2   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=3, dilation=0)
        self.bnd2    = nn.BatchNorm2d(128)
        self.conv3   = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=3, dilation=0)
        self.bnd3    = nn.BatchNorm2d(256)
     
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, dilation=0, output_padding=3)
        self.bn1     = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=0, output_padding=3)
        self.bn2     = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=0, output_padding=3)
        self.bn3     = nn.BatchNorm2d(64) #Removed conlvutional layers 4-5

        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)

    def forward(self, x):

        x1 = self.bnd1(self.relu(self.conv1(x)))
        # Complete the forward function for the rest of the encoder
        x2 = self.bnd2(self.relu(self.conv2(x1)))
        out_encoder = self.bnd3(self.relu(self.conv3(x2)))

        y1 = self.bn1(self.relu(self.deconv1(out_encoder)))    
        # Complete the forward function for the rest of the decoder
        y2 = self.bn2(self.relu(self.deconv2(y1)))     
        out_decoder = self.bn3(self.relu(self.deconv3(y2)))   
        
        score = self.classifier(out_decoder)                   

        return score  # size=(N, n_class, x.H/1, x.W/1)