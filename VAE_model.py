import torch
import torch.nn as nn
import torch.nn.functional as F

class AE(torch.nn.Module):
  def __init__(self):
    super(AE, self).__init__()
    self.encoder = torch.nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=(2, 2), stride=(2),padding=1),
        torch.nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2),padding=1),
        
        nn.Conv2d(32, 128, kernel_size=(2, 2), stride=(1)),
        torch.nn.ReLU(),
        nn.MaxPool2d(kernel_size=(1, 1), stride=(1)),
        
        nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1)),
        torch.nn.ReLU(),

        nn.MaxPool2d(kernel_size=(1, 1), stride=(1))
      )

    self.decoder = torch.nn.Sequential(        
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1),padding=1),
        torch.nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(128, 32, kernel_size=(2, 2), stride=(1, 1)),
        torch.nn.ReLU(),
        nn.Upsample(scale_factor=1, mode='nearest'),
        nn.Conv2d(32, 3, kernel_size=(4, 4), stride=(1, 1))
    )

  def forward(self, x):
    latent = self.encoder(x)
    x_out = self.decoder(latent)
    return x_out


class InvBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, padding, stride=1):
        super(InvBottleneck, self).__init__()
        self.invconv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=1, bias=False)
        self.invbn1 = nn.BatchNorm2d(planes)
        self.invconv2 = nn.ConvTranspose2d(planes, planes, kernel_size=4,
                               stride=stride, padding=padding, bias=False)
        self.invbn2 = nn.BatchNorm2d(planes)
        self.invconv3 = nn.ConvTranspose2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.invbn3 = nn.BatchNorm2d(self.expansion*planes)

        #needed to take care of broken res' due to deconv!
        self.shortcut = nn.Sequential()
        if stride == 1:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_planes, self.expansion*planes,
                kernel_size=4, stride=1, padding=2, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_planes, self.expansion*planes,
                kernel_size=4, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.invbn1(self.invconv1(x)))
        out = F.relu(self.invbn2(self.invconv2(out)))
        out = self.invbn3(self.invconv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block,invblock, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 16, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 16, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 16, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, 16, num_blocks[4], stride=2)
        self.layer6 = self._make_layer(block, 128, num_blocks[4], stride=2)
        #vae-bottleneck
        self.linear = nn.Linear(2_048, 4_000)
        self.invlinear = nn.Linear(4_000, 2_048)
        
        self.invlayer6 = self._make_inv_layer(invblock, 128, num_blocks[4], stride=2)
        self.invlayer5 = self._make_inv_layer(invblock, 16, num_blocks[4], stride=2)
        self.invlayer4 = self._make_inv_layer(invblock, 16, num_blocks[4], stride=2)
        self.invlayer3 = self._make_inv_layer(invblock, 16, num_blocks[4], stride=2)
        self.invlayer2 = self._make_inv_layer(invblock, 16, num_blocks[4], stride=2)
        self.invlayer1 = self._make_layer(block, 16, num_blocks[4], stride=1)
        self.invconv1 = nn.Conv2d(64, 3, kernel_size=3,stride=1, padding=1, bias=False)
    #we need a new make layer function to account for padding in deconv layers
    def _make_inv_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        padding=0
        for idx, stride in enumerate(strides):
            if idx==1: padding=2
            layers.append(block(self.in_planes, planes, padding, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)    

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.shape)
        out = self.layer1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        #print(out.shape)
        out = self.layer5(out)
        #print(out.shape)
        out = self.layer6(out)
        #print(out.shape)
        out = F.avg_pool2d(out, 2)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.linear(out)
        #print(out.shape)
        out = self.invlinear(out)
        #print(out.shape)
        out = out.reshape(-1, 512, 2, 2)
        #print(out.shape)
        out = F.upsample(out, scale_factor=2, mode='nearest')
        #print(out.shape,'here')
        out = self.invlayer6(out)
        #print(out.shape)
        out = self.invlayer5(out)
        #print(out.shape)
        out = self.invlayer4(out)
        #print(out.shape)
        out = self.invlayer3(out)
        # print(out.shape)
        out = self.invlayer2(out)
        # print(out.shape)
        out = self.invlayer1(out)
        # print(out.shape)
        out = self.invconv1(out)
        # print(out.shape)
        return out