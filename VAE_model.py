import torch
import torch.nn as nn

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
