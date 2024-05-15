import torch
import torch.nn as nn

class DualReconLoss(nn.Module):
    def __init__(self, lambda_fake=1.0):
        super(DualReconLoss, self).__init__()
        self.lambda_fake = lambda_fake

    def forward(self, recons, x, y):
        # y == 1 for real images, y == 0 for fake images
        loss_real = torch.tensor(0., device=x.device)
        loss_fake = torch.tensor(0., device=x.device)
      
        real_index = (y == 1).nonzero(as_tuple=True)[0]
        fake_index = (y == 0).nonzero(as_tuple=True)[0]

        if real_index.numel() > 0:
            real_x = x[real_index]
            real_recons = recons[real_index]
            loss_real = torch.mean(torch.abs(real_recons - real_x))

        if fake_index.numel() > 0:
            fake_x = x[fake_index]
            fake_recons = recons[fake_index]
            loss_fake = torch.mean(torch.abs(fake_recons - fake_x)) * self.lambda_fake
          
        return loss_real - loss_fake


class RealReconLoss(nn.Module):
    def __init__(self):
        super(RealReconLoss, self).__init__()

    def forward(self, recons, x, y):
        # y == 1 for real images, y == 0 for fake images
        loss_real = torch.tensor(0., device=x.device)

        real_index = (y == 1).nonzero(as_tuple=True)[0]

        if real_index.numel() > 0:
            real_x = x[real_index]
            real_recons = recons[real_index]
            loss_real = torch.mean(torch.abs(real_recons - real_x))

        return loss_real
