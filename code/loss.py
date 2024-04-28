import torch
import torch.nn as nn


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


import torch
from torch.nn.modules.loss import _Loss


class Criterion(nn.Module):
    def __init__(self, way=2, shot=5):
        super(Criterion, self).__init__()
        self.amount = way * shot
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, probs, target):  # (Q,C) (Q)
        target = target[self.amount:]
        # print(target)
        # target_onehot = torch.zeros_like(probs).to(probs.device)
        # target_onehot = target_onehot.scatter(1, target.reshape(-1, 1), 1)

        target_onehot = -1 * torch.ones_like(probs).to(probs.device)
        target_onehot = target_onehot.scatter(1, target.reshape(-1, 1), 1)
        # print(target_onehot)
        # probs = torch.sigmoid(probs)
        loss = torch.mean((probs - target_onehot) ** 2)
        #
        # loss = self.loss_func(probs, target_onehot)

        pred = torch.argmax(probs, dim=1)
        acc = torch.sum(target == pred).float() / target.shape[0]
        return loss, acc


if __name__ == '__main__':
    loss_func = Criterion()
    probs = torch.randn(5, 2)
    targets = torch.randint(2, (15, ))
    loss_func(probs, targets)


if __name__ == '__main__':
    loss_func = AsymmetricLossOptimized()
    x = torch.randn((4, ))
    y = torch.randint(2, (4, ))
    print(x, y)
    loss = loss_func(x, y)
    print(loss)