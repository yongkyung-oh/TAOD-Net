import torch
import torch.nn.functional as F
from torch import nn


# https://pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html?highlight=focal+loss#torchvision.ops.sigmoid_focal_loss
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


class FocalLoss(nn.Module):
    def __init__(self, 
                    alpha: float = 0.25,
                    gamma: float = 2,
                    reduction: str = "mean",
                ):
        super(FocalLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        return sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma, self.reduction)

    
class Focal_Multilabel(nn.Module):
    def __init__(self, 
                    alpha: list = [0.25]*5,
                    gamma: float = 2,
                    num_classes: int = 5,
                    reduction: str = "mean",
                ):
        super(Focal_Multilabel, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        total_loss = 0
        for idx in range(self.num_classes):
            if len(inputs[:, idx].shape) == 1:
                inputs_i = inputs[:, idx].reshape(-1, 1)
            if len(targets[:, idx].shape) == 1:
                targets_i = targets[:, idx].reshape(-1, 1)
            total_loss += sigmoid_focal_loss(inputs_i, targets_i, self.alpha[idx], self.gamma, self.reduction)
        return total_loss / self.num_classes
    

# https://github.com/Alibaba-MIIL/ASL
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

    def forward(self, inputs, targets):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        x, y = inputs, targets
        
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

        # return -self.loss.sum()
        return -self.loss.mean()
    
    
class ASL_Multilabel(nn.Module):
    def __init__(self, gamma_neg=[4]*5, gamma_pos=[1]*5, num_classes=5, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(ASL_Multilabel, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.num_classes = num_classes
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, inputs, targets):
        total_loss = 0
        for idx in range(self.num_classes):
            if len(inputs[:, idx].shape) == 1:
                x = inputs[:, idx].reshape(-1, 1)
            if len(targets[:, idx].shape) == 1:
                y = targets[:, idx].reshape(-1, 1)
                        
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
            if self.gamma_neg[idx] > 0 or self.gamma_pos[idx] > 0:
                if self.disable_torch_grad_focal_loss:
                    torch.set_grad_enabled(False)
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                              self.gamma_pos[idx] * self.targets + self.gamma_neg[idx] * self.anti_targets)
                if self.disable_torch_grad_focal_loss:
                    torch.set_grad_enabled(True)
                self.loss *= self.asymmetric_w

            total_loss += -self.loss.mean()
        return total_loss / self.num_classes


# https://discuss.pytorch.org/t/two-optimizers-for-one-model/11085/7
class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()