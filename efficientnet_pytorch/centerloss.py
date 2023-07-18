import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=16):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        #得到一个正态分布的tensor
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(device))


    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """

        batch_size = x.size(0)
        #.t()：得到转置矩阵
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        #在原对象基础上进行修改，即把改变之后的变量再赋给原来的变量，inputs的值变成了改变之后的值，即x
        # x.addmm_(1,-2,inputs,inputs_t)： x = 1 × x - 2 ×（inputs @ inputs_t）
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        classes = classes.to(device)
        #unsqueeze：添加一个维度
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        #eq：比较Tensor是否相等
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        #clamp：将张量元素限制在指定区间
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss