import  torch
import  torch.nn.functional as F
import  numpy as np
from    torch import nn

class amcloss(nn.Module):
    def __init__(self, geo_margin=0.5):
        super().__init__()
        self.geo_margin = geo_margin

    def forward(self, pre_fc, features):

        pre_class = F.softmax(features)
        pre_class = torch.argmax(pre_class, dim=1)
        half = int(len(pre_class) / 2)
        # a.eq(b):比较两个张量tensor中，每一个对应位置上元素是否相等–对应位置相等，就返回一个1；否则返回一个0.
        neighbor = pre_class[half:].eq(pre_class[:half])

        #？   Zi,Zj
        phi_emb = F.normalize(pre_fc, p=2, dim=1)
        inner = torch.sum(phi_emb[:half] * phi_emb[half:], dim=1)
        geo_desic = torch.clamp(inner, -1.0 + 1e-07, 1.0 - 1e-07)
        #计算反三角函数
        geo_desic = torch.acos(geo_desic)
        geo_desic2 = self.geo_margin - geo_desic
        zero = torch.zeros_like(geo_desic2)
        geo_desic2 = torch.max(geo_desic2, zero)
        geo_losses = torch.where(neighbor, torch.sqrt(geo_desic), torch.sqrt(geo_desic2))
        geo_loss = torch.mean(geo_losses)
        return geo_loss

