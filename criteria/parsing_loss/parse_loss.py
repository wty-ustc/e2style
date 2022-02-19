import torch
from torch import nn
from criteria.parsing_loss.unet import unet
from configs.paths_config import model_paths

class ParseLoss(nn.Module):
    def __init__(self):
        super(ParseLoss, self).__init__()
        print('Loading UNet')
        self.parsenet = unet()
        self.parsenet.load_state_dict(torch.load(model_paths['parsing_net']))
        self.cosloss = torch.nn.CosineEmbeddingLoss()
        self.parsenet.eval()

    def extract_feats(self, x):
        x_feats = self.parsenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        cos_target = torch.ones((n_samples, 1)).float().cuda()
        loss = 0
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        for i in range(5):
            y_feat_detached = y_feats[i].detach()
            loss += self.cosloss(y_feat_detached, y_hat_feats[i], cos_target)
        
        return loss

