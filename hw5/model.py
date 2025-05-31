import torch.nn.functional as F
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class FaceNet(nn.Module):
    def __init__(self, embedding_dim=512, pretrained='vggface2'):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained=pretrained, classify=False)
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)