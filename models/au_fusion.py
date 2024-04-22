import torch
import torch.nn as nn


class AUFusion(nn.Module):
    def __init__(self, parallel, num_classes, in_features=9):
        super(AUFusion, self).__init__()
        self.linear = nn.Linear(in_features=in_features,
                                out_features=num_classes)
        self.layer_norm = nn.LayerNorm(in_features, eps=1e-6)
        self.layer_norm_List = nn.ModuleList([
            nn.LayerNorm(in_features, eps=1e-6)
            for _ in range(parallel)
        ])

    def forward(self, eyebrow, mouth, gcn):
        # eyebrow: [B, N, 160], gcn: [N, 9, 160]
        # features: [B, N, 9]
        features = torch.empty(eyebrow.size(0), eyebrow.size(1), 9).cuda()
        gcn = gcn.transpose(1, 2)
        for index in range(eyebrow.size(1)):
            temp = (eyebrow[:, index] @ gcn[index])
            features[:, index] = self.layer_norm_List[index](temp)
        return features


if __name__ == "__main__":
    eyebrow_test = torch.randn(1, 160)
    mouth_test = torch.randn(1, 160)
    gcn_test = torch.randn(9, 160)

    model = AUFusion(num_classes=5)
    print(model(eyebrow_test, mouth_test, gcn_test).shape)
