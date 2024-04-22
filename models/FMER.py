import torch
import torch.nn as nn
from .graph_model import GraphLearningModel
# from .au_gcn import GCN
from .npaths_au_gcn import GCN
from .au_fusion import AUFusion
import torch.nn.functional as F


class FMER(nn.Module):
    def __init__(self, adj_matrix, num_classes, parallel,
                 device, hidden_features=80):
        super(FMER, self).__init__()
        self.graph = GraphLearningModel(parallel=parallel)
        self.au_gcn = GCN(parallel=parallel,
                          adj_matrix=adj_matrix,
                          hidden_features=hidden_features)
        self.au_fusion = AUFusion(parallel=parallel, num_classes=num_classes)

        # Used to train the embedding
        # 其实这个au_seq是没有被学习的，其只是用来学习au_gcn当中的embedding
        # 这个au_seq只是用来生成节点特征的
        self.au_seq = torch.arange(9).to(device)
        # self.au_seq = torch.LongTensor([i for i in range(9)]).cuda()
        self.fusion_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(9 * parallel),
            nn.Linear(9 * parallel, num_classes)
            # nn.Dropout(0.5),
            # nn.Linear(9, 4)
        )  # 这里削减了一层全连接层
        # self.fusion_layer = nn.Sequential(
        #     nn.LayerNorm(160),
        #     nn.Linear(160, 9),
        #     nn.ReLU(inplace=True),
        #     nn.Flatten(start_dim=1),
        #     nn.LayerNorm(9 * parallel),
        #     nn.Linear(9 * parallel, num_classes)
        # )
        self.test = nn.Parameter(torch.Tensor(9, 9))

    def forward(self, patches):
        batch_size = patches.size(0)

        # Node learning and edge learning
        # Shape of patches: (batch_size, N, 30, 7, 7)
        # 这里eyebrow和mouth张量的形状均为[B, N, 160]
        # 这里eyebrow张量的形状为[B, N, 160]，mouth为空
        eyebrow, mouth = self.graph(patches)

        # Training the GCN
        # Shape of au_seq: (9)
        # Shape of gcn_output: (N, 9, 160)
        gcn_output = self.au_gcn(self.au_seq)

        # Fuse the graph learning and GCN
        # 这里的output形状为[B, N, 9]
        fusion_output = self.au_fusion(eyebrow, mouth, gcn_output)
        fusion_output = self.fusion_layer(fusion_output)

        # return F.log_softmax(fusion_output, dim=1)
        return F.log_softmax(fusion_output, dim=1)


if __name__ == "__main__":
    test_tensor = torch.rand(1, 30, 7, 7)
    adj_matrix = torch.rand(9, 9)
    model = FMER(adj_matrix=adj_matrix,
                 num_classes=5,
                 device="cpu")

    print(model(test_tensor).shape)