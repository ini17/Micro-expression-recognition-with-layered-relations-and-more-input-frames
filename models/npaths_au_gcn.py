import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features,
                 bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features,
                                                out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter("bias", None)
        self.init_parameters()

    def init_parameters(self):
        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, x):
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias
        return x


class GCN(nn.Module):
    def __init__(self, parallel, adj_matrix, hidden_features: int = 80,
                 num_embeddings: int = 9, in_features: int = 40,
                 out_features: int = 160):
        super(GCN, self).__init__()
        self.parallel = parallel

        # Compute the degree matrix and do the normalization
        self.special_matrix = torch.empty(parallel, 9, 9).cuda()
        temp_matrix = adj_matrix
        adj_matrix += torch.eye(temp_matrix.size(0)).to(adj_matrix.device)
        degree_matrix = torch.sum(temp_matrix != 0.0, axis=1)
        inverse_degree_sqrt = torch.diag(torch.pow(degree_matrix, -0.5))

        r"""
        \begin
        D^{\frac{-1}{2}} A D^{\frac{-1}{2}}
        \end
        """
        temp_matrix = (temp_matrix @ inverse_degree_sqrt).transpose(0, 1) @ inverse_degree_sqrt
        # 添加一层LAM
        self.A_L = nn.Parameter(torch.Tensor(parallel, 9, 9)).cuda()
        # self.A_L = torch.empty(parallel, 9, 9).cuda()  # 不可训练邻接矩阵
        # for i in range(parallel):
        #     self.special_matrix[i] = temp_matrix
        #     self.A_L.data[i] = temp_matrix

        # 使用N路并行的思路计算N路邻接矩阵
        self.graph_weight_one_List = nn.ModuleList([
            GraphConvolution(in_features=in_features,
                             out_features=hidden_features,
                             bias=False)
            for _ in range(parallel)
        ])
        self.graph_weight_two_List = nn.ModuleList([
            GraphConvolution(in_features=hidden_features,
                             out_features=out_features,
                             bias=False)
            for _ in range(parallel)
        ])
        self.embedding_List = nn.ModuleList([
            nn.Embedding(num_embeddings=num_embeddings,
                         embedding_dim=in_features)
            for _ in range(parallel)
        ])

        # self.init_parameters()

    def init_parameters(self):
        std = 1. / math.sqrt(self.A_L.size(1))
        self.A_L.data.uniform_(-std, std)
        for embedding in self.embedding_List:
            embedding.weight.data.normal_(0, 0.01)

    def forward(self, x):
        data = torch.empty(self.parallel, 9, 160).cuda()  # [9, 9, 160]
        for i in range(self.parallel):
            y = self.embedding_List[i](x)
            y = self.A_L[i] @ self.graph_weight_one_List[i](y)
            y = F.leaky_relu(y, 0.2)

            y = self.A_L[i] @ self.graph_weight_two_List[i](y)
            data[i] = F.leaky_relu(y, 0.2)

        return data


if __name__ == "__main__":
    adj_matrix = torch.FloatTensor([
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [1, 0, 1, 0]
    ])

    model = GCN(adj_matrix, num_embeddings=4, hidden_features=80)
    test_tensor = torch.randint(low=0, high=3, size=(4,))
    print(model(test_tensor).shape)
