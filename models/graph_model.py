import torch
import torch.nn as nn
from .transformer_encoder import TransformerEncoder


class ConvBlock(nn.Module):
    def  __init__(self, **kwargs):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(**kwargs),
            nn.BatchNorm2d(kwargs["out_channels"]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DWConv(nn.Module):
    def __init__(self, **kwargs):
        super(DWConv, self).__init__()

        self.block = nn.Sequential(
            ConvBlock(in_channels=kwargs["in_channels"],
                      out_channels=kwargs["in_channels"],
                      kernel_size=kwargs["kernel_size"],
                      padding=kwargs["kernel_size"] // 2,
                      groups=kwargs["in_channels"],
                      bias=False),
            ConvBlock(in_channels=kwargs["in_channels"],
                      out_channels=kwargs["out_channels"],
                      kernel_size=1,
                      bias=False)
        )

    def forward(self, x):
        return self.block(x)


class GraphLearningModel(nn.Module):
    def __init__(self,
                 parallel,
                 input_dim: int = 49,
                 forward_dim: int = 128,
                 num_heads: int = 8,
                 head_dim: int = 16,
                 num_layers: int = 6,
                 attn_drop_rate: float = 0.1,
                 proj_drop_rate: float = 0.5,
                 in_channels: int = 30,
                 stride: int = 1,
                 kernel_size: int = 3,
                 joint_layers: int = 1):
        super(GraphLearningModel, self).__init__()

        # Depth wise convolution for the input
        self.DWConv = nn.Sequential(
            ConvBlock(in_channels=in_channels,
                      out_channels=in_channels,
                      stride=stride,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2,
                      groups=in_channels),
            nn.Flatten(start_dim=2)
        )

        self.eyebrow_encoder = nn.Sequential(*[
            TransformerEncoder(input_dim=input_dim,
                               forward_dim=forward_dim,
                               num_heads=num_heads,
                               head_dim=head_dim,
                               drop_rate=attn_drop_rate)
            for _ in range(num_layers)
        ])
        self.mouth_encoder = nn.Sequential(*[
            TransformerEncoder(input_dim=input_dim,
                               forward_dim=forward_dim,
                               num_heads=num_heads,
                               head_dim=head_dim,
                               drop_rate=attn_drop_rate)
            for _ in range(num_layers)
        ])
        self.joint_encoder = nn.Sequential(*[
            TransformerEncoder(input_dim=input_dim,
                               forward_dim=forward_dim,
                               num_heads=num_heads,
                               head_dim=head_dim,
                               drop_rate=attn_drop_rate)
            for _ in range(joint_layers)
        ])

        self.eyebrow_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(490, 320),
            nn.ReLU(inplace=True),
            nn.Dropout(p=proj_drop_rate),
            nn.Linear(320, 160),
            nn.ReLU(inplace=True)
        )

        self.mouth_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(980, 320),
            nn.ReLU(inplace=True),
            nn.Dropout(p=proj_drop_rate),
            nn.Linear(320, 160),
            nn.ReLU(inplace=True)
        )

        self.joint_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(1470, 980),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(980, 320),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.2),
            nn.Linear(320, 160),
            nn.ReLU(inplace=True)
        )

        self.DWConv_List = nn.ModuleList([
            self.DWConv for _ in range(parallel)
        ])

        # self.Eyebrow_List = nn.ModuleList([
        #     nn.Sequential(
        #         self.eyebrow_encoder,
        #         self.eyebrow_layer
        #     ) for _ in range(parallel)
        # ])
        #
        # self.Mouth_List = nn.ModuleList([
        #     nn.Sequential(
        #         self.mouth_encoder,
        #         self.mouth_layer
        #     ) for _ in range(parallel)
        # ])

        self.Eyebrow_List = nn.ModuleList([
            nn.Sequential(
                self.eyebrow_encoder  # 进行联合训练时不需要首先展平，因此删除eyebrow_layer
            ) for _ in range(parallel)
        ])

        self.Mouth_List = nn.ModuleList([
            nn.Sequential(
                self.mouth_encoder
            ) for _ in range(parallel)
        ])

        self.Joint_List = nn.ModuleList([
            nn.Sequential(
                self.joint_encoder,
                self.joint_layer
            ) for _ in range(parallel)
        ])
        # self.Joint_List2 = nn.ModuleList([
        #     nn.Sequential(
        #         # self.joint_encoder,
        #         self.joint_layer
        #     ) for _ in range(parallel)
        # ])

    def forward(self, x):
        # # Before: Shape of x: (batch_size, 30, 7, 7)
        # # After: Shape of x: (batch_size, 30, 49)
        # x = self.DWConv(x)
        #
        # # Extract the specific part of vectors
        # eyebrow_vector = x[:, :10]
        # mouth_vector = x[:, 10:]
        #
        # # Shape of eyebrow_vector: (batch_size, 490)
        # # Shape of mouth_vector: (batch_size, 980)
        # eyebrow_vector = self.eyebrow_encoder(eyebrow_vector)
        # mouth_vector = self.mouth_encoder(mouth_vector)
        #
        # # Shape of eyebrow_vector: (batch_size, 160)
        # # Shape of mouth_vector: (batch_size, 160)
        # eyebrow_vector = self.eyebrow_layer(eyebrow_vector)
        # mouth_vector = self.mouth_layer(mouth_vector)

        # 这里的输入x的形状为[Batch_size, N, 30, 7, 7]
        # o_eye = torch.empty(x.shape[0], x.shape[1], 160).cuda()
        # o_mou = torch.empty(x.shape[0], x.shape[1], 160).cuda()
        temp_output = torch.empty(x.shape[0], x.shape[1], 30, 49).cuda()
        output = torch.empty(x.shape[0], x.shape[1], 160).cuda()  # [B, N, 160]
        for idx in range(x.shape[1]):

            feature = x[:, idx, :, :]  # feature: [batch, 30, 7, 7]
            feature = self.DWConv_List[idx](feature)  # feature: [batch, 30, 49]
            eyebrow_vector = feature[:, :10]
            mouth_vector = feature[:, 10:]

            o_eye = self.Eyebrow_List[idx](eyebrow_vector)  # o_eye: [B, 10, 49]
            o_mou = self.Mouth_List[idx](mouth_vector)  # o_mou: [B, 20, 49]

            # joint learning
            temp = torch.cat([o_eye, o_mou], dim=1)  # temp: [B, 30, 49]
            output[:, idx] = self.Joint_List[idx](temp)  # 二阶提取
            # output[:, idx] = self.Joint_List2[idx](temp)  # 一阶特征提取
        # 返回的两个张量的维度为[Batch_size, N, 160]
        return output, 0


if __name__ == "__main__":
    test_vector = torch.randn(32, 10,  30, 7, 7)
    model = GraphLearningModel(10)

    eyebrow, mouth = model(test_vector)

    print(eyebrow.shape)
    print(mouth.shape)
