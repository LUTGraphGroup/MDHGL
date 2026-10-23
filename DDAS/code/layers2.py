import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv



class InnerProductDecoder(Module):
    """
    decoder 解码器
    """

    def __init__(self, input_dim, num_drug, dropout=0.1):
        super(InnerProductDecoder, self).__init__()
        self.weight = nn.Parameter(
            torch.empty(size=(input_dim, input_dim))
        )  # 建立一个w权重，用于对特征数进行线性变化
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)  # 对权重矩阵进行初始化
        self.dropout = dropout
        self.num_drug = num_drug

    def forward(self, inputs):
        inputs = F.dropout(inputs, self.dropout)
        D = inputs[0: self.num_drug, :]
       # M = inputs[self.num_drug:, :]
        M = inputs[self.num_drug:self.num_drug + 200, :]
        D = torch.mm(D, self.weight)
        M = torch.t(M)  # 转置
        #D = torch.t(D)
        x = torch.mm(D, M)

        outputs = torch.sigmoid(x)

        return outputs
        # return x





class MambaEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, K=7, use_sparse=False, agg="attention", norm_type="layer", mamba_num_layers=1, dropout=0.1): # K=7 ,mamba_num_layers=1, dropout=0.1
        super().__init__()
        self.K = K
        self.use_sparse = use_sparse
        self.agg = agg
        self.norm_type = norm_type  # 'none' | 'layer' | 'l2'
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = dropout
        self.mamba_num_layers = mamba_num_layers  # 记录 Mamba 层数
        #self.mamba = Mamba(d_model=hidden_dim)
        #定义一个模块列表，其中包含一个或多个mamba
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=hidden_dim,  # 与输入投影维度一致
                state_dim=hidden_dim // 2,  # 状态维度（每层一致）
                d_conv=4,  # 卷积核维度（每层一致）
                expand=2  # 扩展因子（每层一致）
            ) for _ in range(mamba_num_layers)
        ])
        self.mamba_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(mamba_num_layers)
        ]) # 为每层 Mamba 配一个 LayerNorm（Pre-Norm 结构，训练更稳定）
        # self.mamba = Mamba(
        #     d_model=hidden_dim,  # 与输入投影维度一致
        #     state_dim=hidden_dim // 2,  # 状态维度
        #     d_conv=4,  # 局部卷积核维度（官方默认值）
        #     expand=2  # 扩展因子（官方默认值）
        # )
        if norm_type == "layer":
            self.ln = nn.LayerNorm(input_dim)
        self.agg_attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),  # 输出每跳的注意力权重
            nn.Sigmoid()  # 用 Sigmoid 进行归一化
        )

    def normalize(self, x):
        if self.norm_type == "none":
            return x
        elif self.norm_type == "l2":
            return F.normalize(x, p=2, dim=-1)
        elif self.norm_type == "layer":
            return self.ln(x)
        else:
            raise ValueError(f"Unsupported norm type: {self.norm_type}")

#  多跳邻居特征
    def propagate_multi_hop(self, adj, features):
        """
        构建每个节点的0~K跳邻居特征序列，输出shape: (N, K+1, d)
        """
        adj = adj.to(features.dtype)
        N, d = features.shape
        x = features.clone().float()
        node_seq = torch.empty(N, self.K + 1, d, dtype=torch.float32, device=features.device)

        x = self.normalize(x)
        node_seq[:, 0, :] = x  # 第0跳特征

        for k in range(1, self.K + 1):
            if self.use_sparse:
                x = torch.sparse.mm(adj, x)
            else:
                x = torch.matmul(adj, x)
            x = self.normalize(x)
            node_seq[:, k, :] = x

        return node_seq.float()  # [N, K+1, d]

    def forward(self, adj, features):
        """
        features: 节点特征矩阵，shape=(N, d_in)
        returns: 结构感知特征，shape=(N, d_out)
        """
        node_seq = self.propagate_multi_hop(adj, features)  # [N, K+1, d_in]
        node_seq = self.input_proj(node_seq)  # [N, K+1, hidden_dim]
        # out_seq = self.mamba(node_seq)  # [N, K+1, hidden_dim]
        for i in range(self.mamba_num_layers):
            residual = node_seq  # 残差保存
            node_seq = self.mamba_norms[i](node_seq)  # 层归一化
            node_seq = self.mamba_layers[i](node_seq)  # 调用第 i 层 Mamba
            node_seq = F.dropout(node_seq, p=self.dropout, training=self.training)
            node_seq = node_seq + residual  # 残差连接

        if self.agg == "mean":  #各跳特征取平均
            out = node_seq.mean(dim=1)
        elif self.agg == "last":  #取最后一跳特征
            out = node_seq[:, -1, :]
        elif self.agg == "attention":   #注意力
            weights = torch.softmax(self.agg_attention(node_seq), dim=1)  # [N, K+1, 1]
            out = torch.sum(weights * node_seq, dim=1)
        else:
            raise ValueError(f"Unsupported agg mode: {self.agg}")

        return out

#两种特征嵌入的融合
class EmbeddingFusion(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, dropout=0.3, mode="cross"): #num_heads=8, dropout=0.1
        super().__init__()
        self.put_dim = input_dim
        self.mode = mode
        self.proj = nn.Linear(input_dim, output_dim)
        if mode == "fc":  #新增全连接模式
            self.fc_fusion = nn.Sequential(
                nn.Linear(input_dim * 2, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim, output_dim)
            )
        elif mode == "concat":
            self.mlp = nn.Sequential(
                nn.Linear(input_dim * 2, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
            )
        elif mode == "cross":
            self.cross_attn= nn.MultiheadAttention(
                input_dim, num_heads, batch_first=True, dropout=dropout
            )
            self.norm = nn.LayerNorm(input_dim)
        else:
            raise ValueError(f"Unsupported fusion mode {mode}")

    def forward(self, h_m, h_g):

        if self.mode == "fc":
            #全连接融合
            concatenated = torch.cat([h_m, h_g], dim=-1)
            return self.fc_fusion(concatenated)

        elif self.mode == "concat":
            # 拼接 + MLP
            h = torch.cat([h_m, h_g], dim=-1)  # (N, 2*dim)
            return self.mlp(h)

        elif self.mode == "cross":
            # 交叉注意力融合
            qm = h_m.unsqueeze(1)  #维度扩展成 (N, 1, dim)以符合MultiheadAttention输入shape (batch, seq_len, dim)
            kg = h_g.unsqueeze(1)
            #qg = h_g.unsqueeze(1)   #新加
            #km = h_m.unsqueeze(1)  #新加

            out_m, _ = self.cross_attn(qm, kg, kg)
            #out_g, _ = self.cross_attn(qg, km, km)  #新加

            out_m = out_m.squeeze(1)  #维度压缩成[N, dim]
            #out_g = out_g.squeeze(1)   #新加

            #out = (out_m + out_g) / 2  #新加
            out = self.proj(self.norm(out_m))

            return out

        return None