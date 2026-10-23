import math
from torch import nn
import torch

##########################模型权重初始化############################
def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

############################语义级注意力#############
###########🔥 这是用来给【多条元路径特征】自动分配权重的层！
#######有的元路径重要 → 给大权重
#########有的元路径不重要 → 给小权重
###########最后把所有元路径特征加权融合成一个最终特征
class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        ####线性变换层
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        ####激活函数
        self.tanh = nn.Tanh()
        ####注意力向量
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)   #可学习的注意力向量（模型自己学出来的），可学习参数
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        #####把分数变成权重和为1
        self.softmax = nn.Softmax(dim=-1)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):   #embeds是一个列表，里面装着多条元路径的编码
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:     #——---------------------------------------------------元路径i的特征向量是经过Transformer编码后，每个节点在这条元路径上的表示，是一个张量，形状是：(节点数量, 特征维度)
            sp = self.tanh(self.fc(embed)).mean(dim=0)    #非线性变换：之前这条元路径的特征向量是（100节点，64维），变成一个64维的向量。
            beta.append(attn_curr.matmul(sp.t()))         #和注意力向量相乘->得到这条元路径的重要性分数     最后beta=【分数1，分数2，分数3，。。。，分数6】
        beta = torch.cat(beta).view(-1)    #把beta变成一维的扁平向量
        beta = self.softmax(beta)               #得到元路径的权重，，，，变成6个权重（和为1）
        # print("mp ", beta.data.cpu().numpy())  # semantic attention
        z_mp = 0
        for i in range(len(embeds)): #遍历每条元路径   len(embeds)=6
            z_mp += embeds[i] * beta[i]      #加权融合所有元路径特征  embeds[i]：第i条元路径的所有节点特征
        return z_mp    #融合了6条元路径所有语义的最终节点特征




########################################
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv

from torchdiffeq import odeint
from torch_scatter import scatter_mean
from torch_geometric.nn import MeanSubtractionNorm




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
            self.ln = nn.LayerNorm(hidden_dim)
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

class LNN(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 定义动态系统的参数
        self.W = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.01)
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        # 时间常数（可学习）
        self.tau = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x, t=1.0):
        # x形状: [batch_size, input_dim]
        h0 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)  # 初始状态
        # 定义常微分方程ODE函数: dh/dt = f(h, t)
        def ode_func(t, h):
            return (torch.mm(h, self.A) + torch.mm(x, self.W.T) + self.b) / self.tau
        # 用ODE求解器计算动态系统，h0是初始状态
        ht = odeint(ode_func, h0, torch.tensor([0.0, t]).to(x.device))[-1]
        return ht


class DeepSeekMoELayer(nn.Module):

    def __init__(self, in_channels, out_channels, num_experts=4, dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts

        #专家网络（每个专家独立参数），创建 num_experts 个独立的专家网络，每个专家是一个两层 MLP（线性 + SiLU + 线性 + Dropout）,,Sequential是顺序执行
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, out_channels * 2),
                nn.SiLU(),
                nn.Linear(out_channels * 2, out_channels),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])

        # self.experts = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(in_channels, out_channels * 2),
        #         nn.LayerNorm(out_channels * 2),
        #         nn.GELU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(out_channels * 2, out_channels),
        #         nn.LayerNorm(out_channels),
        #         nn.Dropout(dropout)
        #     ) for _ in range(num_experts)
        # ])


        # 门控控制器
        self.gate = nn.Linear(in_channels, num_experts)

        # 图结构处理
        self.graph_proj = nn.Linear(in_channels, in_channels)

        # 残差连接
        if in_channels != out_channels:
            self.res_proj = nn.Linear(in_channels, out_channels)
        else:
            self.res_proj = nn.Identity()

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for expert in self.experts:
            nn.init.xavier_uniform_(expert[0].weight)   #对每个专家的第一个线性层（输入 → 隐藏，对应 expert[0]）的权重进行 Xavier 均匀初始化
            nn.init.zeros_(expert[0].bias)              #将该线性层的偏置项初始化为 0
            nn.init.kaiming_normal_(expert[2].weight)   #对每个专家的第二个线性层（隐藏 → 输出，对应 expert[2]）的权重进行 Kaiming 正态初始化
            # nn.init.zeros_(expert[4].bias)   #新加
        nn.init.xavier_uniform_(self.gate.weight)
        # nn.init.zeros_(self.gate.bias)   #新加

    def forward(self, x, edge_index, edge_attr):
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        edge_attr: [num_edges]
        """
        residual = x

        # 1. 图结构聚合 (GNN风格)
        row, col = edge_index
        agg = scatter_mean(x[row] * edge_attr.unsqueeze(1), col, dim=0, dim_size=x.size(0))
        x = x + F.silu(self.graph_proj(agg))

        # 2. MoE处理
        # 专家动态路由，              计算门控权重 [num_nodes, num_experts]
        gate_scores = F.softmax(self.gate(x), dim=-1)

        # 各专家前向传播
        #遍历每一个专家，把 x 传给这个专家，让他计算输出，每个专家输出都是：[节点数，输出维度]
        expert_outputs = torch.stack([expert(x) for expert in self.experts],
                                     dim=1)  # [num_nodes, num_experts, out_channels]

        # 加权求和，爱因斯坦求和公式：相同字母自动相乘 + 求和
        moe_out = torch.einsum('n e, n e c -> n c', gate_scores, expert_outputs) #n e → gate_scores（节点，专家），n e c → expert_outputs（节点，专家，特征），n c → 输出（节点，特征）

        # 3. 门控残差连接
        return self.res_proj(residual) + F.gelu(moe_out)

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)
    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


##########这是一个自定义神经网络层，叫径向基函数 RBF。
class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2., # 网格最小值
            grid_max: float = 2.,  # 网格最大值
            num_grids: int = 8,    # 网格数量（多少个基函数）
            denominator: float = None,  # larger denominators lead to smoother basis
    ): #在 -2 到 2 之间，摆 8 个光滑的 “小波峰”,这些小波峰就是RBF
        #创建网格（摆小波峰），，生成一组固定的中心点：[-2.0, -1.42, -0.85, -0.28, 0.28, 0.85, 1.42, 2.0]
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        #光滑系数，控制小波峰宽不宽、滑不滑
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)
    #把输入 x 与每个网格中心点做差，除以平滑系数，平方后取负数，再 exp
    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class FastKANLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            use_base_update: bool = True,
            base_activation=F.silu,
            spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, time_benchmark=False):
        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))  #归一化 + 生成 RBF 基函数
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret




