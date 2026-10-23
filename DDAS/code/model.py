from layers2 import *
from parame import parameter_parser
from torchdiffeq import odeint
from torch_scatter import scatter_mean
from torch_geometric.nn import MeanSubtractionNorm

args = parameter_parser()
class GCN(nn.Module):

    def __init__(
        self,
        feat_dim,
        hidden_dim,
        out_dim,
        num_mamba_layers,  # MoE 层数
        dropout,  # dropout 概率
        num_drug,
        temperature  # 对比学习温度系数
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_drug = num_drug
        self.dropout = dropout
        self.num_mamba_layers = num_mamba_layers
        self.temperature = nn.Parameter(torch.tensor(temperature))
        #lnn+deepseek
        # ✅ 保留 LNN 处理邻接矩阵 adj
        self.lnn = LNN(1260, self.hidden_dim)
        self.lin = nn.Linear(self.feat_dim, self.hidden_dim)
        # ✅ 保留 DeepSeekMoE 层
        self.graph_layers = nn.ModuleList([
            DeepSeekMoELayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                num_experts=4,
                dropout=dropout
            ) for _ in range(num_mamba_layers)
        ])

        self.dropout = nn.Dropout(self.dropout)

        # 对比学习投影头
        self.proj_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.KanConv = FastKANLayer(
            64,
            #self.hidden_dim,   #32维度的时候🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀，删除64
            #self.hidden_dim,
           64,     #32维度的时候🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀，删除64
            num_grids=16,
            grid_min=-2.5,
            grid_max=2.5,
            spline_weight_init_scale=0.05,
            base_activation=nn.SiLU()
        )
        self.kan_norm = MeanSubtractionNorm()

        # 异构视角编码器
        self.strencoder = MambaEncoder(
            feat_dim, hidden_dim, args.hops, False, args.agg
        )
        # self.fusion = UnifiedFusion(
        #     hidden_dim, out_dim,  mode=args.fusion_mode
        # )

        self.fusion = EmbeddingFusion(
                hidden_dim, out_dim
            )

        # 解码器：链路预测
        self.decoder = InnerProductDecoder(out_dim, num_drug)

    def forward(self,index,attr, feature, Adj):
        Adj = Adj.float()  # 邻接矩阵转 float32
        attr = attr.float()  # 边属性转 float32

        # === lnn+deepseek更新 ===
        # ✅ 只保留 LNN + MoE 分支
        x_deepseek = self.lnn(Adj)  # LNN 处理邻接矩阵
        x_deepseek = x_deepseek.float()
        for layer in self.graph_layers:
            x_deepseek = layer(x_deepseek, index, attr)
            x_deepseek = F.relu(x_deepseek)
            x_deepseek = self.dropout(x_deepseek)

        # ===mamba===
        x_str = self.strencoder(Adj, feature)  # mamba特征
        #对比学习
        z_deepseek = self.proj_head(x_deepseek)
        z_str= self.proj_head(x_str)
        contrastive_loss = self.contrastive_loss(z_str, z_deepseek)

        # =两种视角属性融合 ===
        x_fusion1 = self.fusion(x_str,x_deepseek)
        x_kan = self.KanConv(x_fusion1)
        x_fusion = self.kan_norm(x_kan)

        # =解码为链路分数
        output = self.decoder(x_fusion)

        return output,  contrastive_loss ,x_fusion #修改前返回的

    def contrastive_loss(self, z1, z2):
        batch_size = z1.size(0)
        z1 = F.normalize(z1, dim=1)   #L2 归一化
        z2 = F.normalize(z2, dim=1)

        sim_matrix = torch.matmul(z1, z2.T) / torch.clamp(self.temperature, min=0.07, max=1.0)   #将温度限制在 [0.07, 1.0] 范围内


        ####在对比学习中，对角线元素 sim_matrix[i][i] 对应正样本对（如同一图像的不同增强视图）的相似度，而非对角线元素对应负样本对
        pos_sim = torch.diag(sim_matrix)   #提取对角线元素
        logits = sim_matrix - pos_sim[:, None]  #每行减去该行对应的对角线值
        loss = -torch.log(torch.exp(pos_sim) / torch.exp(logits).sum(dim=1))
        return loss.mean()

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