from utils import *
from graphormer.layers import *
import torch.nn.functional as F

from graphormer.layers import *
# from parame import parameter_parser
from torchdiffeq import odeint
from torch_scatter import scatter_mean
from torch_geometric.nn import MeanSubtractionNorm


# args = parameter_parser()
class TransformerModel(nn.Module):
    def __init__(
            self,
            hops,
            feat_dim,
            hidden_dim,
            out_dim,
            num_mamba_layers,  # MoE 层数
            mamba_num_layers,  #mamba层
            dropout_rate,  # dropout 概率
            temperature,  # 对比学习温度系数
            num_drug,
            num_meta,
    ):
        super().__init__()
        self.hops= hops
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_mamba_layers = num_mamba_layers   #moe层
        self.mamba_num_layers = mamba_num_layers
        self.dropout_rate = dropout_rate
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.num_drug = num_drug
        self.num_meta = num_meta

        self.lin = nn.Linear(feat_dim, hidden_dim)
        # self.att_embeddings_nope = nn.Linear(128, 128)
        # self.att_embeddings_nope_1 = nn.Linear(64, 128)
        # lnn+deepseek
        # ✅ 保留 LNN 处理邻接矩阵 adj
        # self.lnn1 = LNN(845, self.hidden_dim)   #药物疾病845
        # self.lnn2 = LNN(615, self.hidden_dim)   #代谢物疾病615
        # self.lnn3 = LNN(1055, self.hidden_dim)   #药物代谢物1055
        # ✅ 保留 DeepSeekMoE 层
        self.graph_layers = nn.ModuleList([
            DeepSeekMoELayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                num_experts=4,
                dropout=dropout_rate
            ) for _ in range(num_mamba_layers)
        ])

        self.dropout = nn.Dropout(self.dropout_rate)

        # 对比学习投影头
        self.proj_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.KanConv = FastKANLayer(
            256,
            # self.hidden_dim,   #32维度的时候🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀，删除64
            # self.hidden_dim,
            256,  # 32维度的时候🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀，删除64
            num_grids=16,
            grid_min=-2.5,
            grid_max=2.5,
            spline_weight_init_scale=0.05,
            base_activation=nn.SiLU()
        )
        self.kan_norm = MeanSubtractionNorm()

        # 异构视角编码器
        self.strencoder = MambaEncoder(hidden_dim, hidden_dim, hops, False)

        self.fusion = EmbeddingFusion(hidden_dim, hidden_dim)


        ##########🔥维度变化
        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))
        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)                     #这是【注意力机制里的打分层】！把 两个特征拼接起来，通过线性层输出 1 个分数 → 用来判断谁重要、谁不重要！
        self.Linear1 = nn.Linear(int(self.hidden_dim / 2), self.out_dim)

        self.scaling = nn.Parameter(torch.ones(1) * 0.5)    #这是创建一个可学习的缩放因子（参数），初始值设为 0.5，模型训练时会自动调整它，用来控制注意力的强弱大小！

        self.semantic_attention = Attention(self.hidden_dim, self.dropout_rate)       #计算权重


        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 3)   #输出：对应3个类别的分数
        )
        self.apply(lambda module: init_params(module, n_layers=self.mamba_num_layers))   #给模型里所有层自动初始化权重

    def encode(self, feature, Adj):
        Adj = Adj.float()  # 邻接矩阵转 float32
        # ===================== 【自动生成 edge_index 和 edge_attr】 =====================
        # 从邻接矩阵 Adj 里直接恢复边！不需要外部传入！
        edge_index = Adj.nonzero(as_tuple=True)  # 找到所有有连接的位置
        edge_index = torch.stack(edge_index, dim=0)  # [2, E]
        edge_attr = Adj[edge_index[0], edge_index[1]]  # 边的权重

        # ===================== 动态创建 LNN（自动匹配维度！） =====================
        num_nodes = Adj.size(0)
        lnn_layer = LNN(num_nodes, self.hidden_dim).to(Adj.device)  # ✅ 自动匹配

        # === lnn+deepseek更新 ===
        # ✅ 只保留 LNN + MoE 分支
        x_deepseek = lnn_layer(Adj)  # LNN 处理邻接矩阵
        x_deepseek = x_deepseek.float()
        for layer in self.graph_layers:
            x_deepseek = layer(x_deepseek, edge_index, edge_attr)
            x_deepseek = F.relu(x_deepseek)
            x_deepseek = self.dropout(x_deepseek)

        # ===mamba===
        feature = self.lin(feature)
        x_str = self.strencoder(Adj, feature)  # mamba特征
        # #对比学习
        # z_deepseek = self.proj_head(x_deepseek)
        # z_str= self.proj_head(x_str)
        # contrastive_loss = self.contrastive_loss(z_str, z_deepseek)

        # =两种视角属性融合 ===
        x_fusion = self.fusion(x_str,x_deepseek)
        return x_fusion

    # 给融合后的 128 维特征用（不再做 lin 投影）
    def encode2(self, feature, Adj):
        Adj = Adj.float()

        edge_index = Adj.nonzero(as_tuple=True)
        edge_index = torch.stack(edge_index, dim=0)
        edge_attr = Adj[edge_index[0], edge_index[1]]

        num_nodes = Adj.size(0)
        lnn_layer = LNN(num_nodes, self.hidden_dim).to(Adj.device)

        x_deepseek = lnn_layer(Adj)
        for layer in self.graph_layers:
            x_deepseek = layer(x_deepseek, edge_index, edge_attr)
            x_deepseek = F.relu(x_deepseek)
            x_deepseek = self.dropout(x_deepseek)

        # ✅ 这里 不 再 使 用 self.lin！
        x_str = self.strencoder(Adj, feature)
        x_fusion = self.fusion(x_str, x_deepseek)
        return x_fusion


    def forward(self,Dr_Di_adj, M_Di_adj,Dr_Down_M_adj, Dr_Up_M_adj, Dr_Di_feature,M_Di_feature, train_edge_index, val_edge_index):

        # ===================== 1. 两条元路径编码 =====================
        Dr_Di = self.encode(Dr_Di_feature, Dr_Di_adj)
        M_Di = self.encode(M_Di_feature, M_Di_adj)

        # ===================== 2. 第一次融合 =====================
        #feat_cat = torch.cat([Dr_Di, M_Di], dim=0)
        # 2. ✅ 只提取【药物】和【代谢物】特征（关键！）
        drug_feat = Dr_Di[:self.num_drug]  # 药物（前 num_drug 个）
        meta_feat = M_Di[:self.num_meta]  # 代谢物（前 num_meta 个）

        # 3. ✅ 正确拼接：药物 + 代谢物
        feat_cat = torch.cat([drug_feat, meta_feat], dim=0)

        # ===================== 3. 第二次双分支编码 =====================
        feat_down = self.encode2(feat_cat, Dr_Down_M_adj)
        feat_up = self.encode2(feat_cat, Dr_Up_M_adj)

        # ===================== 4. 语义注意力（你的层！） =====================
        drug_feat_down = feat_down[:self.num_drug]
        drug_feat_up = feat_up[:self.num_drug]
        meta_feat_down = feat_down[self.num_drug:]
        meta_feat_up = feat_up[self.num_drug:]

        # 你的 Attention 层接收 LIST
        drug_final = self.semantic_attention([drug_feat_down, drug_feat_up])
        meta_final = self.semantic_attention([meta_feat_down, meta_feat_up])


        train_val_edge_index = torch.cat([train_edge_index, val_edge_index], 1)    #把【训练集的边】和【验证集的边】，在列维度（dim=1）拼在一起，合成一个训练 + 验证的总边集合 **！**

        Drug_Meta = []
        #遍历每一对样本对
        for i in range(train_val_edge_index.size(1)):    #train_val_edge_index.size(1)  样本对
            first_element = train_val_edge_index[0, i]   #药物节点
            second_element = train_val_edge_index[1, i]  #代谢物节点
            #取出当前这条边对应的【药物特征】和【代谢物特征】，头尾拼接成一个更长的向量，作为预测用的样本特征！
            final_feat = torch.cat((drug_final[first_element, :], meta_final[second_element, :]), dim=0)

            # ===================== 5. KAN + 解码 =====================
            x_kan = self.KanConv(final_feat)
            x_kan = self.kan_norm(x_kan)
            Drug_Meta.append(x_kan)

            all_features = torch.stack(Drug_Meta)
            out = self.mlp(all_features)

            # ===================== 对比学习损失 =====================
            z1 = self.proj_head(Dr_Di)
            z2 = self.proj_head(M_Di)
            # contrast_loss = self.contrastive_loss(z1, z2)
            contrast_loss = torch.tensor(0.0, device=Dr_Di.device)
            #Drug_embeddings[药物0] = [1, 2, 3]，Meta_embeddings[代谢物0] = [4, 5, 6]，combined_row = [1, 2, 3, 4, 5, 6]
           # Drug_Meta.append(x_kan)
        #把列表里一堆零散的拼接好的【药物 + 代谢物】特征，堆叠成一个完整的张量，变成可以直接送入 MLP 做预测的标准格式
        #Drug_Meta = [
                       #[1,2,3,4,5,6],   # 样本1：药物0+代谢物0
                       #[7,8,9,10,11,12] # 样本2：药物1+代谢物1
                    #...
                      #]
        # Drug_Meta_features = torch.stack(Drug_Meta)   #把列表转换成张量
        # embedings = self.mlp(Drug_Meta_features)    #执行mlp
        return out,contrast_loss

    def contrastive_loss(self, z1, z2):
        # 检查两个视图的样本数量是否相同
        if z1.size(0) != z2.size(0):
            return torch.tensor(0.0, device=z1.device, requires_grad=z1.requires_grad)
        batch_size = z1.size(0)
        z1 = F.normalize(z1, dim=1)   #L2 归一化
        z2 = F.normalize(z2, dim=1)

        sim_matrix = torch.matmul(z1, z2.T) / torch.clamp(self.temperature, min=0.07, max=1.0)   #将温度限制在 [0.07, 1.0] 范围内


        ####在对比学习中，对角线元素 sim_matrix[i][i] 对应正样本对（如同一图像的不同增强视图）的相似度，而非对角线元素对应负样本对
        pos_sim = torch.diag(sim_matrix)   #提取对角线元素
        logits = sim_matrix - pos_sim[:, None]  #每行减去该行对应的对角线值
        loss = -torch.log(torch.exp(pos_sim) / torch.exp(logits).sum(dim=1))
        return loss.mean()





