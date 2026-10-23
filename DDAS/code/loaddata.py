from parame import *
from pathlib import Path
from utils import *
from torch_geometric.data import Data
import pandas as pd
# 获取项目根目录（DMP/）
project_root = Path(__file__).parent.parent  # 假设data.py在DMP/code/下
data_dir = project_root / "data"  # 数据目录绝对路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    args = parameter_parser()

    # 读取邻接矩阵（疾病-代谢物关联）
    CAdj = pd.read_csv('../data/A-dataset/Dr-Di.csv', header=None) #c:association_matrix_completed_5.csv   jaccard补全后为C_association_matrix_completed_fusion,f:F_association_matrix_completed.csv,B:output/B_association_matrix_completed.csv' 或者B_association_matrix_completed_fusion_1.csv
    Adj = pd.read_csv('../data/A-dataset/Dr-Di.csv', header=None)  # header=0第一行作为列名  #c:adj.csv,f:adj_1,B:adj

    Gene_Meta = pd.read_csv('../data/A-dataset/Dr-M.csv', header=None)  # 药物-代谢物关联网络
    Gene_Meta = np.matrix(Gene_Meta, copy=True)
    Gene_Dis = pd.read_csv('../data/A-dataset/Di-M.csv', header=None)  # 疾病-代谢物关联网络
    Gene_Dis = np.matrix(Gene_Dis, copy=True)
    Ori_adj = np.asmatrix(np.where(Adj == 1))  # 交叉验证用
    COri_adj = np.asmatrix(np.where(CAdj == 1))  # 交叉验证用

    print(f"药物疾病邻接矩阵：\n{Adj}")

    # 读取疾病相似性矩阵
    Dis_adj = pd.read_csv('../data/A-dataset/di_di_adj_fus.csv', header=None) #c:原始C_disease_network_simi,C,F:F_disease_network_simi_fusion.csv


    Dis_simi = pd.read_csv('../data/A-dataset/di_di_adj_fus.csv', header=None)
    Ori_dis_index_matrix = np.asmatrix(np.where(Dis_adj == 1))
    Dis_index_matrix = torch.tensor(Ori_dis_index_matrix, dtype=torch.long).to(device)

    # 相似矩阵作为特征矩阵
    Dis_features = pd.read_csv('../data/A-dataset/Mesh2vec.csv', header=0)
    Dis_features = pca_reduce(Dis_features, n_components=64)
    Dis_features = torch.tensor(Dis_features, dtype=torch.float).to(device)



    # 读取药物相似性矩阵

    Drug_adj = pd.read_csv('../data/A-dataset/dr_dr_adj_fus.csv', header=None)
    Drug_simi = pd.read_csv('../data/A-dataset/drug_similarity.csv', header=None)
    Ori_drug_index_matrix = np.asmatrix(np.where(Drug_adj == 1))
    Drug_index_matrix = torch.tensor(Ori_drug_index_matrix, dtype=torch.long).to(device)

    #相似矩阵作为特征矩阵
    Drug_features = pd.read_csv('../data/A-dataset/drug_mol2vec.csv', header=0)
    Drug_features = pca_reduce(Drug_features, n_components=64)
    Drug_features = torch.tensor(Drug_features, dtype=torch.float).to(device)


    ##代谢物特征矩阵
    Gene_features = pd.read_csv('../data/A-dataset/metabolite_mol2vec.csv', header=0)
    Gene_features = pca_reduce(Gene_features, n_components=64)
    Gene_features = torch.tensor(Gene_features, dtype=torch.float).to(device)

    Drug_data = Data(
        x=Drug_features,edge_index=Drug_index_matrix,Meta_adj=Drug_adj,Drug_simi=Drug_simi,
    )
    Dis_data = Data(
        x=Dis_features, edge_index=Dis_index_matrix, Dis_adj=Dis_adj, Dis_simi=Dis_simi
    )

    feature = torch.cat((Drug_features,Dis_features,Gene_features))
    Dis_adj=to_tensor(Dis_adj)
    Drug_adj=to_tensor(Drug_adj)
    return CAdj,Adj, Gene_Meta,Gene_Dis,Dis_data, Drug_data, Ori_adj, COri_adj,Dis_adj, Drug_adj, feature
