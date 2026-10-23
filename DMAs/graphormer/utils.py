import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def load_data(seed, n_components, hops):
    # 读取不同元路径下的语义网络
    Dr_M = pd.read_csv('../data/Dr-M.csv', header=0).values                  #药物-代谢物关联矩阵
    Dr_Down_M = pd.read_csv('../data/Dr-Down-M.csv', header=0).values        #药物-下调-代谢物
    Dr_Up_M = pd.read_csv('../data/Dr-Up-M.csv', header=0).values            #药物-上调-代谢物
    Dr_Di = pd.read_csv('../data/Dr-Di.csv', header=0).values                #药物-疾病
    M_Di = pd.read_csv('../data/M-Di.csv', header=0).values                  #代谢物---疾病
    Drug_mol2vec = pd.read_csv('../data/drug_mol2vec.csv', header=0).values   #药物特征
    Meta_mol2vec = pd.read_csv('../data/metabolite_mol2vec.csv', header=0).values   #代谢物特征
    Dis_MESH2vec = pd.read_csv('../data/Mesh2vec.csv', header=0).values       #疾病特征

    pca = PCA(n_components=n_components)                                                       #PCA降维
    PCA_dis_feature = pca.fit_transform(Dis_MESH2vec)
    PCA_meta_feature = pca.fit_transform(Meta_mol2vec)
    PCA_drug_feature = pca.fit_transform(Drug_mol2vec)

    Dis_feature = torch.FloatTensor(PCA_dis_feature).to(device)
    Meta_feature = torch.FloatTensor(PCA_meta_feature).to(device)
    Drug_feature = torch.FloatTensor(PCA_drug_feature).to(device)


    Dr_Di_feature = torch.cat((Drug_feature, Dis_feature), dim=0).to(device)      #拼接 药物，疾病特征
    M_Di_feature = torch.cat((Meta_feature, Dis_feature), dim=0).to(device)       #拼接   代谢物，疾病特征

    Dr_Down_M_adj = constructNet(torch.tensor(Dr_Down_M)).to(device)   #构造  药物-下调-代谢物  对角矩阵
    Dr_Up_M_adj = constructNet(torch.tensor(Dr_Up_M)).to(device)       #构造   药物-上调-代谢物  对角矩阵

    Dr_Di_adj = constructNet(torch.tensor(Dr_Di)).to(device)           #药物-疾病 对角矩阵

    M_Di_adj = constructNet(torch.tensor(M_Di)).to(device)              #代谢物-疾病 对角矩阵



    # Dr_Di_hops = re_features(Dr_Di_adj, Dr_Di_feature, hops).to(device)   #药物，疾病 [自身特征, 1跳, 2跳, ..., K跳]
    # M_Di_hops = re_features(M_Di_adj, M_Di_feature, hops).to(device)      #代谢物，疾病

    label_matrix = pd.read_csv('../data/label_matrix.csv', header=0).values   #1：上调  2：下调   0：无关联
    index_matrix = np.asmatrix(np.where(label_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    label1_index = temp


    index_matrix = np.asmatrix(np.where(label_matrix == 2))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    label2_index = temp
    return Dr_M, Dr_Di_adj,M_Di_adj ,Dr_Down_M_adj, Dr_Up_M_adj, Dr_Di_feature, M_Di_feature, label_matrix, label1_index, label2_index, k_folds
#输出药物-代谢物，
# 构造的药物下调代谢物浓度的邻接方阵（左上/右下为零块，右上为关联矩阵，左下为其转置），
# 构造的药物上调代谢物浓度的邻接方阵（左上/右下为零块，右上为关联矩阵，左下为其转置）
#  药物‑疾病元路径下，每个节点的 0~hops 阶邻居聚合特征
#药物‑基因元路径的多跳特征
#代谢物‑疾病元路径的多跳特征
#代谢物‑基因元路径的多跳特征
#标签矩阵（值 0/1/2）
#标签为 1 的样本索引的 5 折划分（每折是坐标对列表），，上调的
#标签为 2 的样本索引的 5 折划分（每折是坐标对列表），，下调的
#交叉验证折数，固定为5


def constructNet(association_matrix):
    n, m = association_matrix.shape
    drug_matrix = torch.zeros((n, n), dtype=torch.int8)
    meta_matrix = torch.zeros((m, m), dtype=torch.int8)
    mat1 = torch.cat((drug_matrix, association_matrix), dim=1)
    mat2 = torch.cat((association_matrix.t(), meta_matrix), dim=1)
    adj_0 = torch.cat((mat1, mat2), dim=0)
    return adj_0





def re_features(adj, features, K):   #得到形状 (N, K+1, D) 的张量   计算每个节点的0~K 跳邻居聚合特征
    nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])
    for i in range(features.shape[0]):
        nodes_features[i, 0, 0, :] = features[i]
    x = features + torch.zeros_like(features)
    x = x.double()
    for i in range(K):
        x = torch.matmul(adj, x)
        for index in range(features.shape[0]):
            nodes_features[index, 0, i + 1, :] = x[index]
    nodes_features = nodes_features.squeeze()
    return nodes_features


# Confusion Matrix Visualization
def plot_cm(average_cm, directory, name):
    # plt.figure(figsize=(8, 6))
    sns.heatmap(average_cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.savefig(directory + '/%s.pdf' % name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


'''可视化分类结果'''
def plot_T_SNE(tsne_scores_result, tsne_labels_result, directory, name):
    scaler = StandardScaler()
    tsne_scores_result = scaler.fit_transform(tsne_scores_result)
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    X_test_tsne = tsne.fit_transform(tsne_scores_result)

    class_labels = {0: 'NR', 1: 'UR', 2: 'DR'}
    tsne_labels_result = [class_labels[y] for y in tsne_labels_result]

    # plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_test_tsne[:, 0], y=X_test_tsne[:, 1], hue=tsne_labels_result, palette='bright', s=60)
    plt.title('T_SNE Visualization of MMT-fgMDI', fontsize=14)
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    plt.savefig(directory + '/%s.pdf' % name, dpi=300, bbox_inches='tight')
    plt.legend()
    plt.show()
