from sklearn.decomposition import PCA
import torch
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def constructNet(association_matrix):  # construct association matrix
    n, m = association_matrix.shape  # 此处为疾病
    drug_matrix = torch.zeros((n, n), dtype=torch.int8)
    dis_matrix = torch.zeros((m, m), dtype=torch.int8)
    mat1 = torch.cat((drug_matrix, association_matrix), dim=1)
    mat2 = torch.cat((association_matrix.t(), dis_matrix), dim=1)
    adj_0 = torch.cat((mat1, mat2), dim=0)
    return adj_0

def constructHNet(met_dis_matrix, gene_met=None, gene_dis=None):
    """
    PyTorch 版本
    保持原有特征矩阵构造风格，扩展到包含基因节点
    对角块为零矩阵，关联部分保留
    Returns: 特征矩阵 X: (M + D + G) × (M + D + G)
    """
    num_met = met_dis_matrix.shape[0]
    num_dis = met_dis_matrix.shape[1]
    num_gene = gene_met.shape[1] if (gene_met is not None) else 0

    device = met_dis_matrix.device

    # 对角块全部为 0
    met_met = torch.zeros((num_met, num_met), device=device)
    dis_dis = torch.zeros((num_dis, num_dis), device=device)
    gene_gene = torch.zeros((num_gene, num_gene), device=device)

    # 关联矩阵
    met_dis = met_dis_matrix
    gene_met_feat = gene_met if gene_met is not None else torch.zeros((num_gene, num_met), device=device)
    gene_dis_feat = gene_dis if gene_dis is not None else torch.zeros((num_gene, num_dis), device=device)

    # 拼接（和你原来逻辑完全一样）
    mat1 = torch.hstack((met_met, met_dis, gene_met_feat))
    mat2 = torch.hstack((met_dis.T, dis_dis, gene_dis_feat))
    mat3 = torch.hstack((gene_met_feat.T, gene_dis_feat.T, gene_gene))

    X = torch.vstack((mat1, mat2, mat3))
    return X


def constructFEANet(dis_fea, meta_fea):
    """
    拼接代谢物和疾病特征（代谢物在前，疾病在后），返回保留梯度的张量
    """
    if dis_fea.shape[1] != meta_fea.shape[1]:
        raise ValueError(
            "Feature dimensions do not match between disease and metabolite."
        )

    features = torch.cat([meta_fea, dis_fea], dim=0)  # 按行拼接 [M+N, D]
    return features  # 自动保持 device 和 requires_grad


def pca_reduce(embeddings, n_components=64):
    """
    对嵌入矩阵进行 PCA 降维

    参数：
        embeddings: numpy 数组，形状 [N, D]
        n_components: 降维后的维度，比如 64、128

    返回：
        降维后的嵌入 [N, n_components]
    """
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    return reduced


def constructDecoder(association_tensor):
    drug_tensor = torch.zeros(association_tensor.shape[0], association_tensor.shape[0])
    dis_tensor = torch.zeros(association_tensor.shape[1], association_tensor.shape[1])
    tensor1 = torch.cat((drug_tensor, association_tensor), 1)
    tensor2 = torch.cat((association_tensor.T, dis_tensor), 1)
    adj = torch.cat((tensor1, tensor2), 0)
    return adj


def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[: pos_edge_index.size(1)] = 1
    return link_labels


def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(sorted(list(set(predict_score.flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[
        np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)
    ]
    thresholds = np.asmatrix(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)

    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)

    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)

    def safe_divide(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    mcc_list = safe_divide(TP * TN - FP * FN, np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    # 计算MCC (Matthews Correlation Coefficient)
    # mcc_list = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    # # 处理分母为0的情况
    # mcc_list = np.nan_to_num(mcc_list, nan=0.0)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    mcc = mcc_list[max_index]

    return [
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1_score, 4),
        round(mcc, 4),
    ]


def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(np.interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(
            fprs[i],
            tprs[i],
            alpha=0.4,
            linestyle="--",
            label="Fold %d AUC: %.4f" % (i + 1, auc[i]),
        )

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(
        mean_fpr,
        mean_tpr,
        color="BlueViolet",
        alpha=0.9,
        label="Mean AUC: %.4f $\\pm$ %.4f" % (mean_auc, auc_std),
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.4)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    # plt.savefig(directory / "%s.pdf" % name, dpi=300, bbox_inches="tight")
    plt.savefig(directory / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_prc_curves(precisions, recalls, prc, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(np.interp(1 - mean_recall, 1 - recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(
            recalls[i],
            precisions[i],
            alpha=0.4,
            linestyle="--",
            label="Fold %d AUPR: %.4f" % (i + 1, prc[i]),
        )

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(
        mean_recall,
        mean_precision,
        color="BlueViolet",
        alpha=0.9,
        label="Mean AUPR: %.4f $\\pm$ %.4f" % (mean_prc, prc_std),
    )  # AP: Average Precision

    plt.plot([1, 0], [0, 1], linestyle="--", color="black", alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve")
    plt.legend(loc="lower left")
    plt.savefig(directory / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


import torch
import numpy as np
import pandas as pd


def to_tensor(data, dtype=torch.float32, device=None):
    """
    自动将 numpy / pandas / list / torch.Tensor 数据转换为 torch.Tensor
    并指定dtype和device。
    """

    if isinstance(data, torch.Tensor):
        tensor = data.to(dtype)
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).to(dtype)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        tensor = torch.tensor(data.values, dtype=dtype)
    elif isinstance(data, list):
        tensor = torch.tensor(data, dtype=dtype)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    if device is not None:
        tensor = tensor.to(device)
    return tensor


def cv_model_evaluate(output, val_pos_edge_index, val_neg_edge_index):
    edge_index = torch.cat([val_pos_edge_index, val_neg_edge_index], 1)
    val_scores = output[edge_index[0], edge_index[1]].to(device)
    val_labels = get_link_labels(val_pos_edge_index, val_pos_edge_index).to(device)
    return (
        val_scores.cpu().numpy(),
        val_labels.cpu().numpy(),
        get_metrics(val_labels.cpu().numpy(), val_scores.cpu().numpy()),
    )


def calc_precision_hits_at_k(score_matrix, val_pos_edge_index, k_list=[10,20,50]):
    """
    药物重定位标准排序指标：逐药物计算，最后求均值
    score_matrix: [n_drug, n_dis] 模型预测分数矩阵
    val_pos_edge_index: 验证集正边 [2, E]  第一维drug_idx，第二维disease_idx
    k_list: [10,20,50]
    return: dict {"P@10":xx, "P@20":xx, "P@50":xx, "Hits@10":xx, "Hits@20":xx, "Hits@50":xx}
    """
    import numpy as np
    n_drug, n_dis = score_matrix.shape
    # 构造验证集正样本字典 key:drug_id, value:set(真实疾病id)
    val_pos_dict = {}
    drug_ids = val_pos_edge_index[0].cpu().numpy()
    dis_ids = val_pos_edge_index[1].cpu().numpy()
    for d, di in zip(drug_ids, dis_ids):
        if d not in val_pos_dict:
            val_pos_dict[d] = set()
        val_pos_dict[d].add(di)

    p_k_record = {k: [] for k in k_list}
    hits_k_record = {k: [] for k in k_list}

    for drug_idx in val_pos_dict.keys():
        true_dis_set = val_pos_dict[drug_idx]
        # 当前药物所有疾病预测得分
        drug_scores = score_matrix[drug_idx, :]
        # 从高到低排序疾病索引
        sorted_dis_idx = np.argsort(-drug_scores)
        for K in k_list:
            topk_dis = sorted_dis_idx[:K]
            tp = len(set(topk_dis) & true_dis_set)
            p_k = tp / K
            p_k_record[K].append(p_k)
            hit = 1 if tp > 0 else 0
            hits_k_record[K].append(hit)

    # 求均值
    result = {}
    for K in k_list:
        result[f"P@{K}"] = round(np.mean(p_k_record[K]), 4)
        result[f"Hits@{K}"] = round(np.mean(hits_k_record[K]), 4)
    return result