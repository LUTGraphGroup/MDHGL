from __future__ import division, print_function
import time
import random
import torch.nn.functional as F
from sklearn import metrics
from model import GCN
from loaddata import load_data
from parame import parameter_parser
from utils import *
from torch_geometric.data import Data
from pathlib import Path
import os
import numpy as np
np.set_printoptions(threshold=np.inf)

# project_root = Path(__file__).parent.parent
# data_dir = project_root

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parameter_parser()

# 固定随机种子以确保可复现性
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

# 数据加载
CAdj,Adj, Gene_Meta,Gene_Dis,Dis_data, Drug_data, ori_adj,  Cori_adj, Dis_adj, Drug_adj, feature = load_data()
print(ori_adj.shape)
#交叉验证准备
association_nam = Cori_adj.shape[1]                     #【补全矩阵存在时要改正】
random_index = Cori_adj.T.tolist()
random.shuffle(random_index)
k_folds = 5
CV_size = int(association_nam / k_folds)
temp = (
    np.array(random_index[: association_nam - association_nam % k_folds])
    .reshape(k_folds, CV_size, -1)
    .tolist()
)
temp[k_folds - 1] += random_index[association_nam - association_nam % k_folds :]
random_index = temp
for i, fold in enumerate(random_index):
    print(f"Fold {i+1}: 正样本数 = {len(fold)}")

# 结果容器
acc_result, pre_result, recall_result, f1_result, mcc_result,auc_result, prc_result = (
    [],
    [],
    [],
    [],
    [],
    [],
    [],
)
fprs, tprs, recalls, precisions = [], [], [], []

p10_result, p20_result, p30_result, p40_result, p50_result = [], [], [], [], []
p60_result, p70_result, p80_result, p90_result, p100_result = [], [], [], [], []
h10_result, h20_result, h30_result, h40_result, h50_result = [], [], [], [], []
h60_result, h70_result, h80_result, h90_result, h100_result = [], [], [], [], []

fold_features = []  # 存储每折的(x_fusion, auc, fold_idx)
best_overall_fold = -1
best_overall_auc = 0
best_overall_x_fusion = None

print(f"seed={args.seed}, evaluating metabolite-disease....")

for k in range(k_folds):
    print(f"------ Fold {k + 1}/{k_folds} ------")

    Or_train = np.matrix(Adj, copy=True)
    val_pos_edge_index = torch.tensor(np.array(random_index[k]).T, dtype=torch.long).to(
        device
    )

    # 负采样
    val_neg_edge_index = np.asmatrix(np.where(Or_train < 1)).T.tolist()
    random.shuffle(val_neg_edge_index)
    val_neg_edge_index = torch.tensor(
        np.array(val_neg_edge_index[: val_pos_edge_index.shape[1]]).T, dtype=torch.long
    ).to(device)

    # 移除验证集边
    Or_train[tuple(np.array(random_index[k]).T)] = 0
    train_pos_edge_index = torch.tensor(
        np.asmatrix(np.where(Or_train > 0)), dtype=torch.long
    ).to(device)

    Or_train_matrix = np.matrix(Adj, copy=True)                           #【矩阵补全存在时要改的】CA
    Or_train_matrix[tuple(np.array(random_index[k]).T)] = 0
    # or_adj = constructNet(torch.tensor(Or_train_matrix)).to(
    #     device
    # )  # 已移除验证集的边,剩下的边构成的二维矩阵 构建异构网络

    or_adj = constructHNet(torch.tensor(Or_train_matrix, dtype=torch.float32).to(device),
                        torch.tensor(Gene_Meta, dtype=torch.float32).to(device),
                        torch.tensor(Gene_Dis, dtype=torch.float32).to(device))

    edge_index1 = (or_adj > 0).nonzero().t().contiguous()  # [2, E]
    edge_attr = or_adj[edge_index1[0], edge_index1[1]]  # [E]

    Dis_network = torch.nonzero( Dis_adj, as_tuple=True)  # 返回邻接矩阵中非零元素的坐标（即存在边的节点对）。
    Dis_network = torch.stack(Dis_network)  # 将行列索引元组转换为 (2, num_edges) 的张量，符合PyG的边索引格式
    train_dis_data = Data(x=Dis_data.x, edge_index=Dis_network).to(device)
    # Construct metabolite similarity network
    Drug_network = torch.nonzero(Drug_adj, as_tuple=True)
    Drug_network = torch.stack(Drug_network)
    train_drug_data = Data(x=Drug_data.x, edge_index=Drug_network).to(device)

    # 构建模型
    model = GCN(
        feature.shape[1],
        args.hidden_dim,
        args.out_dim,
        args.num_mamba_layers,
        args.dropout,
       6451,
        args.temperature
    ).to(device)

    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay
    )

    criterion = F.binary_cross_entropy
    best_metrics = {
        "epoch": 0,
        "acc": 0,
        "auc": 0,
        "prc": 0,
        "fpr": None,
        "tpr": None,
        "recall": None,
        "precision": None,
        "full_metrics": None,
    }

    best_fold_auc = 0
    best_fold_x_fusion = None
    best_fold_epoch = 0

    # 训练过程
    for epoch in range(args.epochs):
        start = time.time()
        start_train = time.time()  # 返回当前时间
        model.train()
        optimizer.zero_grad()

        # 训练集负采样
        train_neg_edge_index = np.asmatrix(np.where(Or_train_matrix < 1)).T.tolist()
        random.shuffle(train_neg_edge_index)
        train_neg_edge_index = torch.tensor(
            np.array(train_neg_edge_index[: train_pos_edge_index.shape[1]]).T,
            dtype=torch.long,
        ).to(device)

        output,contrastive_loss,_= model(edge_index1, edge_attr, feature, or_adj)
        edge_index = torch.cat([train_pos_edge_index, train_neg_edge_index], dim=1)
        train_scores = output[edge_index[0], edge_index[1]].to(device)
        train_labels = get_link_labels(train_pos_edge_index, train_neg_edge_index).to(
            device
        )

        loss_train = criterion(train_scores, train_labels).to(device)
        loss = loss_train
        #loss = loss_train + args.Lambda * contrastive_loss
        loss.backward(retain_graph=True)
        optimizer.step()
        #lr_scheduler.step()
        end_train = time.time()  # 返回当前时间
        # 评估
        model.eval()
        with torch.no_grad():
            start_val = time.time()  # 返回当前时间
            train_auc = metrics.roc_auc_score(train_labels.cpu(), train_scores.cpu())
            predict_y_proba = output.reshape(Adj.shape[0], Adj.shape[1]).to(device)
            #predict_y_proba,_,x_fusion = model(edge_index1, edge_attr, feature, or_adj)
            score_val, label_val, metric_tmp = cv_model_evaluate(
                predict_y_proba, val_pos_edge_index, val_neg_edge_index
            )

            fpr, tpr, _ = metrics.roc_curve(label_val, score_val)
            precision, recall, _ = metrics.precision_recall_curve(label_val, score_val)
            val_auc = metrics.auc(fpr, tpr)
            val_prc = metrics.auc(recall, precision)
            end_val = time.time()  # 返回当前时间
            end = time.time()

            print(
                f"Epoch {epoch + 1} | Loss: {loss_train.item():.4f} | Acc: {metric_tmp[0]:.4f} | "
                f"Pre: {metric_tmp[1]:.4f} | Rec: {metric_tmp[2]:.4f} | F1: {metric_tmp[3]:.4f} | "
                f"mcc: {metric_tmp[4]:.4f} | Train AUC: {train_auc:.4f} | AUC: {val_auc:.4f} | PRC: {val_prc:.4f} "
                "Time: %.2f" % (end - start),
                "Time_train: %.2f" % (end_train - start_train),
                "Time_val: %.2f" % (end_val - start_val),
            )


            if (
                 metric_tmp[0] > best_metrics["acc"]
                 and val_auc > best_metrics["auc"]
                 and val_prc > best_metrics["prc"]
            ):
                best_metrics.update(
                    {
                        "epoch": epoch + 1,
                        "acc": metric_tmp[0],
                        "auc": val_auc,
                        "prc": val_prc,
                        "fpr": fpr,
                        "tpr": tpr,
                        "recall": recall,
                        "precision": precision,
                        "full_metrics": metric_tmp,
                    }
                )

                # =========新增P@K Hits@K计算========
                score_mat_np = predict_y_proba.cpu().numpy()
                # 在这里指定完整K序列，覆盖函数默认参数
                k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                rank_metrics = calc_precision_hits_at_k(score_mat_np, val_pos_edge_index, k_list=k_list)
                best_metrics["rank_metrics"] = rank_metrics
                print(f"【New Best Model Found!】 Rank Metrics: {rank_metrics}")
                # ===================================


                if val_auc > best_fold_auc:
                    best_fold_auc = val_auc
                    best_fold_epoch = epoch + 1
                    # 获取当前最佳x_fusion
                    with torch.no_grad():
                        _,_, current_x_fusion = model(edge_index1, edge_attr, feature, or_adj)
                        best_fold_x_fusion = current_x_fusion.cpu().clone()

    # 输出当前折最优结果
    print(
        f"Fold {k + 1} Best @Epoch {best_metrics['epoch']} | "
        f"Acc: {best_metrics['acc']:.4f} | AUC: {best_metrics['auc']:.4f} | PRC: {best_metrics['prc']:.4f}"
    )

    fold_features.append({
        'fold_idx': k + 1,
        'auc': best_metrics['auc'],
        'x_fusion': best_fold_x_fusion,
        'epoch': best_metrics['epoch']
    })

    if best_metrics['auc'] > best_overall_auc:
        best_overall_auc = best_metrics['auc']
        best_overall_fold = k + 1
        best_overall_x_fusion = best_fold_x_fusion

    acc_result.append(best_metrics["acc"])
    pre_result.append(best_metrics["full_metrics"][1])
    recall_result.append(best_metrics["full_metrics"][2])
    f1_result.append(best_metrics["full_metrics"][3])
    mcc_result.append(best_metrics["full_metrics"][4])
    auc_result.append(round(best_metrics["auc"], 4))
    prc_result.append(round(best_metrics["prc"], 4))
    fprs.append(best_metrics["fpr"])
    tprs.append(best_metrics["tpr"])
    recalls.append(best_metrics["recall"])
    precisions.append(best_metrics["precision"])

    #########新加的P@K
    rank_m = best_metrics["rank_metrics"]
    # Precision@K
    p10_result.append(rank_m["P@10"])
    p20_result.append(rank_m["P@20"])
    p30_result.append(rank_m["P@30"])
    p40_result.append(rank_m["P@40"])
    p50_result.append(rank_m["P@50"])
    p60_result.append(rank_m["P@60"])
    p70_result.append(rank_m["P@70"])
    p80_result.append(rank_m["P@80"])
    p90_result.append(rank_m["P@90"])
    p100_result.append(rank_m["P@100"])
    # Hits@K
    h10_result.append(rank_m["Hits@10"])
    h20_result.append(rank_m["Hits@20"])
    h30_result.append(rank_m["Hits@30"])
    h40_result.append(rank_m["Hits@40"])
    h50_result.append(rank_m["Hits@50"])
    h60_result.append(rank_m["Hits@60"])
    h70_result.append(rank_m["Hits@70"])
    h80_result.append(rank_m["Hits@80"])
    h90_result.append(rank_m["Hits@90"])
    h100_result.append(rank_m["Hits@100"])

print("\n" + "=" * 60)
print("最佳模型特征提取结果")
print("=" * 60)

# 找到AUC最高的折
best_idx = np.argmax(auc_result)
best_fold_info = fold_features[best_idx]

print(f"\nAUC最高的折: 第{best_idx + 1}折")
print(f"AUC值: {auc_result[best_idx]:.4f}")
print(f"对应epoch: {best_fold_info['epoch']}")

# 获取并保存X_fusion
best_x_fusion = best_fold_info['x_fusion']

if best_x_fusion is not None:
    # 转换为numpy
    x_fusion_np = best_x_fusion.numpy()

    print(f"\nX_fusion特征信息:")
    print(f"  形状: {x_fusion_np.shape}")
    print(f"  数据类型: {x_fusion_np.dtype}")
    print(f"  取值范围: [{x_fusion_np.min():.4f}, {x_fusion_np.max():.4f}]")
    print(f"  均值: {x_fusion_np.mean():.4f}, 标准差: {x_fusion_np.std():.4f}")

    # 保存为numpy文件
    np.save(f'best_x_fusion_fold{best_idx + 1}_auc{auc_result[best_idx]:.4f}.npy', x_fusion_np)
    print(f"  已保存为: best_x_fusion_fold{best_idx + 1}_auc{auc_result[best_idx]:.4f}.npy")
df = pd.DataFrame(x_fusion_np)
csv_filename = f'best_x_fusion_fold{best_idx + 1}.csv'     #最佳AUC对应的融合特征
df.to_csv(csv_filename, index=False, header=False)
print(f"  已保存为CSV: {csv_filename}")

# 显示前几行
print(f"\n前5行特征预览:")
for i in range(min(5, len(x_fusion_np))):
    print(f"  样本{i}: {x_fusion_np[i, :5].tolist()}...")
else:
    print("警告: 未找到X_fusion特征")

# 打印所有折的AUC对比
print(f"\n所有折的AUC对比:")
for i, (fold_info, auc_val) in enumerate(zip(fold_features, auc_result)):
    marker = " ← 最佳" if i == best_idx else ""
    print(f"  第{i + 1}折: AUC={auc_val:.4f}{marker}")

# 汇总结果
print("\n## Training Finished !")
print("Acc", acc_result)
print("Pre", pre_result)
print("Recall", recall_result)
print("F1", f1_result)
print("mcc", mcc_result)
print("AUC", auc_result)
print("PRC", prc_result)

print(
    f"AUC mean: {np.mean(auc_result):.4f}, std: {np.std(auc_result):.4f}\n"
    f"Acc mean: {np.mean(acc_result):.4f}, std: {np.std(acc_result):.4f}\n"
    f"Pre mean: {np.mean(pre_result):.4f}, std: {np.std(pre_result):.4f}\n"
    f"Recall mean: {np.mean(recall_result):.4f}, std: {np.std(recall_result):.4f}\n"
    f"F1 mean: {np.mean(f1_result):.4f}, std: {np.std(f1_result):.4f}\n"
    f"mcc mean: {np.mean(mcc_result):.4f}, std: {np.std(mcc_result):.4f}\n"
    f"PRC mean: {np.mean(prc_result):.4f}, std: {np.std(prc_result):.4f}\n"
    ##################新加的
    f"P@10 mean: {np.mean(p10_result):.4f}, std: {np.std(p10_result):.4f}\n"
    f"P@20 mean: {np.mean(p20_result):.4f}, std: {np.std(p20_result):.4f}\n"
    f"P@30 mean: {np.mean(p30_result):.4f}, std: {np.std(p30_result):.4f}\n"
    f"P@40 mean: {np.mean(p40_result):.4f}, std: {np.std(p40_result):.4f}\n"
    f"P@50 mean: {np.mean(p50_result):.4f}, std: {np.std(p50_result):.4f}\n"
    f"P@60 mean: {np.mean(p60_result):.4f}, std: {np.std(p60_result):.4f}\n"
    f"P@70 mean: {np.mean(p70_result):.4f}, std: {np.std(p70_result):.4f}\n"
    f"P@80 mean: {np.mean(p80_result):.4f}, std: {np.std(p80_result):.4f}\n"
    f"P@90 mean: {np.mean(p90_result):.4f}, std: {np.std(p90_result):.4f}\n"
    f"P@100 mean: {np.mean(p100_result):.4f}, std: {np.std(p100_result):.4f}\n"
    f"Hits@10 mean: {np.mean(h10_result):.4f}, std: {np.std(h10_result):.4f}\n"
    f"Hits@20 mean: {np.mean(h20_result):.4f}, std: {np.std(h20_result):.4f}\n"
    f"Hits@30 mean: {np.mean(h30_result):.4f}, std: {np.std(h30_result):.4f}\n"
    f"Hits@40 mean: {np.mean(h40_result):.4f}, std: {np.std(h40_result):.4f}\n"
    f"Hits@50 mean: {np.mean(h50_result):.4f}, std: {np.std(h50_result):.4f}\n"
    f"Hits@60 mean: {np.mean(h60_result):.4f}, std: {np.std(h60_result):.4f}\n"
    f"Hits@70 mean: {np.mean(h70_result):.4f}, std: {np.std(h70_result):.4f}\n"
    f"Hits@80 mean: {np.mean(h80_result):.4f}, std: {np.std(h80_result):.4f}\n"
    f"Hits@90 mean: {np.mean(h90_result):.4f}, std: {np.std(h90_result):.4f}\n"
    f"Hits@100 mean: {np.mean(h100_result):.4f}, std: {np.std(h100_result):.4f}"
)



pd.DataFrame(recalls).to_csv('../result/recalls_1.csv', index=False)
pd.DataFrame(precisions).to_csv('../result/precisions_1.csv', index=False)
pd.DataFrame(fprs).to_csv('../result/fprs_2.csv', index=False)
pd.DataFrame(tprs).to_csv('../result/tprs_2.csv', index=False)
# plot_auc_curves(fprs, tprs, auc_result, directory=data_dir, name="auc")
# plot_prc_curves(precisions, recalls, prc_result, directory=data_dir, name="prc")

# 保存最优模型
# save_dir = '/tmp/pycharm_project_161/models'  # 替换为你保存模型的路径（服务器上的相对路径）
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
#
# best_model_path = os.path.join(save_dir, 'best_model.pth')
# torch.save({
#     'epoch': best_metrics['epoch'],
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': best_metrics['full_metrics'][0],  # 这里保存的是最好的训练损失
# }, best_model_path)
#
# print(f"Best model saved at: {best_model_path}")