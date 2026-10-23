from __future__ import division
from __future__ import print_function
import time
import argparse
from model import *
from sklearn import metrics
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from parame import parameter_parser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# training parameters
parser = argparse.ArgumentParser()




args = parser.parse_args()
print('args', args)



Dr_M, Dr_Di_adj, M_Di_adj, Dr_Down_M_adj, Dr_Up_M_adj, Dr_Di_feature, M_Di_feature, label_matrix, label1_index, label2_index, k_folds= load_data(args.seed, args.feat_dim, args.hops)

acc_result = []

cumulative_cm = np.zeros((3, 3), dtype=int)   #存3分类的总混淆矩阵（看对错分布）
micro_fpr_result = []   #存微平均 FPR
micro_tpr_result = []   #存微平均 TPR
macro_fpr_result = []   #存宏平均 FPR
macro_tpr_result = []   #存宏平均 TPR

pre_result = {
                'pre_micro': [],
                'pre_macro': [],
                }
reca_result = {
                'recall_micro': [],
                'recall_macro': [],
                }
f1_result = {
                'f1_micro': [],
                'f1_macro': [],
                }

ma_auc_result = []
mi_auc_result = []
ma_aupr_result = []
mi_aupr_result = []
micro_recall_result= []
micro_pre_result = []
macro_recall_result = []
macro_pre_result = []

tsne_scores_result = []
tsne_labels_result = []


print("seed=%d, evaluating metabolite-disease...." % args.seed)
for k in range(k_folds):
    print("------this is %dth cross validation------" % (k + 1))
    Dr_M_matrix = np.matrix(Dr_M, copy=True)     #和二分类的一样
    val_pos_edge_index = np.array(label1_index[k]+label2_index[k]).T     #比二分类的多个，二分类是一个index
    val_pos_edge_index = torch.tensor(val_pos_edge_index, dtype=torch.long).to(device)    #一样

    val_neg_edge_index = np.asmatrix(np.where(label_matrix < 1)).T.tolist()     #不一样，二分类是Dr_M_matrix中小于1的边，，三分类是标签矩阵label_matrix（2，1，0）中小于1的
    random.seed(args.seed)   #一样
    random.shuffle(val_neg_edge_index)   #一样
    val_neg_edge_index = val_neg_edge_index[:len(label2_index[k])]    #与二分类不一样，，，三分类，所以要取与某一类标签正样本相同数量的负样本
    val_neg_edge_index = np.array(val_neg_edge_index).T
    val_neg_edge_index = torch.tensor(val_neg_edge_index, dtype=torch.long).to(device)

    #移除验证集，，，，与二分类一样
    Dr_M_matrix[tuple(np.array(label1_index[k]+label2_index[k]).T)] = 0   #与二分类一样
    train_pos_edge_index = np.asmatrix(np.where(Dr_M_matrix > 0))    #除了每一轮中的验证集边，剩余的正边是训练正样本
    train_pos_edge_index = torch.tensor(train_pos_edge_index, dtype=torch.long).to(device)

#✅✅✅✅✅✅可修改✅✅✅✅✅
    model = TransformerModel(args.hops,
                             args.feat_dim,
                             args.hidden_dim,
                             args.out_dim,
                             args.num_mamba_layers,  # MoE 层数
                             args.mamba_num_layers,  # mamba层
                             dropout_rate=args.dropout,  # dropout 概率
                             temperature=args.temperature,
                             num_drug=Dr_M.shape[0],
                             num_meta=Dr_M.shape[1],
                             ).to(device)
    print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = F.cross_entropy    ##与二分类不一样，，F.cross_entropy是多分类专用

    best_acc = 0
    best_cm = 0
    best_epoch = 0
    best_ma_auc = 0
    best_mi_auc = 0
    best_ma_aupr = 0
    best_mi_aupr = 0
    best_tpr = 0
    best_fpr = 0
    best_recall = 0
    best_precision = 0
    best_micro_fpr = 0
    best_micro_tpr = 0
    best_macro_fpr = 0
    best_macro_tpr = 0
    best_micro_recall = 0
    best_micro_pre = 0
    best_macro_recall = 0
    best_macro_pre = 0
    best_val_label = 0
    best_val_score = 0
    best_val_feature = 0

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        optimizer.zero_grad()  #以上与二分类一样

        #训练集负采样
        train_neg_edge_index = np.asmatrix(np.where(Dr_M_matrix < 1)).T.tolist()
        random.shuffle(train_neg_edge_index)      #以上与2分类一样，取关联矩阵中<1的边索引
        train_neg_edge_index = train_neg_edge_index[:np.count_nonzero(label_matrix==2)-len(label2_index[k])]    #不一样，，2分类是取与正样本数量一样的负样本，，，，3分类是训练集里正样本类别 2 的数量=让负样本的个数 = 【全体类别 2 总数】减去【当前 fold 里的类别 2 数量】
        train_neg_edge_index = np.array(train_neg_edge_index).T
        train_neg_edge_index = torch.tensor(train_neg_edge_index, dtype=torch.long).to(device)  #以上2步与2分类一样

        train_edge_index = torch.cat([train_pos_edge_index, train_neg_edge_index], 1)     #与2分类一样，，，合成训练集的全部样本
        val_edge_index = torch.cat([val_pos_edge_index, val_neg_edge_index], 1)              ####3############2分类没有这一步

####✅✅✅✅✅✅✅可修改✅✅✅✅
        #output = model(Dr_Down_M_adj, Dr_Up_M_adj, Dr_Di_hops, M_Di_hops, train_edge_index, val_edge_index, args.hops)  #与2分类一样
        output,contrast_loss= model(
            Dr_Di_adj, M_Di_adj,
            Dr_Down_M_adj, Dr_Up_M_adj,
            Dr_Di_feature, M_Di_feature,
            train_edge_index, val_edge_index,
        )

        train_scores = output[: train_edge_index.shape[1]]      #####与二分类不一样，，——————————从模型输出的所有结果里，只取出前【训练集样本】对应的那部分预测分数！
        train_scores = F.softmax(train_scores, dim=1)    #--------把模型输出的分数 → 变成【每一类的概率】，让 3 个值加起来等于 1！
        label_matrix_tensor = torch.tensor(label_matrix).to(device)    #把numpy 标签矩阵 → 变成 PyTorch 张量 → 送到 GPU 上
        train_labels = label_matrix_tensor[train_edge_index[0], train_edge_index[1]].to(device)    #和2分类不一样，  从全局标签表里，精准取出【当前训练集样本】对应的真实标签

        loss_train = criterion(train_scores, train_labels.long()).to(device)   #和2分类描述一样，，但损失函数criterion是三分类损失函数
       # loss_train = criterion(train_scores, train_labels.long()).to(device) +args.Lambda * contrast_loss
        loss_train.backward(retain_graph=True)
        optimizer.step()   #以上都一样

        model.eval()
        with torch.no_grad():  #一样
            score_train_cpu = train_scores.detach().cpu().numpy()
            label_train_cpu = train_labels.detach().cpu().numpy()
            train_auc = metrics.roc_auc_score(label_train_cpu, score_train_cpu, multi_class='ovr')   #和2分类一样

            val_scores = output[train_edge_index.shape[1]:]   #不一样，验证集的预测分数
            val_scores = F.softmax(val_scores, dim=1)         #将预测分数转成概率（0-1，三类加起来=1）
            val_scores_cpu = val_scores.detach().cpu().numpy()

            val_labels = label_matrix_tensor[val_edge_index[0], val_edge_index[1]].to(device)   #不一样，，，，从全局标签表里，精准取出【当前验证集样本】对应的🎉🎉🎉《《《《真实标签》》》》
            val_labels_cpu = val_labels.detach().cpu().numpy()
            ma_val_auc = round(metrics.roc_auc_score(val_labels_cpu, val_scores_cpu, multi_class='ovr', average="macro"), 4)   #三类各自算AUC再平均，看每一类准不准
            mi_val_auc = round(metrics.roc_auc_score(val_labels_cpu, val_scores_cpu, multi_class='ovr', average="micro"), 4)   #全部样本一起算AUC，看整体准不准

            ########🎉🎉🎉生成《《《《预测标签》》》》，用来和真实标签对比算准确率
            val_scores_1 = val_scores.max(1)[1].detach().cpu().numpy()    #—————————————————把验证集的 3 个概率，取出【概率最大的那一类的编号】 → 得到最终预测标签 0 / 1 / 2，，，，，val_scores.max(1)只要第二个返回值（类别编号），因为val_scores返回两个东西（1.最大概率值，2.这个概率对应的类别编号）
            val_acc = round(accuracy_score(val_labels_cpu, val_scores_1), 4)    #计算准确率

            cm = confusion_matrix(val_labels_cpu, val_scores_1)   #confusion_matrix（）是一个现成的工具函数，----生成3x3的表格（混淆矩阵），

            val_bin = label_binarize(val_labels_cpu, classes=[0, 1, 2])      ##把单个数字标签 → 变成一行向量，，比如哪一类是 1，对应位置就标 1，其余标 0！，其他两类也是一样
            mi_val_aupr = round(metrics.average_precision_score(val_bin, val_scores_cpu, average="micro"), 4)
            ma_val_aupr = round(metrics.average_precision_score(val_bin, val_scores_cpu, average="macro"), 4)


            #传入真实标签+预测标签
            val_result = classification_report(val_labels_cpu, val_scores_1)  #classification_report自动算，自动输出，一次性输出：精确率、召回率、F1 分数
            #算模型预测的 “精准度”，micro 看整体，macro 看每类平均
            precision_score_average_micro = precision_score(val_labels_cpu, val_scores_1, average='micro')
            precision_score_average_macro = precision_score(val_labels_cpu, val_scores_1, average='macro')
            precision_scores = {

                'pre_micro': round(precision_score_average_micro, 4),
                'pre_macro': round(precision_score_average_macro, 4)

            }

            recall_score_average_micro = recall_score(val_labels_cpu, val_scores_1, average='micro')
            recall_score_average_macro = recall_score(val_labels_cpu, val_scores_1, average='macro')

            recall_scores = {

                'recall_micro': round(recall_score_average_micro, 4),
                'recall_macro': round(recall_score_average_macro, 4)
            }

            f1_score_average_micro = f1_score(val_labels_cpu, val_scores_1, average='micro')
            f1_score_average_macro = f1_score(val_labels_cpu, val_scores_1, average='macro')
            f1_scores = {

                'f1_micro': round(f1_score_average_micro, 4),
                'f1_macro': round(f1_score_average_macro, 4)
            }

            end = time.time()                  #一样
            # if (epoch + 1) % 10 == 0:
            print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.item(), 'Train AUC: %.4f' % train_auc,
                  'val acc: %.4f' % val_acc,
                  'ma AUC: %.4f' % ma_val_auc,
                  'mi AUC: %.4f' % mi_val_auc,
                  'ma AUPR: %.4f' % ma_val_aupr,
                  'mi AUPR: %.4f' % mi_val_aupr,
                  'ma pre: %.4f' % precision_score_average_macro,
                  'ma recall: %.4f' % recall_score_average_macro,
                  'ma f1: %.4f' % f1_score_average_macro,
                  'Time: %.2f' % (end - start))

            fpr = dict()
            tpr = dict()
            val_auc = dict()
            precision = dict()
            recall = dict()
            val_aupr = dict()

            n_classes = val_bin.shape[1]   #3类就是3
            for i in range(n_classes):
                fpr[i], tpr[i], _ = metrics.roc_curve(val_bin[:, i], val_scores_cpu[:, i])    #**********算第 i 类的 ROC 曲线：假正率 fpr、真正率 tpr
                val_auc[i] = metrics.auc(fpr[i], tpr[i])  #算第 i 类的 AUC 值
                precision[i], recall[i], _ = metrics.precision_recall_curve(val_bin[:, i], val_scores_cpu[:, i])  #算第 i 类的 精确率-召回率曲线
                val_aupr[i] = metrics.auc(recall[i], precision[i])   #**********算第 i 类的 AUPR 值

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = metrics.roc_curve(val_bin.ravel(), val_scores_cpu.ravel())    #**********计算多分类任务中，全局整体的 micro 模式 ROC 曲线
            val_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

            # Compute micro-average PR curve and PR area
            precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(val_bin.ravel(), val_scores_cpu.ravel())

            # First aggregate all false positive rates
            #👍1.把 类别 0、1、2 三条 ROC 曲线里所有出现过的 FPR（假正率）数值全部收集起来，去重、排序，得到一条统一的、公共的横轴坐标，专门用来后面画宏平均（macro）ROC 曲线
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            # Then interpolate all ROC curves at this points
            #👍2.创建一个和 all_fpr 长度一模一样、全部是 0 的数组，用来后面累加、计算【3 个类别的平均 TPR】
            mean_tpr = np.zeros_like(all_fpr)
            #👍3.把 3 个类别的 ROC 曲线，全部对齐到统一的横轴 all_fpr 上，然后把它们的 TPR 加起来！为了最后算【平均 macro TPR】
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            # Finally average it and compute AUC
            mean_tpr /= n_classes     #👍4.除以 3，算出【宏平均（macro）TPR】
            fpr["macro"] = all_fpr    #👍将去重后的fpr作为 macro 曲线的横坐标
            tpr["macro"] = mean_tpr   #👍把刚刚算好的 平均 TPR作为 macro 曲线的纵坐标。
            val_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])   #👍用 macro 统一的 FPR（横轴） 和 macro 平均的 TPR（纵轴）计算出 宏平均 AUC！

            #🎯 Compute macro-average PR curve and PR area
            # First aggregate all recall
            all_recall = np.unique(
                np.concatenate([recall[i] for i in range(n_classes)]))
            # 🎯Then interpolate all ROC curves at this points
            mean_pre = np.zeros_like(all_recall)
            for i in range(n_classes):
                mean_pre += np.interp(1-all_recall, 1-recall[i], precision[i])
            #🎯 Finally average it and compute AUC
            mean_pre /= n_classes           #将三个类别的pre加起来，再÷3，得一组
            recall["macro"] = all_recall
            precision["macro"] = mean_pre

            if val_acc > best_acc and ma_val_auc > best_ma_auc and ma_val_aupr > best_ma_aupr:
                best_acc = val_acc
                best_pre = precision_scores
                best_rec = recall_scores
                best_f1 = f1_scores
                best_ma_auc = ma_val_auc
                best_mi_auc = mi_val_auc
                best_ma_aupr = ma_val_aupr
                best_mi_aupr = mi_val_aupr
                best_epoch = epoch + 1
                best_cm = cm
                best_micro_fpr = fpr["micro"]
                best_micro_tpr = tpr["micro"]
                best_macro_fpr = fpr["macro"]
                best_macro_tpr = tpr["macro"]

                best_micro_recall = recall["micro"]
                best_micro_pre = precision["micro"]
                best_macro_recall = recall["macro"]
                best_macro_pre = precision["macro"]

                best_val_score = val_scores_cpu
                best_val_label = val_labels_cpu


    print('Fold:', k + 1, 'Best Epoch:', best_epoch,
              'best_acc: %.4f' % best_acc,
              'best_pre: %.4f' % best_pre['pre_macro'],
              'best_rec: %.4f' % best_rec['recall_macro'],
              'best_f1: %.4f' % best_f1['f1_macro'],
              'best_ma_auc: %.4f' % best_ma_auc,
              'best_mi_auc: %.4f' % best_mi_auc,
              'best_ma_aupr: %.4f' % best_ma_aupr,
              'best_mi_aupr: %.4f' % best_mi_aupr
              )

    for key, value in best_pre.items():
        if key in pre_result:
            pre_result[key].append(value)
        else:
            pre_result[key] = value

    for key, value in best_rec.items():
        if key in reca_result:
            reca_result[key].append(value)
        else:
            reca_result[key] = value

    for key, value in best_f1.items():
        if key in f1_result:
            f1_result[key].append(value)
        else:
            f1_result[key] = value

    acc_result.append(best_acc)
    cumulative_cm += best_cm
    average_cm = cumulative_cm / k_folds
    ma_auc_result.append(best_ma_auc)
    mi_auc_result.append(best_mi_auc)
    ma_aupr_result.append(best_ma_aupr)
    mi_aupr_result.append(best_mi_aupr)

    micro_fpr_result.append(best_micro_fpr)
    micro_tpr_result.append(best_micro_tpr)
    macro_fpr_result.append(best_macro_fpr)
    macro_tpr_result.append(best_macro_tpr)

    micro_recall_result.append(best_micro_recall)
    micro_pre_result.append(best_micro_pre)
    macro_recall_result.append(best_macro_recall)
    macro_pre_result.append(best_macro_pre)

    tsne_scores_result.append(best_val_score)
    tsne_labels_result.append(best_val_label)

tsne_scores_result = np.vstack(tsne_scores_result)
tsne_labels_result = np.concatenate(tsne_labels_result)


##t-SNE 画图用的数据
pd.DataFrame(tsne_scores_result).to_csv('../result/tsne_scores.csv', index=False)
pd.DataFrame(tsne_labels_result).to_csv('../result/tsne_labels.csv', index=False)

pd.DataFrame(micro_fpr_result).to_csv('../result/micro_fpr.csv', index=False)
pd.DataFrame(micro_tpr_result).to_csv('../result/micro_tpr.csv', index=False)
pd.DataFrame(macro_fpr_result).to_csv('../result/macro_fpr.csv', index=False)
pd.DataFrame(macro_tpr_result).to_csv('../result/macro_tpr.csv', index=False)

pd.DataFrame(micro_recall_result).to_csv('../result/micro_recall.csv', index=False)
pd.DataFrame(micro_pre_result).to_csv('../result/micro_pre.csv', index=False)
pd.DataFrame(macro_recall_result).to_csv('../result/macro_recall.csv', index=False)
pd.DataFrame(macro_pre_result).to_csv('../result/macro_pre.csv', index=False)

averages_pre = {key: round(np.mean(values), 4) for key, values in pre_result.items()}   #平均值
pre_std_devs = {key: round(np.std(values), 4) for key, values in pre_result.items()}    #标准差
# print('pre_result', pre_result)
# print('averages_pre', averages_pre)
# print('pre_std_devs', pre_std_devs)

averages_recall = {key: round(np.mean(values), 4) for key, values in reca_result.items()}
recall_std_devs = {key: round(np.std(values), 4) for key, values in reca_result.items()}
# print('reca_result', reca_result)
# print('averages_recall', averages_recall)
# print('recall_std_devs', recall_std_devs)

averages_f1 = {key: round(np.mean(values), 4) for key, values in f1_result.items()}
f1_std_devs = {key: round(np.std(values), 4) for key, values in f1_result.items()}
# print('f1_result', f1_result)
# print('averages_f1', averages_f1)
# print('f1_std_devs', f1_std_devs)


print('## Training Finished !')
print('-----------------------------------------------------------------------------------------------')
print('acc: %s, mean: %.4f, variance: %.4f \n' % (acc_result, np.mean(acc_result), np.std(acc_result)))

print("Average Confusion Matrix:\n", average_cm)

print('macro_auc: %s, mean: %.4f, variance: %.4f \n' % (ma_auc_result, np.mean(ma_auc_result), np.std(ma_auc_result)))
print('micro_auc: %s, mean: %.4f, variance: %.4f \n' % (mi_auc_result, np.mean(mi_auc_result), np.std(mi_auc_result)))

print('macro_aupr: %s, mean: %.4f, variance: %.4f \n' % (ma_aupr_result, np.mean(ma_aupr_result), np.std(ma_aupr_result)))
print('micro_aupr: %s, mean: %.4f, variance: %.4f \n' % (mi_aupr_result, np.mean(mi_aupr_result), np.std(mi_aupr_result)))

print('micro_pre: %s, mean: %.4f, variance: %.4f \n' % (pre_result['pre_micro'], averages_pre['pre_micro'], pre_std_devs['pre_micro']))
print('macro_pre: %s, mean: %.4f, variance: %.4f \n' % (pre_result['pre_macro'], averages_pre['pre_macro'], pre_std_devs['pre_macro']))


print('micro_recall: %s, mean: %.4f, variance: %.4f \n' % (reca_result['recall_micro'], averages_recall['recall_micro'], recall_std_devs['recall_micro']))
print('macro_recall: %s, mean: %.4f, variance: %.4f \n' % (reca_result['recall_macro'], averages_recall['recall_macro'], recall_std_devs['recall_macro']))


print('micro_f1: %s, mean: %.4f, variance: %.4f \n' % (f1_result['f1_micro'], averages_f1['f1_micro'], f1_std_devs['f1_micro']))
print('macro_f1: %s, mean: %.4f, variance: %.4f \n' % (f1_result['f1_macro'], averages_f1['f1_macro'], f1_std_devs['f1_macro']))

"""SNE Visualization"""
plot_T_SNE(tsne_scores_result, tsne_labels_result, directory='../result', name='T_SNE')       #这两个列表专门用来画t-SNE 降维可视化图，把模型学到的 高维特征（比如 64 维） 压缩成 2 维，然后画成散点图，看同类样本聚不聚在一起。

# Confusion Matrix Visualization
plot_cm(average_cm, directory='../result', name='average_cm')

