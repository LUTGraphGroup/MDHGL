import matplotlib.pyplot as plt
import numpy as np

models = [
    "MDHGL",
    "W/O‑NODE-MoE",
    "W/O‑MambaEncoder",
    "W/O‑CrossAtt",
    "W/O‑FastKAN"
]
# 指标顺序：Macro‑AUC, Macro‑AUPR, Acc, ma‑Pre, ma‑Recall, ma‑F1, Micro‑AUC, Micro‑AUPR
data = np.array([
    [0.8926, 0.8119, 0.7462, 0.7478, 0.7431, 0.7428, 0.8954, 0.8117],
    [0.8789, 0.7964, 0.7331, 0.7358, 0.7349, 0.7353, 0.8821, 0.7927],
    [0.8746, 0.7990, 0.7364, 0.7331, 0.7264, 0.7238, 0.8807, 0.7983],
    [0.8876, 0.8018, 0.7421, 0.7459, 0.7410, 0.7414, 0.8926, 0.8085],
    [0.8861, 0.8006, 0.7406, 0.7440, 0.7369, 0.7361, 0.8919, 0.8080],
])

data = data * 100
x_labels = ["Ma AUC", "Ma AUPR", "Acc", "Ma Pre", "Ma Recall", "Ma F1", "Mi AUC", "Mi AUPR"]
n_metrics = len(x_labels)
n_models = len(models)

bar_width = 0.10
x = np.arange(n_metrics)

# 蓝色系，由深到浅，主模型最深
# 参考图低饱和配色版本
colors= [
    "#0E606B",   # MDHGL 【和上图完全一致】
    "#1597A5",
    "#FFC24B",
    "#F66F69",
    "#90C9E7"
]


plt.figure(figsize=(18, 7), dpi=150)
for i in range(n_models):
    plt.bar(x + i * bar_width, data[i], width=bar_width*0.92, color=colors[i], label=models[i])

plt.xticks(x + bar_width*(n_models-1)/2, x_labels, rotation=40, fontsize=18)
plt.ylabel("Values(%)", fontsize=20)
plt.tick_params(axis='y', labelsize=14)
plt.legend(loc="upper right", frameon=True, fontsize=14)

plt.ylim(70, 92)
plt.grid(axis='y', alpha=0.25)
plt.tight_layout()

plt.savefig('Ablation_dma.pdf', format='pdf', bbox_inches='tight')
plt.show()
