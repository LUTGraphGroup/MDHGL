import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()

    # ========= 模型结构参数 =========
    parser.add_argument("--feat_dim", type=int, default=64, help="输入节点特征维度")
    parser.add_argument("--hidden_dim", type=int, default=128, help="隐层维度")  #128
    parser.add_argument("--out_dim", type=int, default=64, help="输出维度")    #64

    # ====特征融合方式=====
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="concat",
        choices=["fc", "cross", "concat"],
        help="mamba融合方式",
    )
    parser.add_argument(
        "--fusion_head", type=int, default=8, help="注意力融合时的头数") #8
    parser.add_argument(
        "--fusion_dropout", type=float, default=0.4, help="注意力融合时的dropout") #0.4

    # ========= moe参数 =========
    parser.add_argument("--num_mamba_layers", type=int, default=1, help="num_deepseekmoe_layers.")
    parser.add_argument("--num_kan_layers", type=int, default=1, help="layers2.")
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')  #0.3
    parser.add_argument('--temperature', type=float, default=0.6, help='temperature')   #0.6
    parser.add_argument('--Lambda', type=float, default=0.3, help='songshi quanzhong')

    # ========= mamba邻居跳数 =========
    parser.add_argument("--hops", type=int, default=6, help="邻居跳数") #6
    parser.add_argument(
        "--agg",
        type=str,
        default="attention",
        choices=["mean", "last","attention"],
        help="mamba编码后的聚合方式",
    )
    parser.add_argument("-- mamba_num_layers", type=int, default=1, help="mamba层") #1  B:1
    parser.add_argument("--mam_dropout", type=float, default=0.2, help="mambaDropout")  # 0.1


    # ========= 优化器与学习率调度 =========
    parser.add_argument("--peak_lr", type=float, default=0.0005, help="初始学习率")  #0.005
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="权重衰减 (L2 正则化)")#5e-4

    # ========= 训练参数 =========
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数") #500
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="禁用 CUDA"
    )
    parser.add_argument('--case',
                        type=float,
                        default=146,
                        help='Case Study.')  # 序号减2

    return parser.parse_args()
