import os
import argparse

from douzero.evaluation.simulation import evaluate

# rlcard vs mdp vs random vs baselines/sl/landlord_down.ckpt
# 使用训练好的模型（将路径替换为实际的 .ckpt 文件路径）
# method = [
#     "douzero_checkpoints/douzero/landlord_weights_1036800.ckpt",
#     "douzero_checkpoints/douzero/landlord_up_weights_1036800.ckpt",
#     "douzero_checkpoints/douzero/landlord_down_weights_1036800.ckpt",
# ]

method = ["mdp", "rlcard", "rlcard"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dou Dizhu Evaluation")
    parser.add_argument("--landlord", type=str, default=method[0])
    parser.add_argument("--landlord_up", type=str, default=method[1])
    parser.add_argument("--landlord_down", type=str, default=method[2])

    parser.add_argument("--eval_data", type=str, default="eval_data.pkl")
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--gpu_device", type=str, default="")
    args = parser.parse_args()

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    evaluate(
        args.landlord,
        args.landlord_up,
        args.landlord_down,
        args.eval_data,
        args.num_workers,
    )
