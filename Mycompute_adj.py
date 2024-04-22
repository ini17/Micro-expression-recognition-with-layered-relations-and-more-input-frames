import re
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataloader import LOSO_sequence_generate


# 这里尝试根据OpenFace输出的AU发生情况来计算邻接矩阵
# 并且，这里需要注意的是，在每个视频样本中仅使用了apex帧的AU信息进行计算
parallel = 10
AU_CODE = ["01", "02", "04", "10", "12", "14", "15", "17", "25"]


def evaluate_adj(df, arg, npz_index):
    assert isinstance(df, (str, pd.DataFrame)), "Type not supported"
    if isinstance(df, str):
        # Read in data
        df = pd.read_csv(args.csv_name)

    for n_path in range(1, parallel):
        # 创建一个空矩阵来为邻接矩阵计数
        count_matrix = np.zeros((9, 9))

        # 创建一个空列来计算AU发生次数
        count_au = np.zeros(9)
        for idx in range(df.shape[0]):
            # 获取每个样本的Subject和filename，以及他们的onset和apex帧位置
            subject = df.loc[idx, "Subject"]
            filename = df.loc[idx, "Filename"]
            frame_loc = n_path
            # data = pd.read_csv(f"{args.OpenFace_place}/sub{str(subject).zfill(2)}/{filename}.csv").loc[frame_loc]
            # 这个地方折磨我半天，后来发现OpenFace输出文件中列名前面是有个空格的
            loc_name = [f" AU{AU_name}_c" for AU_name in AU_CODE]
            data = pd.read_csv(f"{args.OpenFace_place}/sub{str(subject).zfill(2)}/{filename}.csv").loc[frame_loc, loc_name]
            aus = []
            for index, value in enumerate(data):
                if value == 1.:
                    aus.append(index)
            for i in range(len(aus)):
                first_code = aus[i]
                for j in range(i+1, len(aus)):
                    second_code = aus[j]
                    count_matrix[first_code, second_code] += 1
                    count_matrix[second_code, first_code] += 1
                count_au[first_code] += 1

        # Replace 0 in count_au to 1
        # 防止出现除以0的情况
        # 如果有某个AU出现次数为0，其概率就相当于除以自身（防止出现0除以0）
        count_au = np.where(count_au == 0.0, 1, count_au)

        # Compute the adjacent matrix
        # count_matrix的形状为(9, 9)，count_at.reshape(-1, 1)之后的形状为(9, 1)是一个列向量
        # 当矩阵除以列向量的时候，个人理解是将矩阵的每一列拿出来单独进行逐元素相除
        # 因此，这里第一行除以了count_au[0]，第二行除以了count_au[1]，以此类推
        adj_matrix = count_matrix / count_au.reshape(-1, 1)

        # Show the information
        print("AU appers:\n", count_au)

        if arg["save_img"]:
            plt.matshow(adj_matrix, cmap="summer")
            for (i, j), z in np.ndenumerate(adj_matrix):
                plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

            plt.savefig(arg["jpg_name"], format="svg", dpi=1200)

        out_dir = arg["npz_name"] + f"/{npz_index}_{n_path}.npz"
        np.savez(out_dir,
                 adj_matrix=adj_matrix)


def save_LOSO_adj(args):
    data = pd.read_csv(args.label_csv_name)
    train_list, _ = LOSO_sequence_generate(data, "Subject")
    os.makedirs(args.npz_place, exist_ok=True)
    for idx, train_info in enumerate(train_list):
        evaluate_adj(df=train_info,
                     arg={
                         "npz_name": f"{args.npz_place}",
                         "jpg_name": f"{args.image_place}/{idx}.svg",
                         "save_img": args.save_img
                     },
                     npz_index=idx)


if __name__ == "__main__":
    # Argument parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_csv_name",
                        type=str,
                        # required=True,
                        default=r"B:\0_0NewLife\CASME_2\4classes.csv",
                        help="Filename")
    parser.add_argument("--OpenFace_place",
                        type=str,
                        # required=True,
                        default=f"B:\\0_0NewLife\\0_Papers\\FGRMER\\CASME2\\"
                                f"OpenFace_Output\\RAW_selected_EVM_Inter_{parallel}",
                        help="root for OpenFace outputs")
    parser.add_argument("--npz_place",
                        type=str,
                        # required=True,
                        # default="(git)4-npz",
                        # default=r"B:\0_0NewLife\0_Papers\FGRMER\test\(OpenFace)npz",
                        default=rf"B:\0_0NewLife\0_Papers\FGRMER\CASME2\npz\RAW_selected_Inter_{parallel}_4npz",
                        help="The root place for saving npz files")
    parser.add_argument("--save_img",
                        action="store_true",
                        default=False)
    parser.add_argument("--image_place",
                        type=str,
                        default=None,
                        help="The root place for saving images")
    args = parser.parse_args()

    save_LOSO_adj(args)
