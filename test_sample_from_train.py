import argparse
import os

import cv2
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from read_file import read_csv
from dataloader import (
    get_loader,
    LOSO_sequence_generate
)
import time
from models.FMER import FMER


parallel = 12
# notes = "尝试直接将联合后的空间特征拉平成160维"
# notes = "原文方法仅替换使用EVM"
description = "ModelABC_v1"
# notes = "在第一层GCN中添加LAM。test"
notes = "上下两路均N路并行，并使用全连接层融合N路特征"
experiments = "FGRMER"
# experiments = "JustTry"


def get_key_by_value(dicts, value):
    for key, val in dicts.items():
        if val == value:
            return key


def evaluate(test_loader: DataLoader, model: nn.Module, device: torch.device,
             label_mapping: dict):
    # Set into evaluation mode
    model.eval()
    test_accuracy = 0.0
    test_f1_score = 0.0
    num = 0
    all_out_labels = []
    all_gt_labels = []

    with torch.no_grad():
        for patches, labels in test_loader:
            num += len(patches)
            # Move data to device and compute the output
            patches = patches.to(torch.float32)
            patches = patches.to(device)
            labels = labels.to(device)

            output = model(patches)

            # Compute the accuracy
            out_values = output.argmax(-1)
            prediction = (output.argmax(-1) == labels)

            out_labels = []
            gt_labels = []
            for x in out_values:
                out_labels.append(get_key_by_value(label_mapping, x.item()))
            for x in labels:
                gt_labels.append(get_key_by_value(label_mapping, x))
            all_out_labels.extend(out_labels)
            all_gt_labels.extend(gt_labels)
            test_accuracy += prediction.sum().item() / labels.size(0)
            test_f1_score += f1_score(labels.cpu().numpy(), output.argmax(-1).cpu().numpy(),
                                      average="weighted")
        print(f"Total number of samples in test_loader is {num}.")
    print(all_out_labels)
    print(all_gt_labels)
    return test_accuracy / len(test_loader), test_f1_score / len(test_loader)


def LOSO_train(data: pd.DataFrame, sub_column: str, args,
               label_mapping: dict, device: torch.device):

    # Create different DataFrame for each subject
    train_list, test_list = LOSO_sequence_generate(data, sub_column)
    test_accuracy = 0.0
    test_f1_score = 0.0

    for idx in range(len(train_list)):
        npz_file = np.load(f"{args.npz_file}/{idx}.npz")  # ground truth adj matrix
        adj_matrix = torch.FloatTensor(npz_file["adj_matrix"]).to(device)

        print(f"=================LOSO {idx + 1}=====================")
        train_csv = train_list[idx]
        test_csv = test_list[idx]
        print("当前测试集的Subject为：" + test_csv['Subject'][0])

        # Create dataset and dataloader
        _, train_loader = get_loader(csv_file=train_csv,
                                     label_mapping=label_mapping,
                                     batch_size=args.batch_size,
                                     catego=args.catego,
                                     parallel=args.parallel,
                                     mat_dir=args.mat_dir)
        _, test_loader = get_loader(csv_file=test_csv,
                                    label_mapping=label_mapping,
                                    batch_size=len(test_csv),
                                    catego=args.catego,
                                    parallel=args.parallel,
                                    train=False,
                                    mat_dir=args.mat_dir)

        # Read in the model
        model = FMER(adj_matrix=adj_matrix,
                     num_classes=args.num_classes,
                     parallel=args.parallel,
                     device=device).to(device)

        model.load_state_dict(torch.load(f"{args.weight_save_path}/model_best_{idx}.pt",
                                         map_location=device))
        # model.load_state_dict(torch.load(f"{args.weight_save_path}/train_epoch_{idx}.pt",
        #                                  map_location=device))

        temp_test_accuracy, temp_f1_score = evaluate(test_loader=test_loader,
                                                     model=model,
                                                     device=device,
                                                     label_mapping=label_mapping)

        print(f"In LOSO {idx + 1}, test accuracy: {temp_test_accuracy:.4f}, f1-score: {temp_f1_score:.4f}")

        test_accuracy += temp_test_accuracy
        test_f1_score += temp_f1_score
    print(f"LOSO accuracy: {test_accuracy / len(train_list):.4f}, f1-score: {test_f1_score / len(train_list):.4f}")


def main():
    # Argument parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel",
                        type=int,
                        default=parallel,
                        help="How many paths to do parallel operations")
    parser.add_argument("--csv_path",
                        type=str,
                        # required=True,
                        default=r"B:\0_0NewLife\datasets\CASME_2\4classes.csv",  # CASME2
                        # default=r"B:\0_0NewLife\datasets\SMIC\HS_cropped.csv",  # SMIC
                        help="Path for the csv file for training data")
    parser.add_argument("--mat_dir",
                        type=str,
                        default=r"B:\0_0NewLife\0_Papers\FGRMER\CASME2\mat\MagNet",  # CASME2
                        # default=r"B:\0_0NewLife\0_Papers\FGRMER\SMIC\mat\MagNet",  # SMIC
                        help="Path for the mat files")
    parser.add_argument("--npz_file",
                        type=str,
                        # required=True,
                        # default=r"B:\0_0NewLife\0_Papers\FGRMER\CASME2\npz\(OpenFace)4-npz",
                        # default=r"B:\0_0NewLife\0_Papers\FGRMER\CASME2\npz\RAW_selected_Inter_10_4npz",
                        default=r"B:\0_0NewLife\0_Papers\FGRMER\CASME2\npz\4-npz",  # CASME2
                        # default=r"B:\0_0NewLife\0_Papers\FGRMER\SMIC\npz",  # CASME2
                        help="Files' root for npz")
    parser.add_argument("--catego",
                        type=str,
                        # required=True,
                        default="CASME",
                        help="SAMM or CASME dataset")
    parser.add_argument("--num_classes",
                        type=int,
                        default=4,
                        help="Classes to be trained")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Training batch size")
    parser.add_argument("--weight_save_path",
                        type=str,
                        default="model",
                        help="Path for the saving weight")
    parser.add_argument("--epochs",
                        type=int,
                        default=25,
                        help="Epochs for training the model")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-4,
                        help="Learning rate for training the model")
    args = parser.parse_args()

    mlflow.set_tracking_uri("../../../mlruns")
    mlflow.set_experiment(experiments)
    mlflow.start_run(run_name=notes)

    # Training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read in the data
    data, label_mapping = read_csv(args.csv_path)

    # Create folders for the saving weight
    os.makedirs(args.weight_save_path, exist_ok=True)

    # Train the model
    LOSO_train(data=data,
               sub_column="Subject",
               label_mapping=label_mapping,
               args=args,
               device=device)
    mlflow.end_run()


if __name__ == "__main__":
    main()
