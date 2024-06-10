import argparse
import os
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
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parallel = 12
description = "modelABC_v1"
notes = f"CASME II 4分类"
experiments = "Ablative_Analysis"


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    # for p, t in zip(preds, labels):
    for p, t in zip(labels, preds):
        conf_matrix[p, t] += 1
    return conf_matrix


def mlflow_log(args):
    mlflow.log_params({
        "current_time": str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))),
        "csv_path": args.csv_path,
        "npz_file": args.npz_file,
        "catego": args.catego,
        "num_classes": args.num_classes,
        "batch_size": args.batch_size,
        "weight_save_path": args.weight_save_path,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate
    })
    mlflow.set_tag("description", description)


def train(LOSO: int, epochs: int, criterion: nn.Module, optimizer: torch.optim,
          model: nn.Module, scheduler: torch.optim.lr_scheduler, train_loader: DataLoader,
          device: torch.device, model_best_name: str):
    """Train the model

    Parameters
    ----------
    epochs : int
        Epochs for training the model
    model : DSSN
        Model to be trained
    train_loader : DataLoader
        DataLoader to load in the data
    device: torch.device
        Device to be trained on
    model_best_name: str
        Name of the weight file to be saved
    """
    best_accuracy = -1
    # Set model in training mode
    model.train()

    for epoch in range(epochs):
        train_loss = 0.0
        train_accuracy = 0.0

        for patches, labels in train_loader:
            patches = patches.to(device)
            labels = labels.to(device)

            output = model(patches)

            # Compute the loss
            loss = criterion(output, labels)
            train_loss += loss.item()

            # Update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute the accuracy
            prediction = (output.argmax(-1) == labels)
            train_accuracy += prediction.sum().item() / labels.size(0)

        if scheduler is not None:
            scheduler.step()

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        mlflow.log_metrics({
            f"Test Accuracy in LOSO {str(LOSO + 1).zfill(2)}": float("%.2f" % (100 * train_accuracy))
        }, step=epoch + 1)

        print(f"Epoch: {epoch + 1}")
        print(f"Loss: {train_loss}")
        print(f"Accuracy: {train_accuracy}")

        if train_accuracy > best_accuracy:
            torch.save(model.state_dict(), model_best_name)
            best_accuracy = train_accuracy
            print("Save model")
            if best_accuracy == 1:
                return 1
    return best_accuracy


def evaluate(test_loader: DataLoader, model: nn.Module, conf_matrix: torch.Tensor, device: torch.device):
    # Set into evaluation mode
    model.eval()
    test_accuracy = 0.0
    test_f1_score = 0.0
    num = 0

    with torch.no_grad():
        for patches, labels in test_loader:
            num += len(patches)
            # Move data to device and compute the output
            patches = patches.to(device)
            labels = labels.to(device)
            targets = labels.squeeze()

            output = model(patches)

            # Compute the accuracy
            prediction = (output.argmax(-1) == labels)
            test_accuracy += prediction.sum().item() / labels.size(0)
            test_f1_score += f1_score(labels.cpu().numpy(), output.argmax(-1).cpu().numpy(),
                                      average="weighted")

            conf_matrix = confusion_matrix(output, targets, conf_matrix)

        print(f"Total number of samples in test_loader is {num}.")

    return test_accuracy / len(test_loader), test_f1_score / len(test_loader), conf_matrix


def LOSO_train(data: pd.DataFrame, sub_column: str, args,
               label_mapping: dict, device: torch.device):
    # Create different DataFrame for each subject
    train_list, test_list = LOSO_sequence_generate(data, sub_column)
    test_accuracy = 0.0
    test_f1_score = 0.0
    conf_matrix = torch.zeros(args.num_classes, args.num_classes)

    for idx in range(len(train_list)):
        npz_file = np.load(f"{args.npz_file}/{idx}.npz")  # ground truth adj matrix
        adj_matrix = torch.FloatTensor(npz_file["adj_matrix"]).to(device)

        print(f"=================LOSO {idx + 1}=====================")
        train_csv = train_list[idx]
        test_csv = test_list[idx]

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

        if idx != 0:
            model.load_state_dict(torch.load(f"{args.weight_save_path}/model_best_{idx - 1}.pt",
                                             map_location=device))

        # Create criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate)

        # Train the data
        best_accuracy = train(LOSO=idx,
                              epochs=args.epochs,
                              criterion=criterion,
                              optimizer=optimizer,
                              scheduler=None,
                              model=model,
                              train_loader=train_loader,
                              device=device,
                              model_best_name=f"{args.weight_save_path}/model_best_{idx}.pt")
        model.load_state_dict(torch.load(f"{args.weight_save_path}/train_epoch_{idx}.pt",
                                         map_location=device))

        temp_test_accuracy, temp_f1_score, conf_matrix = evaluate(test_loader=test_loader,
                                                                  model=model,
                                                                  conf_matrix=conf_matrix,
                                                                  device=device)
        print(f"In LOSO {idx + 1}, test accuracy: {temp_test_accuracy:.4f}, f1-score: {temp_f1_score:.4f}")
        mlflow.log_metrics({
            "Test Accuracy": float("%.2f" % (100 * temp_test_accuracy)),
            "Test F1-Score": float("%.2f" % (100 * temp_f1_score)),
            "Best Accuracy": float("%.2f" % (100 * best_accuracy))
        }, step=idx + 1)
        test_accuracy += temp_test_accuracy
        test_f1_score += temp_f1_score

    loso_accuracy = test_accuracy / len(train_list)
    loso_f1_score = test_f1_score / len(train_list)
    print(f"LOSO accuracy: {loso_accuracy:.4f}, f1-score: {loso_f1_score:.4f}")
    mlflow.log_metrics({
        "LOSO Accuracy": float(format(100 * loso_accuracy, '.2f')),
        "LOSO F1-Score": float(format(100 * loso_f1_score, '.2f'))
    })

    conf_matrix = np.array(conf_matrix.cpu())
    corrects = conf_matrix.diagonal(offset=0)
    per_kinds = conf_matrix.sum(axis=1)
    print(conf_matrix)
    # 获取每种Emotion的识别准确率
    print("每种情感总个数：", per_kinds)
    print("每种情感预测正确的个数：", corrects)
    print("每种情感的识别准确率为：{0}".format([rate * 100 for rate in corrects / per_kinds]))

    # 绘制混淆矩阵
    emotion = 4  # 这个数值是具体的分类数，大家可以自行修改
    labels = ['negative', 'others', 'positive', 'surprise']

    # 显示数据
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(emotion):
        for y in range(emotion):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")

    plt.tight_layout()  # 保证图不重叠
    plt.yticks(range(emotion), labels)
    plt.xticks(range(emotion), labels, rotation=45)  # X轴字体倾斜45°
    # plt.show()
    plt.savefig("test.png")
    plt.close()

    return loso_accuracy, loso_f1_score


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
                        help="Path for the csv file for training data")
    parser.add_argument("--mat_dir",
                        type=str,
                        default=r"B:\0_0NewLife\0_Papers\FGRMER\CASME2\mat\MagNet",  # CASME2
                        help="Path for the mat files")
    parser.add_argument("--npz_file",
                        type=str,
                        required=True,
                        default=r"B:\0_0NewLife\0_Papers\FGRMER\CASME2\npz\4-npz",  # CASME2
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

    # Start logging with mlflow
    mlflow_log(args)

    # Train the model
    ret = LOSO_train(data=data,
                     sub_column="Subject",
                     label_mapping=label_mapping,
                     args=args,
                     device=device)
    mlflow.end_run()
    return ret


if __name__ == "__main__":
    main()
