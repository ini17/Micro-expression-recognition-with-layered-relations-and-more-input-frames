import argparse
import os
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
from models.FMER import FMER
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parallel = 12
# notes = "尝试直接将联合后的空间特征拉平成160维"
# notes = "原文方法仅替换使用EVM"
description = "modelABC_v1"
# notes = "在第一层GCN中添加LAM。test"
# notes = f"SMIC 3分类——UAR、UF1"
notes = f"CASME II 4分类"
# experiments = "FGRMER"
experiments = "Ablative_Analysis"


def train(epochs: int, criterion: nn.Module, optimizer: torch.optim,
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
        # params = [param for param in model.graph.DWConv[0].parameters()]
        params = [param for param in model.au_gcn.graph_weight_one_List[0].parameters()]
        # print(params)
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

        print(f"Epoch: {epoch + 1}")
        print(f"Loss: {train_loss}")
        print(f"Accuracy: {train_accuracy}")

        if train_accuracy > best_accuracy:
            torch.save(model.state_dict(), model_best_name)
            best_accuracy = train_accuracy
            print("Save model")


def evaluate2(test_loader: DataLoader, model1: nn.Module, model2: nn.Module, device: torch.device):
    # Set into evaluation mode
    model1.eval()
    model2.eval()
    test_accuracy1 = 0.0
    test_f1_score1 = 0.0
    test_accuracy2 = 0.0
    test_f1_score2 = 0.0
    params1 = [param for param in model1.parameters()]
    params2 = [param for param in model2.parameters()]
    ifequal = all(torch.allclose(p1.data, p2.data, atol=1e-4) for p1, p2 in zip(params1, params2))
    print(f"两个模型的参数是否相等：{ifequal}")

    with torch.no_grad():
        for patches, labels in test_loader:
            # Move data to device and compute the output
            patches = patches.to(device)
            labels = labels.to(device)

            output = model1(patches)
            output2 = model2(patches)

            # Compute the accuracy
            prediction1 = (output.argmax(-1) == labels)
            test_accuracy1 += prediction1.sum().item() / labels.size(0)
            test_f1_score1 += f1_score(labels.cpu().numpy(), output.argmax(-1).cpu().numpy(),
                                       average="weighted")

            prediction2 = (output2.argmax(-1) == labels)
            test_accuracy2 += prediction2.sum().item() / labels.size(0)
            test_f1_score2 += f1_score(labels.cpu().numpy(), output2.argmax(-1).cpu().numpy(),
                                       average="weighted")
    ret = [test_accuracy1 / len(test_loader), test_f1_score1 / len(test_loader),
           test_accuracy2 / len(test_loader), test_f1_score2 / len(test_loader)]
    return ret


def evaluate(test_loader: DataLoader, model: nn.Module, device: torch.device):
    # Set into evaluation mode
    model.eval()
    test_accuracy = 0.0
    test_f1_score = 0.0

    with torch.no_grad():
        for patches, labels in test_loader:
            # Move data to device and compute the output
            patches = patches.to(device)
            labels = labels.to(device)

            output = model(patches)
            # output_values = output.argmax(-1)
            # for x in labels:
            #     label_count[x] += 1

            # Compute the accuracy
            predict_label = output.argmax(-1)
            print(predict_label)
            prediction = (output.argmax(-1) == labels)
            test_accuracy += prediction.sum().item() / labels.size(0)
            test_f1_score += f1_score(labels.cpu().numpy(), output.argmax(-1).cpu().numpy(),
                                      average="weighted")

    return test_accuracy / len(test_loader), test_f1_score / len(test_loader)


def LOSO_train(data: pd.DataFrame, sub_column: str, args,
               label_mapping: dict, device: torch.device):
    log_file = open("train.log", "w")

    # Create different DataFrame for each subject
    train_list, test_list = LOSO_sequence_generate(data, sub_column)
    test_accuracy = 0.0
    test_f1_score = 0.0

    for idx in range(len(train_list)):
        npz_file = np.load(f"{args.npz_file}/{idx}.npz")
        adj_matrix = torch.FloatTensor(npz_file["adj_matrix"]).to(device)

        print(f"=================LOSO {idx + 1}=====================")
        train_csv = train_list[idx]
        test_csv = test_list[idx]
        print("当前测试的subject为" + test_csv["Subject"][0])

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
                                    mat_dir=args.mat_dir,
                                    shuffle=False)

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
        train(epochs=args.epochs,
              criterion=criterion,
              optimizer=optimizer,
              scheduler=None,
              model=model,
              train_loader=train_loader,
              device=device,
              model_best_name=f"{args.weight_save_path}/model_best_{idx}.pt")

        # model2 = FMER(adj_matrix=adj_matrix,
        #               num_classes=args.num_classes,
        #               parallel=args.parallel,
        #               device=device).to(device)
        model.load_state_dict(torch.load(f"{args.weight_save_path}/model_best_{idx}.pt",
                                         map_location=device), strict=True)

        # model2.load_state_dict(torch.load(f"{args.weight_save_path}/model_best_{idx+1}.pt",
        #                                   map_location=device), strict=True)
        # params1 = [param for param in model.parameters()]
        # params2 = [param for param in model2.parameters()]
        # params = [(p1, p2) for p1, p2 in zip(params1, params2)]

        temp_test_accuracy, temp_f1_score = evaluate(test_loader=test_loader,
                                                     model=model,
                                                     device=device)
        print(f"In LOSO {idx + 1}, test accuracy: {temp_test_accuracy:.4f}, f1-score: {temp_f1_score:.4f}")
        test_accuracy += temp_test_accuracy
        test_f1_score += temp_f1_score

        # ret = evaluate2(test_loader=test_loader, model1=model, model2=model2, device=device)
        # print(f"In LOSO {idx + 1}, original model test accuracy: {ret[0]:.4f}, f1-score: {ret[1]:.4f}")
        # print(f"In LOSO {idx + 1}, reloaded model test accuracy: {ret[2]:.4f}, f1-score: {ret[3]:.4f}")

        # log_file.write(f"LOSO {idx + 1}: Accuracy: {temp_test_accuracy:.4f}, F1-Score: {temp_f1_score:.4f}\n")

        # test_accuracy += ret[0]
        # test_f1_score += ret[1]

    loso_accuracy = test_accuracy / len(train_list)
    loso_f1_score = test_f1_score / len(train_list)
    print(f"LOSO accuracy: {loso_accuracy:.4f}, f1-score: {loso_f1_score:.4f}")
    log_file.write(
        f"Total: Accuracy {loso_accuracy:.4f}, F1-Score: {loso_f1_score:.4f}\n")
    log_file.close()
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
                        default=10,
                        help="Epochs for training the model")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-4,
                        help="Learning rate for training the model")
    args = parser.parse_args()

    # Training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read in the data
    data, label_mapping = read_csv(args.csv_path)

    # Create folders for the saving weight
    os.makedirs(args.weight_save_path, exist_ok=True)

    # Train the model
    ret = LOSO_train(data=data,
                     sub_column="Subject",
                     label_mapping=label_mapping,
                     args=args,
                     device=device)
    # df = pd.read_excel("test_result.xlsx", header=0, sheet_name=0, index_col=0)
    # row_num = df.shape[0]
    # for label, counts in zip(label_mapping, label_count):
    #     df.loc[row_num, label] = counts
    # df.loc[row_num, 'accuracy'] = ret[0]
    # df.loc[row_num, 'f1_score'] = ret[1]
    # df.to_excel("test_result.xlsx")


if __name__ == "__main__":
    for i in range(1):
        # label_count = [0, 0, 0, 0]
        main()
