import argparse
import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score
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


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
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


def train_and_evaluate(LOSO: int, epochs: int, criterion: nn.Module, optimizer: torch.optim,
                       model: nn.Module, scheduler: torch.optim.lr_scheduler,
                       train_loader: DataLoader, test_loader: DataLoader,
                       device: torch.device, model_best_name: str):
    best_train_accuracy = -1
    best_validation_accuracy = -1
    best_validation_f1_score = -1
    # Set model in training mode
    for epoch in range(epochs):
        train_loss = 0.0
        train_accuracy = 0.0

        model.train()

        for patches, labels in train_loader:
            patches = patches.to(torch.float32)
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

        if scheduler is not None and epoch <= 50:
            scheduler.step()

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)

        mlflow.log_metrics({
            f"Train Accuracy in LOSO {str(LOSO + 1).zfill(2)}": float("%.2f" % (100 * train_accuracy))
        }, step=epoch + 1)

        print(f"In LOSO {LOSO + 1}, epoch {epoch + 1}, train loss: {train_loss}, accuracy: {train_accuracy}.")
        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy

        ## test
        model.eval()
        with torch.no_grad():
            test_accuracy = 0.0
            test_f1_score = 0.0

            for patches, labels in test_loader:
                # Move data to device and compute the output
                patches = patches.to(torch.float32)
                patches = patches.to(device)
                labels = labels.to(device)

                output = model(patches)

                # Compute the accuracy
                prediction = (output.argmax(-1) == labels)
                test_accuracy += prediction.sum().item() / labels.size(0)
                test_f1_score += f1_score(labels.cpu().numpy(), output.argmax(-1).cpu().numpy(),
                                          average="weighted")
            test_accuracy = test_accuracy / len(test_loader)
            test_f1_score = test_f1_score / len(test_loader)
            print(f"In LOSO {LOSO + 1}, epoch {epoch + 1},"
                  f" test accuracy: {test_accuracy:.4f}, f1-score: {test_f1_score:.4f}")
            if test_accuracy > best_validation_accuracy:
                torch.save(model.state_dict(), model_best_name)
                print("Save best test model at %s" % model_best_name)
                best_validation_accuracy = test_accuracy
                best_validation_f1_score = test_f1_score
                # if best_validation_accuracy == 1:
                #     return best_validation_accuracy, best_validation_f1_score
    torch.save(model.state_dict(), f"model/train_epoch_{LOSO}.pt")
    print("Save final train model at %s" % f"model/train_epoch_{LOSO}.pt")
    return best_validation_accuracy, best_validation_f1_score


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
            f"Test Accuracy in LOSO {str(LOSO+1).zfill(2)}": float("%.2f" % (100 * train_accuracy))
        }, step=epoch+1)

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


def evaluate(test_loader: DataLoader, model: nn.Module, device: torch.device):
    # Set into evaluation mode
    model.eval()
    test_accuracy = 0.0
    test_f1_score = 0.0
    num = 0

    with torch.no_grad():
        for patches, labels in test_loader:
            num += len(patches)
            # Move data to device and compute the output
            # patches = patches.to(torch.float32)
            patches = patches.to(device)
            labels = labels.to(device)

            output = model(patches)

            # Compute the accuracy
            prediction = (output.argmax(-1) == labels)
            test_accuracy += prediction.sum().item() / labels.size(0)
            test_f1_score += f1_score(labels.cpu().numpy(), output.argmax(-1).cpu().numpy(),
                                      average="weighted")
            # test_accuracy += recall_score(labels.cpu().numpy(), output.argmax(-1).cpu().numpy(),
            #                               average="macro")
            # test_f1_score += f1_score(labels.cpu().numpy(), output.argmax(-1).cpu().numpy(),
            #                           average="macro")
        print(f"Total number of samples in test_loader is {num}.")

    return test_accuracy / len(test_loader), test_f1_score / len(test_loader)


def LOSO_train(data: pd.DataFrame, sub_column: str, args,
               label_mapping: dict, device: torch.device):

    # Create different DataFrame for each subject
    train_list, test_list = LOSO_sequence_generate(data, sub_column)
    test_accuracy = 0.0
    test_f1_score = 0.0

    for idx in range(len(train_list)):
        # 实际上仅需要一个
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

        if idx != 0:
            model.load_state_dict(torch.load(f"{args.weight_save_path}/train_epoch_{idx-1}.pt",
                                             map_location=device))

        # Create criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate)
        # model.load_state_dict(torch.load(f"{args.weight_save_path}/model_best_{idx}.pt",
        #                                  map_location=device))

        temp_test_accuracy, temp_f1_score = \
            train_and_evaluate(LOSO=idx,
                               epochs=args.epochs,
                               criterion=criterion,
                               optimizer=optimizer,
                               model=model,
                               scheduler=None,
                               train_loader=train_loader,
                               test_loader=test_loader,
                               device=device,
                               model_best_name=f"{args.weight_save_path}/model_best_{idx}.pt")

        print(f"In LOSO {idx + 1}, test accuracy: {temp_test_accuracy:.4f}, f1-score: {temp_f1_score:.4f}")
        mlflow.log_metrics({
            "Test Accuracy": float("%.2f" % (100 * temp_test_accuracy)),
            "Test F1-Score": float("%.2f" % (100 * temp_f1_score)),
            # "Best Accuracy": float("%.2f" % (100 * best_accuracy))
        }, step=idx+1)
        test_accuracy += temp_test_accuracy
        test_f1_score += temp_f1_score

    loso_accuracy = test_accuracy / len(train_list)
    loso_f1_score = test_f1_score / len(train_list)
    print(f"LOSO accuracy: {loso_accuracy:.4f}, f1-score: {loso_f1_score:.4f}")
    mlflow.log_metrics({
        "LOSO Accuracy": float(format(100 * loso_accuracy, '.2f')),
        "LOSO F1-Score": float(format(100 * loso_f1_score, '.2f'))
    })
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
