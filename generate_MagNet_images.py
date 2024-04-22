import os.path
import random
from typing import Union
import numpy as np
import pandas as pd
import os
import cv2
import torch
import scipy.io as scio
from torchvision import transforms
from dataloader.magnet import MagNet
from dataloader.landmarks import detect_landmarks


parallel = 12  # N，即N路增强的输出
category = "CASME2"
# category = "MMEW"
# csv_path = r"B:\0_0NewLife\datasets\SMIC\HS_cropped.csv"  # SMIC
csv_path = r"B:\0_0NewLife\0_Papers\FGRMER\4classes.csv"  # CASME2
# csv_path = r"B:\0_0NewLife\datasets\MMEW_Final\MMEW_Micro_Exp.csv"  # MMEW

device = torch.device("cuda:0")
magnet = MagNet().to(device)
magnet.load_state_dict(torch.load(r"B:\0_0NewLife\0_Papers\FGRMER\weight\magnet.pt",
                                  map_location=device))
transforms = transforms.ToTensor()


def fine_tune_coordinate(x_0, x_1, y_0, y_1, img):
    if x_0 < 0:
        x_1 = x_1 - x_0
        x_0 = 0
    elif x_1 > img.shape[0]:
        x_0 = x_0 - x_1 + img.shape[0]
        x_1 = img.shape[0]
    if y_0 < 0:
        y_1 = y_1 - y_0
        y_0 = 0
    elif y_1 > img.shape[1]:
        y_0 = y_0 - y_1 + img.shape[1]
        y_1 = img.shape[1]
    return x_0, x_1, y_0, y_1


def center_crop(img: np.array, crop_size: Union[tuple, int]) -> np.array:
    """Returns center cropped image

    Parameters
    ----------
    img : [type]
        Image to do center crop
    crop_size : Union[tuple, int]
        Crop size of the image

    Returns
    -------
    np.array
        Image after being center crop
    """
    width, height = img.shape[1], img.shape[0]

    # Height and width of the image
    mid_x, mid_y = int(width / 2), int(height / 2)

    if isinstance(crop_size, tuple):
        crop_width, crop_hight = int(crop_size[0] / 2), int(crop_size[1] / 2)
    else:
        crop_width, crop_hight = int(crop_size / 2), int(crop_size / 2)
    crop_img = img[mid_y - crop_hight:mid_y + crop_hight, mid_x - crop_width:mid_x + crop_width]

    return crop_img


def unit_preprocessing(unit):
    unit = cv2.resize(unit, (256, 256))
    unit = cv2.cvtColor(unit, cv2.COLOR_BGR2RGB)
    unit = np.transpose(unit / 127.5 - 1.0, (2, 0, 1))
    unit = torch.FloatTensor(unit).unsqueeze(0)
    return unit


def unit_preprocessing_EVM(unit):
    unit = cv2.resize(unit, (256, 256))
    # unit = cv2.cvtColor(unit, cv2.COLOR_BGR2RGB)
    return unit


def magnify_postprocessing(unit):
    # Unnormalized the magnify images
    unit = unit[0].permute(1, 2, 0).contiguous()
    unit = (unit + 1.0) * 127.5

    # Convert back to images resize to (128, 128)
    unit = unit.numpy().astype(np.uint8)
    unit = cv2.cvtColor(unit, cv2.COLOR_RGB2GRAY)
    unit = cv2.resize(unit, (128, 128))
    return unit


def magnify_postprocessing_EVM(unit):
    # Convert back to images resize to (128, 128)
    unit = cv2.cvtColor(unit, cv2.COLOR_BGR2GRAY)
    unit = cv2.resize(unit, (128, 128))
    return unit


def unit_postprocessing(unit):
    unit = unit[0]

    # Normalized the images for each channels
    max_v = torch.amax(unit, dim=(1, 2), keepdim=True)
    min_v = torch.amin(unit, dim=(1, 2), keepdim=True)
    unit = (unit - min_v) / (max_v - min_v)

    # Sum up all the channels and take the average
    unit = torch.mean(unit, dim=0).numpy()

    # Resize to (128, 128)
    unit = cv2.resize(unit, (128, 128))
    return unit


def unit_postprocessing_EVM(unit):
    # 将三个通道的像素值取均值
    unit = np.mean(unit, axis=2)
    # Resize to (128, 128)
    unit = cv2.resize(unit, (128, 128))
    return unit


def get_patches(point: tuple):
    start_x = point[0] - 3
    end_x = point[0] + 4

    start_y = point[1] - 3
    end_y = point[1] + 4

    return start_x, end_x, start_y, end_y


def create_patches(image_root, catego, out_dir):
    # Label for the image
    data_info = pd.read_csv(csv_path,
                            dtype={
                                "Subject": str,
                                "Filename": str
                            })
    data_mat = dict()
    # for index in data_info.shape[0]:
    AMP_LIST = [1.2, 1.4, 1.6, 1.8, 2.0,
                2.2, 2.4, 2.6, 2.8, 3.0]
    for amp_factor in AMP_LIST:
        temp_mat = torch.empty(parallel, 30, 7, 7)
        for index in range(data_info.shape[0]):
            subject = data_info.loc[index, "Subject"]
            folder = data_info.loc[index, "Filename"]
            if subject not in data_mat:
                data_mat[subject] = {}
            if folder not in data_mat[subject]:
                data_mat[subject][folder] = {}
            # Read in the image
            frames = []
            if catego == "SAMM":
                file_path = f"{image_root}/{subject}/{folder}"
            elif catego == "Cropped":
                file_path = f"{image_root}/sub{subject.zfill(2)}/{folder}"
            elif catego == "MMEW":
                file_path = f"{image_root}/sub{subject}/{folder}"
            elif catego == "SMIC":
                file_path = f"{image_root}/sub{subject}/{folder}"
            else:
                file_path = f"{image_root}/sub{subject.zfill(2)}/{folder}"

            for n in range(parallel+1):
                # if n != parallel:
                #     frames.append(cv2.imread(f"{file_path}/img{0}.jpg"))
                # else:
                #     frames.append(cv2.imread(f"{file_path}/img{n}.jpg"))
                frames.append(cv2.imread(f"{file_path}/img{n}.jpg"))

            # 每帧都进行预处理
            for n, frame in enumerate(frames):
                if catego == "SAMM":
                    frames[n] = center_crop(frame, (420, 420))

                # Preprocessing of the image
                frames[n] = unit_preprocessing(frame).to(device)

            with torch.no_grad():
                for idx in range(1, len(frames)):
                    print(subject, folder, amp_factor)
                    # 这里有一点没考虑好：这里把amp_factor在每一个 图像中都随机取了一个值
                    # 因此会产生N路图像放大倍数不相同的情况，应该把amp_factor放在for循环外面
                    # 但是感觉好像影响不大，待会试试看吧

                    # Get the magnify results
                    shape_representation, magnify = magnet(batch_A=frames[0],
                                                           batch_B=frames[idx],  # apex_frame
                                                           batch_C=None,
                                                           batch_M=None,
                                                           amp_factor=amp_factor,
                                                           mode="evaluate")

                    # Do the post processing the transform back to numpy
                    magnify = magnify_postprocessing(magnify.to("cpu"))
                    shape_representation = unit_postprocessing(shape_representation.to("cpu"))

                    # Landmarks detection
                    points = detect_landmarks(magnify)

                    patches = []
                    for point in points:
                        start_x, end_x, start_y, end_y = get_patches(point)
                        start_x, end_x, start_y, end_y = fine_tune_coordinate(start_x, end_x,
                                                                              start_y, end_y, shape_representation)
                        patches.append(
                            transforms(np.expand_dims(shape_representation[start_x:end_x, start_y:end_y], axis=-1))
                        )
                    patches = torch.cat(patches, dim=0)  # [30, 7, 7]
                    temp_mat[idx-1] = patches
            print(f"Amp_factor {amp_factor} is accomplished.")
            # file_dir = f"{out_dir}/Inter_offset_{parallel}"  # offset
            file_dir = f"{out_dir}/Inter_{parallel}"  # apex
            os.makedirs(file_dir, exist_ok=True)
            file_name = f"{file_dir}/sub{subject.zfill(2)}_{folder}_{amp_factor}.pt"
            torch.save(temp_mat, file_name)


if __name__ == '__main__':
    out_dir = rf'B:\0_0NewLife\0_Papers\FGRMER\{category}\mat\MagNet'
    # image_root = f"B:/0_0NewLife/0_Papers/FGRMER/{category}/Interpolation/Inter_offset_{parallel+1}"  # offset
    image_root = f"B:/0_0NewLife/0_Papers/FGRMER/{category}/Interpolation/Inter_{parallel+1}"  # apex
    create_patches(image_root, category, out_dir)
