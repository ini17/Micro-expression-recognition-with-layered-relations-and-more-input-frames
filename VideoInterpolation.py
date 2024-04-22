import os
import cv2
import torch
import pandas as pd
from dataloader.RIFE_HDv3 import Model
from dataloader.RIFE_for_FGRMER import inference_img


parallel = 13  # N+1，即包括onset帧在内
catego = "CASME"
AMP_LIST = [x/10 for x in range(12, 31, 2)]


def format_filedir(path, index, category, subject=None, amp_factor=None):
    if category == "SAMM":
        return f"{path}/{subject}_{index:05}.jpg"
    elif category == "Cropped":
        return f"{path}/reg_img{index}.jpg"
    elif category == "Cropped_EVM":
        return f"{path}/{amp_factor}/reg_img{index}.jpg"
    elif category == "MMEW":
        return f"{path}/{index}.jpg"
    else:
        return f"{path}/img{index}.jpg"


def video_interpolation_cropped_without_offset(model, path, onset, apex, device,
                                               parallel, subject=None, outdir=None, folder=None):
    """
        这里对输入帧进行插值，返回插值后的帧序列
        :param model: RIFE插值模型
        :param path: 用于读取的微表情图片的目录
        :param onset: onset标号
        :param apex: apex标号
        :param device: torch.device
        :param parallel: 得到parallel帧输出
        :param subject: 若为SAMM数据集，则需要一个subject来读取图像
        :param outdir: 是否要将图片保存至目录
        :param folder: 保存为CASME2格式
        :return: 一个包含N帧图像的列表
        """
    for amp_factor in AMP_LIST:
        if amp_factor % 1 == 0:
            amp = int(amp_factor)
        else:
            amp = amp_factor
        frames = []
        for idx in range(parallel):
            if idx < parallel - 1:
                timestamp = onset + idx * (apex - onset) / (parallel - 1)
            else:
                frames.append(cv2.imread(format_filedir(path, apex, "Cropped_EVM", subject, amp)))
                break
            # print(timestamp)
            prev_index = int(timestamp)
            prev_dir = format_filedir(path, prev_index, "Cropped_EVM", subject, amp)
            next_index = prev_index + 1
            next_dir = format_filedir(path, next_index, "Cropped_EVM", subject, amp)
            ratio = timestamp - prev_index
            frames.append(inference_img(model, [prev_dir, next_dir], ratio, device=device))

        if outdir is not None:
            out = f"{outdir}\\sub{subject.zfill(2)}\\{folder}\\{amp_factor}"
            os.makedirs(out, exist_ok=True)
            for index, frame in enumerate(frames):
                cv2.imwrite(f"{out}/img{index}.jpg", frame)
    return frames


def video_interpolation(model, path, onset, offset, device,
                        parallel, category, subject=None, outdir=None, folder=None):
    """
    这里对输入帧进行插值，返回插值后的帧序列
    :param model: RIFE插值模型
    :param path: 用于读取的微表情图片的目录
    :param onset: onset标号
    :param apex: apex标号
    :param device: torch.device
    :param parallel: 得到parallel帧输出
    :param category: 输入的图像类别
    :param subject: 若为SAMM数据集，则需要一个subject来读取图像
    :param outdir: 是否要将图片保存至目录
    :param folder: 保存为CASME2格式
    :return: 一个包含N帧图像的列表
    """
    frames = []
    for idx in range(parallel):
        if idx < parallel - 1:
            timestamp = onset + idx * (offset - onset) / (parallel - 1)
        else:
            frames.append(cv2.imread(format_filedir(path, offset, category, subject)))
            break
        # print(timestamp)
        prev_index = int(timestamp)
        prev_dir = format_filedir(path, prev_index, category, subject)
        next_index = prev_index + 1
        next_dir = format_filedir(path, next_index, category, subject)
        ratio = timestamp - prev_index
        frames.append(inference_img(model, [prev_dir, next_dir], ratio, device=device))

    if outdir is not None:
        if category == "CASME":
            out = f"{outdir}\\sub{subject.zfill(2)}\\{folder}"
        elif category == "MMEW":
            out = f"{outdir}\\sub{subject}\\{folder}"
        os.makedirs(out, exist_ok=True)
        for index, frame in enumerate(frames):
            cv2.imwrite(f"{out}/img{index}.jpg", frame)
    return frames


if __name__ == '__main__':
    RIFE = Model()
    RIFE.load_model("dataloader/weight", -1)
    RIFE.eval()
    RIFE.device()

    device = torch.device("cuda:0")

    csv_file = r"B:\0_0NewLife\0_Papers\FGRMER\4classes.csv"  # CASME 2
    image_root = r"B:\0_0NewLife\datasets\CASME_2\RAW_selected"  # raw_casme2
    # image_root = r"B:\0_0NewLife\0_Papers\FGRMER\CASME2\Cropped_EVM"  # cropped_evm
    # out_dir = f"B:\\0_0NewLife\\0_Papers\\FGRMER\\CASME2\\Interpolation\\Inter_offset_{parallel}"  # offset
    out_dir = f"B:\\0_0NewLife\\0_Papers\\FGRMER\\CASME2\\Interpolation\\Inter_{parallel}"  # apex

    # csv_file = r"B:\0_0NewLife\datasets\MMEW_Final\MMEW_Micro_Exp.csv"  # MMEW
    # image_root = r"B:\0_0NewLife\datasets\MMEW_Final\Micro_Expression"  # raw_casme2
    # out_dir = f"B:\\0_0NewLife\\0_Papers\\FGRMER\\MMEW\\Interpolation\\Inter_{parallel}"
    data = pd.read_csv(csv_file,
                       dtype={"Subject": str})
    for idx in range(data.shape[0]):
        print(f"Processing line {idx + 1} for Interpolation_{parallel}.")
        emotion = data.loc[idx, "Estimated Emotion"]
        folder = data.loc[idx, "Filename"]
        onset_name = data.loc[idx, "OnsetFrame"]
        # offset_name = data.loc[idx, "OffsetFrame"]  # offset
        offset_name = data.loc[idx, "ApexFrame"]  # apex
        subject = data.loc[idx, "Subject"]

        if catego == "SAMM":
            file_path = f"{image_root}/{subject}/{folder}"
        elif catego == "Cropped":
            file_path = f"{image_root}/sub{subject.zfill(2)}/{folder}"
        elif catego == "Cropped_EVM":
            file_path = f"{image_root}/sub{subject.zfill(2)}/{folder}"
        elif catego == "MMEW":
            file_path = f"{image_root}/{emotion}/{folder}"
        else:
            file_path = f"{image_root}/sub{subject.zfill(2)}/{folder}"

        frames = video_interpolation(model=RIFE,
                                     path=file_path,
                                     onset=onset_name,
                                     offset=offset_name,
                                     device=device,
                                     parallel=parallel,
                                     category=catego,
                                     subject=subject,
                                     outdir=out_dir,
                                     folder=folder)

