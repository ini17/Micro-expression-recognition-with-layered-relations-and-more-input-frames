import os
import cv2
import torch
import pandas as pd
from dataloader.RIFE_HDv3 import Model
from dataloader.RIFE_for_FGRMER import inference_img


parallel = 10  # N+1，即包括onset帧在内
AMP_LIST = [x/10 for x in range(12, 31, 2)]


def format_filedir(path, index):
    return f"{path}/reg_image{index}.bmp"


def file_exists(onset_str, path, timestamp):
    prev = int(timestamp)
    nex = prev + 1
    forward, backward = 0, 0
    front, back = 0, 0
    for i in range(4):
        prev = prev - i
        forward = format_filedir(path, prev if len(onset_str) == len(str(prev)) else "0" + str(prev))
        if os.path.exists(forward):
            front = prev
            break
    for i in range(4):
        nex = nex + i
        backward = format_filedir(path, nex if len(onset_str) == len(str(nex)) else "0" + str(nex))
        if os.path.exists(backward):
            back = nex
            break
    ratio = (timestamp - front) / (back - front)
    return [forward, backward, ratio]


def video_interpolation(model, path, onset_str, offset_str, device,
                        parallel, subject=None, outdir=None, folder=None):
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
    onset, offset = int(onset_str), int(offset_str)
    for idx in range(parallel):
        if idx < parallel - 1:
            timestamp = onset + idx * (offset - onset) / (parallel - 1)
        else:
            frames.append(cv2.imread(format_filedir(path, offset_str)))
            break
        # print(timestamp)
        # prev_index = int(timestamp)
        # ratio = timestamp - prev_index
        [prev_dir, next_dir, ratio] = file_exists(onset_str, path, timestamp)
        frames.append(inference_img(model, [prev_dir, next_dir], ratio, device=device))

    if outdir is not None:
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

    csv_file = r"B:\0_0NewLife\datasets\SMIC\HS_cropped.csv"
    image_root = r"B:\0_0NewLife\datasets\SMIC\HS_cropped"  # SMIC
    out_dir = f"B:\\0_0NewLife\\0_Papers\\SMC\\SMIC\\Interpolation\\Inter_offset_{parallel}"
    data = pd.read_csv(csv_file,
                       dtype={
                           "Subject": str,
                           "OnsetFrame": str,
                           "OffsetFrame": str
                       })

    for idx in range(data.shape[0]):
        print(f"Processing line {idx + 1} for Interpolation_{parallel}.")
        emotion = data.loc[idx, "Estimated Emotion"]
        folder = data.loc[idx, "Filename"]
        onset_name = data.loc[idx, "OnsetFrame"]
        offset_name = data.loc[idx, "OffsetFrame"]
        subject = data.loc[idx, "Subject"]

        file_path = f"{image_root}/s{subject}/micro/{emotion}/{folder}"

        frames = video_interpolation(model=RIFE,
                                     path=file_path,
                                     onset_str=onset_name,
                                     offset_str=offset_name,
                                     device=device,
                                     parallel=parallel,
                                     subject=subject,
                                     outdir=out_dir,
                                     folder=folder)

