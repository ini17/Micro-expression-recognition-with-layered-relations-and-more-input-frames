import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")


def inference_img(model, img_list, ratio, device, exp=1, rthreshold=0.01, rmaxcycles=8):
    """
    这里使用RIFE方法进行视频插值，返回插值后的图像
    :param model: .to(device)后的RIVE_HDv3模型
    :param img_list: 两帧图像，用于在这两帧之间进行插值
    :param ratio: 以第一张图像为0，第二张图像为1，给出一个想要的时刻值
    :param device: torch.device
    :param exp: 不太懂
    :param rthreshold: 当ratio落在0+rthreshold和1-rthreshold之间时，直接返回0或1
    :param rmaxcycles: 最大对开循环数
    :return: 返回插值后的图像
    """

    img0 = cv2.imread(img_list[0], cv2.IMREAD_UNCHANGED)
    img1 = cv2.imread(img_list[1], cv2.IMREAD_UNCHANGED)
    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    img0_ratio = 0.0
    img1_ratio = 1.0
    if ratio <= img0_ratio + rthreshold / 2:
        middle = img0
    elif ratio >= img1_ratio - rthreshold / 2:
        middle = img1
    else:
        tmp_img0 = img0
        tmp_img1 = img1
        for inference_cycle in range(rmaxcycles):
            middle = model.inference(tmp_img0, tmp_img1)
            middle_ratio = (img0_ratio + img1_ratio) / 2
            if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                break
            if ratio > middle_ratio:
                tmp_img0 = middle
                img0_ratio = middle_ratio
            else:
                tmp_img1 = middle
                img1_ratio = middle_ratio
    return (middle[0]*255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    parser.add_argument('--img', dest='img',
                        nargs=2,
                        # required=True)
                        default=["test/reg_img46.jpg", "test/reg_img86.jpg"])
    parser.add_argument('--exp', default=1, type=int)
    parser.add_argument('--ratio', default=0.5, type=float, help='inference ratio between two images with 0 - 1 range')
    parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')
    parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
    parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')

    args = parser.parse_args()
