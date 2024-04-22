import argparse
import os
import cv2
import pandas as pd


parallel = 10


def RunOpenFace(parse: argparse):
    data = pd.read_csv(parse.csv_path,
                       dtype={"Subject": str,
                              "Filename": str,
                              "OnsetFrame": int,
                              "OffsetFrame": int})
    nums = 0
    for idx in range(data.shape[0]):
        subject = data.loc[idx, 'Subject']
        folder = data.loc[idx, 'Filename']
        cmd = ""

        os.makedirs(f"{args.openface_out_dir}/sub{subject.zfill(2)}", exist_ok=True)

        if parse.catego == "SAMM":
            # onset_path = f"{parse.image_root}/{subject}/{folder}/{subject}_{onset_name:05}.jpg"
            # offset_path = f"{parse.image_root}/{subject}/{folder}/{subject}_{offset_name:05}.jpg"
            # 未完待续，仿照上述path和下面的images读取方式来写
            pass
        else:
            cmd = f"{args.openface} -fdir {parse.save_path}/sub{subject.zfill(2)}/{folder}" \
                  f" -aus -out_dir {args.openface_out_dir}/sub{subject.zfill(2)}"
        os.system(cmd)
        nums += 1
        if nums % 5 == 0:
            print(f"\n\n\nAlready process {nums} videos.\n\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path",
                        type=str,
                        # required=True,
                        default=r"B:\0_0NewLife\CASME_2\4classes.csv",
                        help="Path for the csv file for training data")
    parser.add_argument("--save_path",
                        type=str,
                        # required=True,
                        default=r"B:\0_0NewLife\0_Papers\FGRMER\CASME2\Interpolation\RAW_selected_EVM_Inter_10",
                        help="Path for the augmented images")
    parser.add_argument("--image_root",
                        type=str,
                        # required=True,
                        default=r"B:\0_0NewLife\CASME_2\RAW_selected",
                        help="Root for the training images")
    parser.add_argument("--catego",
                        type=str,
                        # required=True,
                        default="CASME",
                        help="SAMM or CASME dataset")
    parser.add_argument("--openface",
                        type=str,
                        default="\"B:\\0_0NewLife\\Transition Period\\2022.09.22\\"
                                "OpenFace_2.2.0_win_x64\\FeatureExtraction.exe\"",
                        help="Path for OpenFace software")
    parser.add_argument("--openface_out_dir",
                        type=str,
                        default=f"B:\\0_0NewLife\\0_Papers\\FGRMER\\CASME2\\"
                                f"OpenFace_Output\\RAW_selected_EVM_Inter_{parallel}",
                        help="Path for the OpenFace outputs")
    args = parser.parse_args()
    # aug_images(args)
    RunOpenFace(args)


