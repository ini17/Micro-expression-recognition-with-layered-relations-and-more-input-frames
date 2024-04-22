import os
import re
import pandas as pd


patterns = re.compile(r"\d+")
csv_file = r"B:\0_0NewLife\datasets\SMIC\HS_cropped.csv"
df = pd.DataFrame(columns=["Subject", "Filename", "OnsetFrame", "OffsetFrame", "Estimated Emotion"])
num = 0

for root, dirs, files in os.walk(r"B:\0_0NewLife\datasets\SMIC\HS_cropped"):
    # root：当前目录
    # dirs：当前目录下包含的所有文件夹
    # files：当前目录下包含的所有文件
    if len(dirs) == 0:
        split_name = root.split("\\")
        if "micro" in split_name and split_name[-1] not in ["negative", "positive", "surprise"]:
            df.loc[num, "Subject"] = str(split_name[-4][1:])
            df.loc[num, "Filename"] = split_name[-1]
            df.loc[num, "Estimated Emotion"] = split_name[-2]
            onset = files[0].split(".")[0]
            offset = files[-1].split(".")[0]
            temp = str(re.findall(patterns, onset)[0])
            if temp[0] == "0":
                print(temp)
            df.loc[num, "OnsetFrame"] = temp
            df.loc[num, "OffsetFrame"] = str(re.findall(patterns, offset)[0])
            num += 1
            # print(str(split_name[-4][1:]).zfill(2))
df.to_csv(csv_file, index=False)


