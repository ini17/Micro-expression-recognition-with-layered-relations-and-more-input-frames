import dlib
import numpy as np
import cv2
import os

# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器
predictor = dlib.shape_predictor(r'B:\0_0NewLife\0_Papers\FGRMER\weight/shape_predictor_68_face_landmarks.dat')

# 读取图像的路径
path_read = r"B:\0_0NewLife\0_Papers\FGRMER\SMIC\Interpolation\raw_Inter_offset_13"

# 用来存储生成的单张人脸的路径
path_save = r"B:\0_0NewLife\0_Papers\FGRMER\SMIC\Interpolation\Inter_offset_13"


def main():
    for root, dirs, files in os.walk(path_read):
        # if len(dirs) == 0 and "s4_sur_05" in root:
        if len(dirs) == 0:
            for file in files:
                img = cv2.imread(os.path.join(path_read, root, file))
                faces = detector(img, 1)
                path = os.path.join(path_save, "\\".join(root.split("\\")[-2:]))
                print(path)
                for num, face in enumerate(faces):
                    t, b, l, r = face.top(), face.bottom(), face.left(), face.right()
                    img_blank = img[t - 10: b + 10, l-10: r+10]
                    # 存在本地
                    os.makedirs(path, exist_ok=True)
                    # print(os.path.join(path_save, "\\".join(root.split("\\")[-2:]), file))
                    cv2.imwrite(os.path.join(path, file), img_blank)


if __name__ == '__main__':
    main()