# -*- coding:UTF-8 -*-
import cv2
import numpy as np

def img_trans(img):
    # gamma变换
    gamma, gain, scale = 1.0, 1, 255
    # gamma, gain, scale = 0.7, 1, 255
    gamma_img = np.zeros_like(img)
    for i in range(3):
        gamma_img[:, :, i] = ((img[:, :, i] / scale) ** gamma) * scale * gain
    return gamma_img



class Beauty():
    def __init__(self, pic):
        self.pic_path = pic

    def Skinning_Wrinkle_Removing(self,img):
        step = 5
        kernel = (32, 32)
        # img = cv2.imread(self.pic_path)
        img = img / 255.0
        sz = img.shape[:2]
        sz1 = (int(round(sz[1] * step)), int(round(sz[0] * step)))
        sz2 = (int(round(kernel[0] * step)), int(round(kernel[0] * step)))
        sI = cv2.resize(img, sz1, interpolation=cv2.INTER_LINEAR)
        sp = cv2.resize(img, sz1, interpolation=cv2.INTER_LINEAR)
        msI = cv2.blur(sI, sz2)
        msp = cv2.blur(sp, sz2)
        msII = cv2.blur(sI * sI, sz2)
        msIp = cv2.blur(sI * sp, sz2)
        vsI = msII - msI * msI
        csIp = msIp - msI * msp
        recA = csIp / (vsI + 0.01)
        recB = msp - recA * msI
        mA = cv2.resize(recA, (sz[1], sz[0]), interpolation=cv2.INTER_LINEAR)
        mB = cv2.resize(recB, (sz[1], sz[0]), interpolation=cv2.INTER_LINEAR)
        gf = mA * img + mB
        gf = gf * 255
        gf[gf > 255] = 255
        final = gf.astype(np.uint8)
        return final

    def main(self):
        image_before = cv2.imread(self.pic_path)
        final = self.Skinning_Wrinkle_Removing(image_before)
        # final = img_trans(final)
        cv2.imwrite("/media/mjy/25346075-c1e8-41fb-af83-28b0448f7cf1/dilili/result/Beauty.png",
                    final)




# pic = "/media/mjy/25346075-c1e8-41fb-af83-28b0448f7cf1/dilili/pic/shen.png"
# pic = "/media/mjy/25346075-c1e8-41fb-af83-28b0448f7cf1/dilili/pic/Freckle.jpg"
# Beauty = Beauty(pic)
# n = input("Please input: ")
# Beauty.main(n)
# Beauty.FreckleRemoving()
# Beauty.Skinning_Wrinkle_Removing()



