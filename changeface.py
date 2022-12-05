# -*- coding:UTF-8 -*-
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy
import dlib
import sys
import cv2

class ChangeFace():
    def __init__(self,pic1,pic2):
        self.pic1_path = pic1
        self.pic2_path = pic2
        sys.argv = ["main.py",pic1,pic2]
        # sys.argv = ["main.py","/media/mjy/25346075-c1e8-41fb-af83-28b0448f7cf1/dilili/pic/snap.jpeg","/media/mjy/25346075-c1e8-41fb-af83-28b0448f7cf1/dilili/pic/shen.jpeg"]
        PREDICTOR_PATH = "/media/mjy/25346075-c1e8-41fb-af83-28b0448f7cf1/dilili/shape_predictor_68_face_landmarks.dat"

        # 对人脸的五官进行点的定位（根据68点）
        self.FACE_POINTS = list(range(17, 68))
        self.MOUTH_POINTS = list(range(48, 61))
        self.RIGHT_BROW_POINTS = list(range(17, 22))
        self.LEFT_BROW_POINTS = list(range(22, 27))
        self.RIGHT_EYE_POINTS = list(range(36, 42))
        self.LEFT_EYE_POINTS = list(range(42, 48))
        self.NOSE_POINTS = list(range(27, 35))
        self.JAW_POINTS = list(range(0, 17))
        self.SCALE_FACTOR = 1
        self.FEATHER_AMOUNT = 11

        # 对提取到的人脸部位进行排列
        self.ALIGN_POINTS = (self.LEFT_BROW_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_EYE_POINTS +
                        self.RIGHT_BROW_POINTS + self.NOSE_POINTS + self.MOUTH_POINTS)

        self.OVERLAY_POINTS = [
            self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_BROW_POINTS + self.RIGHT_BROW_POINTS,
            self.NOSE_POINTS + self.MOUTH_POINTS,
        ]

        # 颜色校正期间使用的模糊量, as a fraction of the
        # pupillary distance.
        self.COLOUR_CORRECT_BLUR_FRAC = 0.6

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)

    # 定义函数get_face_mask()为一张图像和一个标记矩阵生成一个遮罩，它画出了两个白色的凸多边形：一个是眼睛周围的区域，一个是鼻子和嘴部周围的区域。这样一个遮罩同时为这两个图像生成，使用与步骤2中相同的转换，可以使图像2的遮罩转化为图像1的坐标空间。之后，通过一个element-wise最大值，这两个遮罩结合成一个。结合这两个遮罩是为了确保图像1被掩盖，而显现出图像2的特性。
    # 第一段为如何提取并遮罩，第二段代码为遮罩覆盖以实现颜色校正。
    def get_landmarks(self,im):
        rects = self.detector(im, 1)

        if len(rects) > 1:
            raise print('TooManyFaces')
        if len(rects) == 0:
            raise print('NoFaces')

        return numpy.matrix([[p.x, p.y] for p in self.predictor(im, rects[0]).parts()])


    def annotate_landmarks(self, im, landmarks):
        im = im.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
        return im


    def draw_convex_hull(self,im, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)


    def get_face_mask(self, im, landmarks):
        im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

        for group in self.OVERLAY_POINTS:
            self.draw_convex_hull(im,
                             landmarks[group],
                             color=1)

        im = numpy.array([im, im, im]).transpose((1, 2, 0))

        im = (cv2.GaussianBlur(im, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0) > 0) * 1.0
        im = cv2.GaussianBlur(im, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0)

        return im


    def correct_colours(self, im1, im2, landmarks1):
        blur_amount = self.COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
            numpy.mean(landmarks1[self.LEFT_EYE_POINTS], axis=0) -
            numpy.mean(landmarks1[self.RIGHT_EYE_POINTS], axis=0))
        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

        # Avoid divide-by-zero errors.
        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

        return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                im2_blur.astype(numpy.float64))

    # 缩放、旋转以完成完全覆盖
    # 1.将输入矩阵转换为浮点数
    # 2.每一个点集减去它的矩心
    # 3.每一个点集除以它的标准偏差这会消除组件缩放偏差的问题
    # 4.使用奇异值分解计算旋转部分
    # 5.利用仿射变换矩阵返回完整的转化
    def transformation_from_points(self, points1, points2):
        points1 = points1.astype(numpy.float64)
        points2 = points2.astype(numpy.float64)
        c1 = numpy.mean(points1, axis=0)
        c2 = numpy.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2
        s1 = numpy.std(points1)
        s2 = numpy.std(points2)
        points1 /= s1
        points2 /= s2
        U, S, Vt = numpy.linalg.svd(points1.T * points2)
        # The R we seek is in fact the transpose of the one given by U * Vt. This
        # is because the above formulation assumes the matrix goes on the right
        # (with row vectors) where as our solution requires the matrix to be on the
        # left (with column vectors).
        R = (U * Vt).T
        return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                           c2.T - (s2 / s1) * R * c1.T)),
                             numpy.matrix([0., 0., 1.])])

    # 读取并实现脸部代换
    # 首先通过CV2实现提取之前68点找到的位置，辅以校正颜色后的read之后进行定点
    def read_im_and_landmarks(self, fname):
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        im = cv2.resize(im, (im.shape[1] * self.SCALE_FACTOR,
                             im.shape[0] * self.SCALE_FACTOR))
        s = self.get_landmarks(im)

        return im, s

    def warp_im(self, im, M, dshape):
        output_im = numpy.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(im,
                       M[:2],
                       (dshape[1], dshape[0]),
                       dst=output_im,
                       borderMode=cv2.BORDER_TRANSPARENT,
                       flags=cv2.WARP_INVERSE_MAP)
        return output_im

    # 应用遮罩进行覆盖操作
    def main(self):
        print("Whether beauty is needed? y/n")
        choose = input()
        if choose=='y':
            from Beauty import Beauty
            pic2 = self.pic2_path
            Beauty2 = Beauty(pic2)
            Beauty2.main()
            sys.argv[2] = "/media/mjy/25346075-c1e8-41fb-af83-28b0448f7cf1/dilili/result/Beauty.png"
            # #
            # pic1 = self.pic1_path
            # Beauty1 = Beauty(pic1)
            # Beauty1.main()
            # sys.argv[1] = "/media/mjy/25346075-c1e8-41fb-af83-28b0448f7cf1/dilili/result/Beauty.png"
            # print('Bueaty success!')


        im1, landmarks1 = self.read_im_and_landmarks(sys.argv[1])
        im2, landmarks2 = self.read_im_and_landmarks(sys.argv[2])

        M = self.transformation_from_points(landmarks1[self.ALIGN_POINTS],
                                       landmarks2[self.ALIGN_POINTS])

        mask = self.get_face_mask(im2, landmarks2)
        warped_mask = self.warp_im(mask, M, im1.shape)
        combined_mask = numpy.max([self.get_face_mask(im1, landmarks1), warped_mask],
                                  axis=0)
        warped_im2 = self.warp_im(im2, M, im1.shape)
        warped_corrected_im2 = self.correct_colours(im1, warped_im2, landmarks1)

        output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

        # 最后用CV2输出换脸jpg
        cv2.imwrite('/media/mjy/25346075-c1e8-41fb-af83-28b0448f7cf1/dilili/result/change-face.jpg', output_im)


# pic1 = "/media/mjy/25346075-c1e8-41fb-af83-28b0448f7cf1/dilili/pic/snap.jpeg"
# pic2 = "/media/mjy/25346075-c1e8-41fb-af83-28b0448f7cf1/dilili/pic/shen.jpeg"
# ChangeFace = ChangeFace(pic1,pic2)
# ChangeFace.main()