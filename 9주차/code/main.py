import cv2
import numpy as np
import time
#sift와 surf는 opencv 3.4x버전에서만 실행이 가능하다 버전을 낮춰야한다

filepath = 'butterfly.png'
img = cv2.imread(filepath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

for i in range(3):
    x, y = input('img size ').split()
    x = int(x)
    y = int(y)
    resize_img = cv2.resize(img, (x, y))


    sift = cv2.xfeatures2d.SIFT_create()  # SIFT 검출기 생성
    #시간
    start = time.time()
    kpts = sift.detect(image=resize_img, mask=None)  # SIFT keypoints 검출
    print("sift :", time.time() - start, "sec")

    surf = cv2.xfeatures2d.SURF_create()
    #시간
    start = time.time()
    kpts_1 = surf.detect(image=resize_img, mask=None)  # SURF keypoints 검출
    print("surf :", time.time() - start, "sec")
    print("\n")