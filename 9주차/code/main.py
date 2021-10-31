import cv2
import numpy as np
import time
#sift와 surf는 opencv 3.4x버전에서만 실행이 가능하다 버전을 낮춰야한다
#가상환경 설치하고 pip install opencv-contrib-python==3.4.2.16설치
#설치시 오류나는 경우가 있다. 아나콘다 프롬포트로하니 안되고 파이참 콘솔로하니 된다 경로설정문제?

filepath = 'box.png'
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