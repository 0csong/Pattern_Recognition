import cv2
import numpy as np
def pipeline(img):
# img= cv2.imread('./test_images/solidWhiteRight.jpg')

    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    blurred_img=cv2.GaussianBlur(gray_img,(15,15),0.0)

    edge_img=cv2.Canny(blurred_img,70,140)

    return edge_img

#비디오 읽는법
cap=cv2.VideoCapture('./test_videos/solidWhiteRight.mp4')
while True:
    ok,frame=cap.read()
    if not ok:
        break

    edge_img=pipeline(frame)
    cv2.imshow('edge',edge_img)
    key=cv2.waitKey(30)#30ms동안 어떤키를 입력받으면 정수값 아니면 -1
    if key==ord('x'):
        break

cap.release()


