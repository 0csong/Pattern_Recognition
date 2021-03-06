import cv2
import numpy as np

# 모델영상의 2차원 H-S 히스토그램 계산
img_m = cv2.imread('model.jpg')
hsv_m = cv2.cvtColor(img_m, cv2.COLOR_BGR2HSV)
hist_m = cv2.calcHist([hsv_m], [0, 1], None, [180, 256], [0, 180, 0, 256])  # [0, 1]는 각각 H,S

# 입력영상의 2차원 H-S 히스토그램 계산
img_i = cv2.imread('hand.jpg')
hsv_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2HSV)
hist_i = cv2.calcHist([hsv_i], [0, 1], None, [180, 256], [0, 180, 0, 256])

# 히스토그램 정규화
# 각 이미지 정규화
# cv2.normalize(hist_m, hist_m, 0, 1, cv2.NORM_MINMAX)
# cv2.normalize(hist_i, hist_i, 0, 1, cv2.NORM_MINMAX)
hist_m = hist_m / (img_m.shape[0] * img_m.shape[1])
hist_i = hist_i / img_i.size

print("maximum of hist_m : %.1f" % hist_m.max())
print("maximum of hist_i : %.1f" % hist_i.max())

# 비율 히스토그램 계산
# hist_r = np.minimum(hist_m / (hist_i + 1e-7), 1)  # (hist_i + 1e-7)나누는 분모가 0이되는것을 막기위해 더함
hist_r = hist_m / (hist_i + 1e-7)
hist_r = np.minimum(hist_r, 1.0)
print("range of hist_r : [%.1f, %.1f]" % (hist_r.min(), hist_r.max()))

# 히스토그램 역투영 수행
height, width = img_i.shape[0], img_i.shape[1]
result = np.zeros_like(img_i, dtype='float32')
h, s, v = cv2.split(hsv_i)

for i in range(height):
    for j in range(width):
        h_value = h[i, j]
        s_value = s[i, j]
        confidence = hist_r[h_value, s_value]# hist_r은 2차원 배열임으로 그 특정값은 신뢰도
        result[i, j] = confidence

# result = cv2.calcBackProject([hsv_i], [0, 1], hist_r, [0, 180, 0, 256], 1)

# 이진화수행(최소값이 임계값 0.02보다 크면 255, 아니면 0)
ret, thresholded = cv2.threshold(result, 0.02, 255, cv2.THRESH_BINARY)
cv2.imwrite('result.jpg', thresholded)

# 모폴로지 연산적용
img = cv2.imread('result.jpg')
kernel = np.ones((13, 13), np.uint8)  # 잡음 제거를 위해
improved = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('morpology.jpg', improved)
