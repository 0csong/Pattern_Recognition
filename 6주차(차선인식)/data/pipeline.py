import numpy as np
import cv2


def set_region_of_interest(img, vertices):
    """

    :param img:       대상 이미지
    :param vertices:  이미지에서 남기고자 하는 영역의 꼭짓점 좌표 리스트
    :return:
    관심 영역만 마스킹 된 이미지
    """

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)

    return cv2.bitwise_and(img, mask)


def run(img):
    height, width = img.shape[:2]

    vertices = np.array([[(50, height),
                          (width // 2 - 45, height // 2 + 60),
                          (width // 2 + 45, height // 2 + 60),
                          (width - 50, height)]])

    # 1) BGR -> GRAY 영상으로 색 변환
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) 이미지 내 노이즈를 완화시키기 위해 blur 효과 적용
    blur_img = cv2.GaussianBlur(gray_img, (7, 7), 0)

    # 3) 캐니 엣지 검출을 사용하여 엣지 영상 검출
    edge_img = cv2.Canny(blur_img, 150, 270)

    # 4) 관심 영역(ROI; Region Of Interest)을 설정하여 배경 영역 제외
    roi_img = set_region_of_interest(edge_img, vertices)

    # 5) 허프 변환을 사용하여 조건을 만족하는 직선 검출
    lines = cv2.HoughLinesP(roi_img, 0.7, np.pi / 180, 30, minLineLength=10, maxLineGap=150)
    #lines = cv2.HoughLines(img, rho, theta, threshold, lines, srn=0, stn=0, min_theta, max_theta)->2차원 (r,theta)배열
    #lines = cv2.HoughLinesP(img, rho, theta, threshold, lines, minLineLength, maxLineGap)→ lines(x1,y1),(x2,y2)

    # 6) 찾은 직선들을 입력 이미지에 그리기
    result=img
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(result, (x1,y1), (x2, y2), (0,0,255), 3)

    return result
