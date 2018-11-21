# coding=utf-8


# 2018.01.07(일) 강연우 작성
# 정면, 측면 이미지 불 필요한 부분을 잘라내고
# 일정 비율로 줄여진 이미지를 반환하는 함수 작성

# 보완해야할 부분
# 배경 이미지 탐색시 조건문에 == 255 수정 해야함
# 현재 >150 으로 변경하면 컴파일 자체가 되지 않음

# 2017.12.30(토) 강연우 작성
# 이미지 전처리기
# 사진 가운데 인물이 있다는 전제하에 진행


import cv2
import copy
import numpy as np
from PIL import Image

# 정면 이미지에서 구한 좌표값 + 150픽셀을 저장할 변수

global top_y
global bottom_y
global middle_y
global left_x
global right_x

top_y = 0
bottom_y = 0
middle_y = 0 # 상위 y, 하위y, 중간y
left_x = 0
right_x = 0 # 좌축 x, 우측x




# 배경색 탐지 알고리즘
# 이미지와 찾을 점의 이름을 매개 변수로 넘긴다
# 배경의 시작점 객체를 반환한다
class background_point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def getBackgroundStartPoint(pcImage, flag):
    outImage = copy.copy(pcImage)

    rowNumber = outImage.shape[0]
    colNumber = outImage.shape[1]

    col_start = 0
    col_end = colNumber - 1
    row_start = 0
    row_end = rowNumber - 1

    global top_y
    global bottom_y
    global middle_y

    bgpoint = background_point(0, 0)

    if flag == "top":
        bgcheck = 0 # 배경 흰색 출현 체크
        for i in range(0,rowNumber):
            data_bg = outImage[i]
            for j in range(int(col_end / 2.0), int(col_end / 2.0)+1):
                if np.any(data_bg[j] == 255):  # 흰 배경에 닿으면

                    if bgcheck >= 100:
                        bgpoint.x = j
                        bgpoint.y = i
                        break

                    else: # 체크 횟수 증가
                        bgcheck += 1

            if bgpoint.x > 0:
                break

    elif flag == "bottom":
        bgcheck = 0  # 배경 흰색 출현 체크
        for k in range(rowNumber-1, 0, -1):
            data_bg = outImage[k]
            for l in range(int(col_end / 2.0), int(col_end / 2.0)+1):
                if np.any(data_bg[l] == 255):  # 흰 배경에 닿으면

                    if bgcheck >= 200:
                        bgpoint.x = l
                        bgpoint.y = k
                        break

                    else: # 체크 횟수 증가
                        bgcheck += 1

            if bgpoint.y > 0:
                break

    elif flag == "left":
        global middle_y
        global bottom_y
        global top_y

        middle_y = int( (bottom_y - top_y) / 2.0) + top_y
        bgcheck = 0  # 배경 흰색 출현 체크
        for i in range(middle_y, middle_y+1):
            for j in range(0, col_end):
                data_bg = outImage[j]
                if np.any(data_bg[i] == 255):  # 흰 배경에 닿으면

                    if bgcheck >= 100:
                        bgpoint.x = j
                        bgpoint.y = i
                        break

                    else:
                        bgcheck += 1

            if bgpoint.x > 0:
                break

    elif flag == "right":
        #global middle_y
        #global bottom_y
        #global top_y

        middle_y = int((bottom_y - top_y) / 2.0) + top_y
        bgcheck = 0  # 배경 흰색 출현 체크
        for k in range(middle_y, middle_y+1):
            for l in range(col_end, 0, -1):
                data_bg = outImage[l]
                if np.any(data_bg[k] == 255):  # 흰 배경이 아닌 점에 닿으면

                    if bgcheck >= 200:
                        bgpoint.x = l
                        bgpoint.y = k
                        break

                    else:
                        bgcheck += 1

            if bgpoint.x > 0:
                break

    else:
        print("FLAG INPUT ERROR")

    return bgpoint





# 머리끝, 발끝 탐색 알고리즘
# 머리 좌표
class front_head:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 발끝 좌표점
class front_foot:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 정면 키 좌표 (머리와 발끝 객체를 가진다.)
class frontheight:
    front_head=front_head(0,0)
    front_foot=front_foot(0,0)

    def setFrontHeight(self, front_head, front_foot):
        self.front_head=front_head
        self.front_foot=front_foot

# 정면 키의 좌표를 가져오는 메소드
def getFrontHeight(pcImage):

        outImage = copy.copy(pcImage)

        rowNumber = outImage.shape[0]
        colNumber = outImage.shape[1]
        col_start = 0
        col_end = colNumber - 1

        # 머리
        head = front_head(0, 0)
        # 탐색 시작점을 가져옴
        startPoint = getBackgroundStartPoint(pcImage, "top")
        print("-- startPoint(TOP) --")
        print(startPoint.x)
        print(startPoint.y)

        for i in range(startPoint.y, rowNumber):
            data_head = outImage[i]
            for j in range(startPoint.x, startPoint.x+1):
                if np.any(data_head[j] < 150):  # 흰 배경이 아닌 곳에 닿으면
                    head.x = j
                    head.y = i
                    break

            if head.x > 0:
                break

        # 발
        foot = front_foot(0, 0)
        # 탐색 시작점을 가져옴
        startPoint = getBackgroundStartPoint(pcImage, "bottom")
        print("-- startPoint(BOTTOM) --")
        print(startPoint.x)
        print(startPoint.y)

        for k in range(startPoint.y -1, 0, -1):
            data_foot = outImage[k]
            for l in range(col_start, col_end):
                if np.any(data_foot[l] < 150):  # 흰 배경이 아닌 곳에 닿으면
                    foot.x = l
                    foot.y = k
                    break

            if foot.y > 0:
                break

        # 머리 좌표 및 발 좌표 가져오기
        print("--front head--")
        print(head.x)
        print(head.y)

        print("--front foot--")
        print(foot.x)
        print(foot.y)

        my_front_height=frontheight()
        my_front_height.setFrontHeight(head,foot)

        return my_front_height




# 왼손 좌표
class left_hand:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 오른손 좌표점
class right_hand:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 손끝의 지문 좌표 (왼손과 오른손 객체를 가진다.)
class hand:
    left_hand=left_hand(0,0)
    right_hand=right_hand(0,0)

    def setHand(self,left,right):
        self.right_hand=right
        self.left_hand=left

# 왼손, 오른손 탐색 알고리즘
# 손의 좌표를 가져오는 메소드
def getHand(pcImage):

        outImage = copy.copy(pcImage)  # 객체 복사하기
        rowNumber = pcImage.shape[0]
        colNumber = pcImage.shape[1]

        # 튜플에 해당 값을 삽입
        row_start = 0
        row_end = rowNumber - 1

        # 왼쪽손
        left = left_hand(0, 0)
        # 탐색 시작점을 가져옴
        startPoint = getBackgroundStartPoint(pcImage, "left")
        print("-- startPoint(LEFT) --")
        print(startPoint.x)
        print(startPoint.y)
        for i in range(startPoint.x, colNumber-1):
            for j in range(startPoint.y, startPoint.y+1):
                data_left = outImage[j]
                if np.any(data_left[i] < 150): #흰 배경이 아닌 점에 닿으면
                    left.x = i
                    left.y = j
                    break

            if left.x > 0:
                 break


        # 오른손
        right = right_hand(0, 0)
        # 탐색 시작점을 가져옴
        startPoint = getBackgroundStartPoint(pcImage, "right")
        print("-- startPoint(RIGHT) --")
        print(startPoint.x)
        print(startPoint.y)
        for k in range(startPoint.x, 0, -1):
            for l in range(startPoint.y, startPoint.y+1):
                data_right = outImage[l]
                if np.any(data_right[k] < 150):  # 흰 배경이 아닌 점에 닿으면
                    right.x = k
                    right.y = l
                    break

            if right.x > 0:
                break

        # 왼손 좌표 및 오른손 좌표 가져오기
        print("--left hand--")
        print(left.x)
        print(left.y)

        print("--right hand--")
        print(right.x)
        print(right.y)

        my_hand=hand()
        my_hand.setHand(left, right)

        return my_hand

# 정면 이미지 함수
# 정면 이미지 경로를 매개변수로 받아 처리된 정면 이미지를 반환하는 함수
def getFrontImg(targetimage):

    img = cv2.imread(targetimage, 1) # 원본 이미지 저장
    print("-- origin image size --")
    print(img.shape[1])
    print(img.shape[0])

    # 흑백 처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, 0)

    # 머리끝, 발끝 점
    front_height_points = getFrontHeight(gray)
    cv2.circle(gray, (front_height_points.front_head.x, front_height_points.front_head.y), 10, (0, 0, 255), -1)  # 머리점 찍기
    cv2.circle(gray, (front_height_points.front_head.x, front_height_points.front_foot.y), 10, (0, 0, 255), -1)  # 발끝점 찍기 - 발끝의 x는 머리의 x로 둠

    # 상단 자를 부분
    head_p = front_height_points.front_head.y
    global top_y
    top_y = head_p - 150

    # 하단 자를 부분
    foot_p =front_height_points.front_foot.y
    global bottom_y
    bottom_y = foot_p + 150

    global middle_y
    middle_y = int((bottom_y - top_y) / 2.0)

    # 왼손, 오른손 점
    front_hand_points = getHand(gray)
    cv2.circle(gray, (front_hand_points.left_hand.x, front_hand_points.left_hand.y), 10, (0, 0, 255), -1)  # 왼손 찍기
    cv2.circle(gray, (front_hand_points.right_hand.x, front_hand_points.right_hand.y), 10, (0, 0, 255), -1)  # 오른손 찍기

    # 좌측 자를 부분
    left_p = front_hand_points.left_hand.x
    global left_x
    left_x = left_p - 150

    # 우측 자를 부분
    right_p = front_hand_points.right_hand.x
    global right_x
    right_x = right_p + 150

    # 자르기
    imgCrop = gray[top_y:bottom_y, left_x:right_x]

    # 이미지 축소 ( 600 * ? )
    ratio = 300.0 / imgCrop.shape[1]
    ddim = (300, int(imgCrop.shape[0] * ratio))
    graysize = cv2.resize(imgCrop, ddim, interpolation=cv2.INTER_AREA)
    grayresize = cv2.bilateralFilter(graysize, 9, 41, 41)

    print("-- resized --")
    print(grayresize.shape[1])
    print(grayresize.shape[0])

    return grayresize # 처리된 정면 사진 반환

# 측면 이미지 함수
# 정면 이미지 처리시 구해진 좌표를 이용해 측면 이미지를 처리하여 반환하는 함수
def getSideImg(targetimage):
    img = cv2.imread(targetimage, 1)  # 원본 이미지 저장
    print("-- origin image size --")
    print(img.shape[1])
    print(img.shape[0])

    # 흑백 처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, 0)

    # 자르기
    imgCrop = gray[top_y:bottom_y, left_x:right_x]

    # 이미지 축소 ( 600 * ? )
    ratio = 300.0 / imgCrop.shape[1]
    ddim = (300, int(imgCrop.shape[0] * ratio))
    graysize = cv2.resize(imgCrop, ddim, interpolation=cv2.INTER_AREA)
    grayresize = cv2.bilateralFilter(graysize, 9, 41, 41)

    print("-- resized --")
    print(grayresize.shape[1])
    print(grayresize.shape[0])

    return grayresize  # 처리된 측면 사진 반환



# 정면사진 함수 테스트
image = getFrontImg("IMG_1932.jpg")
#image = getFrontImg("student1_front.bmp")
#image = getFrontImg("IMG_1943.jpg")
#image = getFrontImg("IMG_1932.jpg")
#image = getFrontImg("kang_front_origin.jpg")

# 전역 변수 사용 확인
print("-- GLOBAL --")
print("top_y: ", top_y)
print("bottom_y: ", bottom_y)
print("middle_y: ", middle_y)
print("left_x: ", left_x)
print("right_x: ", right_x)



# 측면사진 함수 테스트
image2 = getSideImg("IMG_1947.jpg")

cv2.imshow("Cropped Image Test", image)
cv2.imshow("Cropped Image Test_Side", image2)
cv2.waitKey(0)