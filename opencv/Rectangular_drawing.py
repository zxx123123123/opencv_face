import cv2 as cv

img = cv.imread("pessoas_seus_pets_Carly_Davidson_01.jpg")

cv.imshow('read_img',img)

x,y,w,h = 100,100,100,100
cv.rectangle(img,(x,y,x+w,y+h),color=(0,0,255),thickness=1)

cv.circle(img,center=(x+w,y+h),radius=100,color=(255,0,0),thickness=5)

cv.imshow('img',img)

while True:
    if ord('q') == cv.waitKey(0):
        break

cv.destroyAllWindows()