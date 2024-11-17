import cv2 as cv

img = cv.imread("pessoas_seus_pets_Carly_Davidson_01.jpg")

cv.imshow('read_img',img)

cv.waitKey(0)

cv.destroyAllWindows()