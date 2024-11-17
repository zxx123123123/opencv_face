import cv2 as cv

img = cv.imread("pessoas_seus_pets_Carly_Davidson_01.jpg")

cv.imshow('read_img',img)

# gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray_img = cv.cvtColor(img,cv.COLOR_RGB2BGR)

cv.imshow('gray',gray_img)
cv.imwrite('grat02.jpg',gray_img)

cv.waitKey(0)

cv.destroyAllWindows()