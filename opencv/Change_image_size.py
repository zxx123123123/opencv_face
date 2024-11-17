import cv2 as cv

img = cv.imread("pessoas_seus_pets_Carly_Davidson_01.jpg")

cv.imshow('read_img',img)

resize_img = cv.resize(img,dsize=(200,200))
cv.imshow("resize",resize_img)
print(img.shape)
print(resize_img.shape)

while True:
    if ord('q') == cv.waitKey(0):
        break

cv.destroyAllWindows()