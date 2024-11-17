import cv2 as cv



def facedetect(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_detect = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")

    face = face_detect.detectMultiScale(gray,1.1,8,0,(100,100),(300,300))
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x + w,y + h),color=(0,0,255),thickness=2)

    cv.imshow("result",img)


if __name__ == "__main__":
    img = cv.imread("pessoas_seus_pets_Carly_Davidson_01.jpg")

    facedetect(img)
    while True:
        if ord('q') == cv.waitKey(0):
            break

    cv.destroyAllWindows()

