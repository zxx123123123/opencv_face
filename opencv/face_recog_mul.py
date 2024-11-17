import cv2 as cv


def facedetect(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_detect = cv.CascadeClassifier("haarcascades/haarcascade_frontalcatface.xml")

    face = face_detect.detectMultiScale(gray,1.1,8,0)
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x + w,y + h),color=(0,0,255),thickness=2)

    cv.imshow("result",img)


if __name__ == "__main__":
    img = cv.imread("Bombeiros-de-resgatam-seis-pessoas-e-um-cachorro-que-estavam-ilhados-em-Garuva-03-scaled.jpeg")

    facedetect(img)
    while True:
        if ord('q') == cv.waitKey(0):
            break

    cv.destroyAllWindows()

