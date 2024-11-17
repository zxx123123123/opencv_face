import os
import cv2
from PIL import Image
import numpy as np

def getimageAndlabels(path):
    facesSamples = []
    ids = []
    foldpaths = [os.path.join(path,f) for f in os.listdir(path)]
    # imagepaths = [os.path.join(foldpaths,f) for f in os.listdir(path)]

    face_detecter = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
    id = 0
    for foldpath in foldpaths:
        imagepaths = [os.path.join(foldpath, f) for f in os.listdir(foldpath)]
        # id = int(os.path.split(imagepath)[1].split('.')[0])

        for imagepath in imagepaths:
            PIL_img = Image.open(imagepath).convert('L')
            img_numpy = np.array(PIL_img,'uint8')
            faces = face_detecter.detectMultiScale(img_numpy)
            # id = int(os.path.split(imagepath)[1].split('.')[0])
            for x,y,w,h in faces:
                ids.append(id)
                facesSamples.append(img_numpy[y:y + h,x:x+w])
        print('id: ',id)
        print("fs: ",facesSamples)
        id += 1
    return facesSamples,ids

if __name__ == "__main__":
    #存放图片的路径
    path = "data"
    faces,ids = getimageAndlabels(path)

    recog = cv2.face.LBPHFaceRecognizer_create()
    recog.train(faces,np.array(ids))

    recog.write("trainer/trainer.yml")