import inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import cv2

# img = cv2.imread("C:/Users/user1/Desktop/Al pacino.jpg")
# while True:
#     cv2.imshow('Al-pacino,s picture', img)
#
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# cv2.destroyAllWindows()

# image_blank = np.zeros(shape=(600, 600, 3), dtype=np.int16)
#
# line_red = cv2.line(image_blank,(100,0),(100,600),(200,200,0), 20)
# plt.imshow(line_red)
#
# rectangle = cv2.rectangle(image_blank,(100,20),(450,250),(0,255,0), -5)
#
# circle = cv2.circle(image_blank,(255,130), 75, (255,0,0), -1)
#
# rectangle2 = cv2.rectangle(image_blank,(100,280),(410,500),(255,255,255), -5)
#
# circle2 = cv2.circle(image_blank,(255,385), 72, (255,0,0), -1)
#
# font = cv2.FONT_HERSHEY_SIMPLEX
# text = cv2.putText(image_blank,'Bangladesh vs Japan',(150,560), font, 1,(255,0,0),2,cv2.LINE_AA)
# plt.imshow(text)
#
#
# plt.imshow(image_blank)
# plt.show()


#C:/Users/user1/Desktop/Al pacino4.jpg     , ,   E:/New folder/baby1.png

test_image = cv2.imread('C:/Users/user1/Desktop/human3.jpg')

#Converting to grayscale
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# Displaying the grayscale image
plt.imshow(test_image_gray, cmap='gray')

plt.show()

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)





haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5);

haar_cascade_eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

eyes_rects = haar_cascade_eyes.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5);

haar_cascade_hand = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")

hand_rects = haar_cascade_hand.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5);


# Let us print the no. of faces found
print('Faces found: ', len(faces_rects))
print('Eyes found: ', len(eyes_rects))
print('UpperBody found: ', len(hand_rects))



for (x,y,w,h) in faces_rects:
     cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

for (x,y,w,h) in eyes_rects:
     cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 0, 250), 2)

for (x,y,w,h) in hand_rects:
     cv2.rectangle(test_image, (x, y), (x+w, y+h), (250, 0, 0), 2)

plt.imshow(convertToRGB(test_image))

plt.show()

def detect_faces(cascade, test_image, scaleFactor = 1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 15)

    return image_copy

def detect_eyes(cascade, test_image, scaleFactor = 1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    eyes_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in eyes_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 15)

    return image_copy

def detect_hand(cascade, test_image, scaleFactor = 1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    hand_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in hand_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 15)

    return image_copy