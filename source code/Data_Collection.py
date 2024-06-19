import os
import cv2

# first step is to collect images of hand and store it in folder say "data"

# if such directory does not exist then create one
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Handsigns- A,B,C,D,E,F
number_of_classes = 2
dataset_size = 3 #for each class

# select camera index.
cap = cv2.VideoCapture(0)
# cap is a videocapture object

# creating 4 subdirectories for 4 classes
for j in range(0,number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        # waiting frame
        ret, frame = cap.read()
        # mirror the frame vertically
        frame=cv2.flip(frame,1)
        # show text on the frame
        cv2.putText(frame,
                    'Press "Q" to Start',
                    (240, 100), #text pos
                    cv2.FONT_HERSHEY_DUPLEX, #font
                    1.3, #fontscale
                    (0, 255, 0), #rgb color
                    2, #thickness
                    cv2.LINE_AA)
        # display the frame
        frame = cv2.rectangle(frame, (220, 20), (640, 460), (100, 255, 100), 10)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    #stop if 200 imgs captured for each class
    while counter < dataset_size:
        # reading frame
        ret, frame = cap.read()
        # mirror the frame vertically
        frame=cv2.flip(frame,1)
        cut_frame = frame[20:460, 220:640]

        cv2.imshow("Preview Frame", cut_frame)
        frame = cv2.rectangle(frame, (220, 20), (640, 460), (100, 255, 100), 10)
        cv2.imshow('Frame', frame)
        # img captured every 1500 msecs
        cv2.waitKey(1500)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), cut_frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()