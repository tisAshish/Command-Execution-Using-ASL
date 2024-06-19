import os
import mediapipe as mp
import cv2
import pickle

# different mediapipe objects for display
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# model for detecting hand landmarks
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
# DATA_DIR is the main data directory, dir_ refers to the 4 subdirectories/classes

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_name in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img_path = os.path.join(DATA_DIR, dir_, img_name)
        img = cv2.imread(img_path)
        # cv2 reads image in BGR so convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # results stores all the image hand landmarks
        results = hands.process(img_rgb)

        # now iterating through results, if multiple landmarks found then...
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    # print(hand_landmarks.landmark[i])
                    x = hand_landmarks.landmark[i].x  # x coordinate
                    y = hand_landmarks.landmark[i].y  # y coordinate
                    data_aux.append(x)
                    data_aux.append(y)

            data.append(data_aux[:42])
            labels.append(dir_)

# now lets save our dataset by pickling it
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'label': labels}, f)

