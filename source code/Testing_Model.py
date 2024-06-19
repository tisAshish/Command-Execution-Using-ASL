import cv2
import mediapipe as mp
import pickle
import numpy as np

# Different mediapipe objects to show landmarks
mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles

hands=mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)


# Loading/Unpickling our model
with open('model1.p','rb') as f:
    model_dict=pickle.load(f)
f.close()

model=model_dict['model']


# Create a labels dictionary
labels_dict={0:'A',1:'B',2:'C',
             3:'D',4:'E',5:'F',
             6:'G',7:'H',8:'I',
             9:'J',10:'K',11:'L',
             12:'M',13:'N',14:'O',
             15:'P',16:'Q',17:'R',
             18:'S',19:'T',20:'U',
             21:'V',22:'W',23:'X',
             24:'Y',25:'Z'}

start=[220,80]
end=[640,460]
color=(255, 0, 127) #bgr format

cap=cv2.VideoCapture(0)

while True:
    # waiting frame
    ret, frame = cap.read()
    # mirror the frame vertically
    frame=cv2.flip(frame,1)
    # show text on the frame
    cv2.putText(frame,
                'Press "Q" to Start!!',
                (100, 50), #text pos
                cv2.FONT_HERSHEY_DUPLEX, #font
                1.5, #fontscale
                color, #rgb color
                2, #thickness
                cv2.LINE_AA)
    # display the frame
    frame = cv2.rectangle(frame, (start[0], start[1]), (end[0], end[1]), color, 5)
    
    # Naming window
    cv2.namedWindow('Window', cv2.WINDOW_NORMAL) 
  
    # Resizing window
    cv2.resizeWindow('Window', 500, 500)
    
    # Moving window
    cv2.moveWindow('Window', 20, 20)
    
    cv2.imshow('Window', frame)
    
    if cv2.waitKey(10) == ord('q'):
        break

while True:
    # Initializing few auxiliary lists
    data_aux=[]
    
    ret, frame=cap.read()
    frame=cv2.flip(frame,1)
    cut_frame=frame[start[1]:end[1],start[0]:end[0]]
    frame_rgb=cv2.cvtColor(cut_frame, cv2.COLOR_BGR2RGB)
    results=hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                cut_frame, # Image to draw
                hand_landmarks, # Model output
                mp_hands.HAND_CONNECTIONS, # Hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
                )
            
        
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x=hand_landmarks.landmark[i].x # X coordinate
                y=hand_landmarks.landmark[i].y # Y coordinate
                
                data_aux.append(x)
                data_aux.append(y)
                   
        '''
        I have trained the RandomForestClassfier with 42 features
        This means I have used only 1 hand in frame. 
        So while predicting if I use more than 1 hand 
        then it will throw an error.
        To avoid that, the data_aux list 
        that stores the number of features is sliced to 42 elements
        '''
        # Prediction is a list of 1 item
        prediction=model.predict([np.asarray(data_aux[:42])])
        
        # Predicted Character
        predicted_char=labels_dict[int(prediction[0])]
        
        # Now lets display this character on the frame
        cv2.putText(frame,
                    predicted_char,# To display 
                    (150, 60), # Text pos
                    cv2.FONT_HERSHEY_DUPLEX, # Font
                    1.5, # Fontscale
                    color, # rgb color
                    2, # Thickness
                    cv2.LINE_AA)   
    
    frame = cv2.rectangle(frame, (start[0], start[1]), (end[0], end[1]), color, 5)
    
    # Naming window
    cv2.namedWindow('Window', cv2.WINDOW_NORMAL) 
  
    # Resizing window
    cv2.resizeWindow('Window', 500, 500)
    
    # Moving window
    cv2.moveWindow('Window', 20, 20) 
  
    cv2.imshow('Window',frame)
    
    #cv2.moveWindow('frame', 30, 50)
    if cv2.waitKey(10) == ord('e'):
        break
    
cap.release()
cv2.destroyAllWindows()