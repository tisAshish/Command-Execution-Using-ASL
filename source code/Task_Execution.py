import cv2
import mediapipe as mp
import pickle
import numpy as np
import webbrowser
import wikipedia
import pyttsx3
from pynput.keyboard import Controller, Key
from youtube_search import YoutubeSearch
from bs4 import BeautifulSoup
import requests

# Different mediapipe objects to show landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
keyboard = Controller()

# Loading/Unpickling our model
with open('model1.p', 'rb') as f:
    model_dict = pickle.load(f)
    f.close()

model = model_dict['model']

# Create a labels dictionary
labels_dict = {0: 'A',1: 'B',2: 'C',3: 'D',4: 'E',5: 'F',6: 'G',7: 'H',8: 'I',
               9: 'J',10: 'K',11: 'L',12: 'M',13: 'N',14: 'O',15: 'P',16: 'Q',
              17: 'R',18: 'S',19: 'T',20: 'U',21: 'V',22: 'W',23: 'X',
              24: 'Y',25: 'Z'}


def wikipedia_info(formed_word):
    try:
        wikipedia.set_lang("en")  # Set Wikipedia language to English
        summary = wikipedia.summary(formed_word, sentences=2)  # Get summary of the formed word
        print("Wikipedia Summary:", summary)

        # Initialize the text-to-speech engine
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech

        # Read aloud the summary
        engine.say(summary)
        engine.runAndWait()
    except wikipedia.exceptions.PageError:
        print("Wikipedia page not found for the word:", formed_word)
    except wikipedia.exceptions.DisambiguationError:
        print("Ambiguous term. Please provide more specific word.")
    except Exception as e:
        print("Error:", str(e))


def youtube_video(formed_word):
    try:
        results = YoutubeSearch(formed_word, max_results=1).to_dict()
        if results:
            video_url = "https://www.youtube.com" + results[0]['url_suffix']
            webbrowser.open(video_url)
            print("Playing video for:", formed_word)
        else:
            print("No video found for:", formed_word)
    except Exception as e:
        print("Error playing YouTube video:", str(e))


def read_weather():
    try:
        url = "https://www.myweather2.com/City-Town/India/Kolkata.aspx"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        weather_info = soup.find('div',id='weather_summary_spot').text.strip()
        
        # Initialize the text-to-speech engine
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech

        # Read aloud the weather info
        engine.say(f"{weather_info}")
        engine.runAndWait()

    except Exception as e:
        print("Error reading weather information:", str(e))

    
start=[20,80]
end=[300,380]
color=(255, 0, 127) #bgr format

cap=cv2.VideoCapture(0)

###WAITING WINDOW-----------------------
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
    # Display rectangle
    frame = cv2.rectangle(frame, (start[0], start[1]), (end[0], end[1]), color, 5)
    
    # Naming window
    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
  
    # Resizing window
    cv2.resizeWindow("Resized_Window", 500, 500)
    
    # Moving window
    cv2.moveWindow("Resized_Window", 20, 20) 
  
    # Displaying the frame 
    cv2.imshow("Resized_Window", frame) 
    
    if cv2.waitKey(25) == ord('q'):
        break

# Empty list to store predicted letters
pred_letters = []
command=set()

###COMMAND PREDICTION WINDOW----------------------
while True:
    #command prediction frame
    data_aux=[]
    
    ret, frame=cap.read()
    frame=cv2.flip(frame,1)
    cut_frame=frame[start[1]:end[1],start[0]:end[0]]
    frame_rgb=cv2.cvtColor(cut_frame, cv2.COLOR_BGR2RGB)
    results=hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x=hand_landmarks.landmark[i].x # X coordinate
                y=hand_landmarks.landmark[i].y # Y coordinate
                
                data_aux.append(x)
                data_aux.append(y)
        
        # Prediction is a list of 1 item
        prediction=model.predict([np.asarray(data_aux[:42])])
        
        # Predicted Character
        pred_char=labels_dict[int(prediction[0])]
        pred_letters.append(pred_char)
        
        
        # Now lets display this character on the frame
        cv2.putText(frame,
                    pred_char,# To display 
                    (150, 60), # Text pos
                    cv2.FONT_HERSHEY_DUPLEX, # Font
                    1.5, # Fontscale
                    color, # rgb color
                    2, # Thickness
                    cv2.LINE_AA)
        
    frame = cv2.rectangle(frame, (start[0], start[1]), (end[0], end[1]), color, 5)
    
    # Naming window
    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
  
    # Resizing window
    cv2.resizeWindow("Resized_Window", 500, 500)
    
    # Moving window
    cv2.moveWindow("Resized_Window", 20, 20) 
  
    # Displaying the frame 
    cv2.imshow("Resized_Window", frame) 
        
    # Check if specific gesture is performed (e.g., fist close)
    if len(pred_letters) >= 5:  # Minimum 5 letters required
        command = set(pred_letters[-5:]) # Convert last five letters to set
    if (command == {'W'} or command=={'C'} or command=={'B'} or command == {'K'}):
        #command recognized so exit while
        break
    
    if cv2.waitKey(25) == ord('e'):
        break
 
print('Command Recognized=',command)

###WEATHER COMMAND--------------------------------

if command == {'B'}:
    cap.release()
    cv2.destroyAllWindows()
    
    # Read weather info aloud
    read_weather()
    
    
    
###WIKIPEDIA OR YOUTUBE COMMAND ----------------------------------
    
if command == {'W'} or command == {'C'}:
    pred_letters = []
    formed_word=""
    while True:
        #frame for forming words to give as input to command
        data_aux=[]
        
        ret, frame=cap.read()
        frame=cv2.flip(frame,1)
        cut_frame=frame[start[1]:end[1],start[0]:end[0]]
        frame_rgb=cv2.cvtColor(cut_frame, cv2.COLOR_BGR2RGB)
        results=hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x=hand_landmarks.landmark[i].x # X coordinate
                    y=hand_landmarks.landmark[i].y # Y coordinate
                
                    data_aux.append(x)
                    data_aux.append(y)
        
            # Prediction is a list of 1 item
            prediction=model.predict([np.asarray(data_aux[:42])])
        
            # Predicted Character
            pred_char=labels_dict[int(prediction[0])]
            pred_letters.append(pred_char)
        
            formed_word = "".join(set(pred_letters))
               
        # Now lets display this character on the frame
        cv2.putText(frame,
                    formed_word,# To display 
                    (20, 60), # Text pos
                    cv2.FONT_HERSHEY_DUPLEX, # Font
                    1.5, # Fontscale
                    color, # rgb color
                    2, # Thickness
                    cv2.LINE_AA)
        frame = cv2.rectangle(frame, (start[0], start[1]), (end[0], end[1]), color, 5)
        
        # Naming window
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
      
        # Resizing window
        cv2.resizeWindow("Resized_Window", 500, 500)
        
        # Moving window
        cv2.moveWindow("Resized_Window", 20, 20) 
      
        # Displaying the frame 
        cv2.imshow("Resized_Window", frame) 
    
        if cv2.waitKey(25) == ord('d'):
            formed_word=formed_word[:-1]
            pred_letters=pred_letters[:-1]
    
        if cv2.waitKey(25) == ord('c'):
            formed_word=""
            pred_letters=[]
        
        if cv2.waitKey(25) == ord('e'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
             
    print("Formed Text=",formed_word)

    if command=={'W'}:
        wikipedia_info(formed_word)
        
    if command=={'C'}:
        youtube_video(formed_word)


###VOLUME INCREASE DECREASE COMMAND--------------------

if command == {'K'}:
    while True:
        ret, frame=cap.read()
        frame=cv2.flip(frame,1)
        cut_frame=frame[start[1]:end[1],start[0]:end[0]]
        frame_rgb=cv2.cvtColor(cut_frame, cv2.COLOR_BGR2RGB)
        results=hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    pass
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = cut_frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Index finger
                if id == 8:
                    cv2.circle(cut_frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    
                    if cx <= 120:
                        keyboard.press(Key.media_volume_down)
                        keyboard.release(Key.media_volume_down)
                        
                    elif cx >= 200:
                        keyboard.press(Key.media_volume_up)
                        keyboard.release(Key.media_volume_up)
                        
                        
        frame = cv2.rectangle(frame, (start[0], start[1]), (end[0], end[1]), color, 5)
        
        # Naming window
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
      
        # Resizing window
        cv2.resizeWindow("Resized_Window", 500, 500)
        
        # Moving window
        cv2.moveWindow("Resized_Window", 20, 20) 
      
        # Displaying the frame 
        cv2.imshow("Resized_Window", frame)
        
        if cv2.waitKey(25) == ord('e'):
            break
        
cap.release()
cv2.destroyAllWindows()