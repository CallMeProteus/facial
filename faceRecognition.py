
import cv2
import load_known_faces
import recognize_faces
import display_frame
import numpy as np
from PIL import Image, ImageTk
import TkinterInitializer

class FaceRecognitionApp(TkinterInitializer.TkinterInitializer):
    def __init__(self, known_faces_dir):
        super().__init__()

        ''' defining a cascade, in other words, how a face looks like'''
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Load the images of known faces and their names
        self.known_faces, self.known_names = load_known_faces.load_known_faces(known_faces_dir, self.face_cascade)

        # Create a dictionary that maps the face encodings to the names of the people in the dataset
        self.face_encodings_dict = {}
        for i in range(len(self.known_names)):
            self.face_encodings_dict[self.known_names[i]] = self.known_faces[i]

        # Initialize the video capture device
        self.cap = cv2.VideoCapture(0)

        # start updating the frame
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()

        # Recognize faces in the frame and display matching faces
        frame = recognize_faces.recognize_faces(frame, self.face_cascade,
         self.face_encodings_dict,lbl=self.update_name_label, 
         tlbl=self.update_time_label,
         listbox = self.update_listbox1,
         currentuser = self.update_current_user_label
         )

        self.update_label_image(frame)

        # Schedule the update function to be called again
        self.Frame2.after(50, self.update_frame)

    def run(self):
        # run the main loop
        self.root.mainloop()

        # Release the video capture device and close all windows
        self.cap.release()
        cv2.destroyAllWindows()
