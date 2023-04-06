import cv2

def display_frame(frame):
    cv2.imshow('Facial Recognition', frame)
    return cv2.waitKey(1) & 0xFF == ord('q')
