"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import cv2


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)

        img = cv2.line(img, (100,100), (300,300), (0,0,255),4) 

        img = cv2.rectangle(img, (250,30), (450,200), (0,255,0), 5)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(img, 'This one!', (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.putText(img, text, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()
