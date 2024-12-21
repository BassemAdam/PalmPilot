import os
import cv2
import numpy as np
import util




def process_segments(image):
    # Convert the image to HSV and YCrCb color spaces
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
    
    # Optimized thresholds for skin detection
    lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
    upper_hsv = np.array([20, 150, 255], dtype=np.uint8)
    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
    
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    # Combine masks with weights
    combined_mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    return skin_mask









def generate_images() -> None:

    # if data folder does not exist, generate it
    if not os.path.exists(os.path.join(util.script_dir, util.DATA_DIR)):
        os.makedirs(os.path.join(util.script_dir, util.DATA_DIR))

    # Initialize CAM
    cap = cv2.VideoCapture(0)

    # for each gesture, generate images
    for j in util.gestures:
        
        # if gesture folder does not exist, generate it
        if not os.path.exists(os.path.join(util.script_dir, util.DATA_DIR, j)):
            os.makedirs(os.path.join(util.script_dir, util.DATA_DIR, j))

        # logging
        print(f'Collecting data for class {j}')
        
        # waiting for the user to get ready
        while True:
            # read the frame and flip it
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            # ROI based solution
            roi = frame[50:250, 50:250]
            cv2.rectangle(frame, (49, 49), (251, 251), (0, 255, 0), 0)
            # print statement
            cv2.putText(frame, f'Press "Q" {j}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            # debugging
            roi = process_segments(roi)
            cv2.imshow('roi', roi)
            # if user pressed q, break the waiting and start taking snippets
            if cv2.waitKey(10) == ord('q'):
                return
            if cv2.waitKey(10) == ord('r'):
                break

        # iterator of the current gesture dataset
        counter = 0
        while counter < util.dataset_size:
            # read the frame and flip it
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            # ROI based solution
            roi = frame[50:250, 50:250]
            cv2.rectangle(frame, (49, 49), (251, 251), (0, 255, 0), 0)
            # print statement
            cv2.imshow('frame', frame)
            # segment the image using kmeans
            roi = process_segments(roi)
            # debugging
            cv2.imshow('roi', roi)
            # await between each snippet
            cv2.waitKey(25)
            # save the image in the directory
            cv2.imwrite(os.path.join(util.script_dir, util.DATA_DIR, j, '{}.png'.format(counter)), roi)
            # increment the counter
            counter += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    generate_images()