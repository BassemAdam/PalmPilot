import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from time import sleep

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.prev_y = None
        self.scroll_threshold = 20
        self.smoothing = 0.5
        self.is_activated = False
        self.pinch_threshold = 0.05  # Distance threshold for pinch detection

    def detect_pinch(self, thumb_tip, index_tip):
        # Calculate distance between thumb and index finger
        distance = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )
        return distance < self.pinch_threshold

    def detect_and_scroll(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get thumb and index fingertip coordinates
                thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
                index_tip = hand_landmarks.landmark[8]  # Index tip
                
                # Check pinch gesture
                is_pinched = self.detect_pinch(thumb_tip, index_tip)
                
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Handle activation state
                if is_pinched and not self.is_activated:
                    self.is_activated = True
                    self.prev_y = index_tip.y  # Reset reference point
                elif not is_pinched:
                    self.is_activated = False
                
                # Calculate movement only when activated
                if self.is_activated and self.prev_y is not None:
                    curr_y = index_tip.y
                    y_delta = (curr_y - self.prev_y) * 1000
                    
                    if abs(y_delta) > self.scroll_threshold:
                        scroll_amount = int(y_delta)
                        pyautogui.scroll(-scroll_amount)
                        
                        # Visual feedback
                        color = (0, 0, 255) if y_delta > 0 else (0, 255, 0)
                        text = "Scrolling Down" if y_delta > 0 else "Scrolling Up"
                        cv2.putText(frame, text, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Update previous position with smoothing
                    self.prev_y = self.prev_y * (1 - self.smoothing) + curr_y * self.smoothing
                
                # Show activation state
                state_color = (0, 255, 0) if self.is_activated else (0, 0, 255)
                state_text = "Activated" if self.is_activated else "Deactivated"
                cv2.putText(frame, state_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)
                
        return frame

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    pyautogui.FAILSAFE = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame and handle scrolling
        output_frame = detector.detect_and_scroll(frame)
        
        # Display result
        cv2.imshow("Hand Scrolling", output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()