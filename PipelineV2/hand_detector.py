# hand_detector.py
import mediapipe as mp

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def detect(self, frame):
        results = self.hands.process(frame)
        return results.multi_hand_landmarks if results.multi_hand_landmarks else None