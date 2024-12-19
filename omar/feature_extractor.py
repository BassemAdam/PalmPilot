# feature_extractor.py
import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.prev_landmarks = None
        self.gesture_history = []
        self.history_size = 10
        
    def get_finger_states(self, hand):
        # Return if fingers are extended (True) or closed (False)
        thumb_tip = hand[4]
        thumb_ip = hand[3]
        fingers = []
        
        # Thumb
        fingers.append(thumb_tip.x > thumb_ip.x)
        
        # Other fingers - compare y coordinates
        for i in range(8, 21, 4):
            tip = hand[i].y
            pip = hand[i-2].y
            fingers.append(tip < pip)
            
        return fingers
    
    def extract_features(self, landmarks):
        if not landmarks:
            return None
            
        hand = landmarks[0].landmark
        fingers_extended = self.get_finger_states(hand)
        
        # Calculate key points and distances
        thumb_tip = hand[4]
        index_tip = hand[8]
        middle_tip = hand[12]
        ring_tip = hand[16]
        pinky_tip = hand[20]
        
        pinch_distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + 
                               (thumb_tip.y - index_tip.y)**2)
        
        # Track hand movement
        movement = {'x': 0, 'y': 0}
        if self.prev_landmarks:
            prev_palm = self.prev_landmarks[0].landmark[0]
            curr_palm = hand[0]
            movement = {
                'x': curr_palm.x - prev_palm.x,
                'y': curr_palm.y - prev_palm.y
            }
        
        self.prev_landmarks = landmarks
        
        # Store gesture history
        self.gesture_history.append(fingers_extended)
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
            
        return {
            'fingers_extended': fingers_extended,
            'pinch_distance': pinch_distance,
            'movement': movement,
            'gesture_history': self.gesture_history,
            'hand_landmarks': hand
        }