# gesture_classifier.py

class GestureClassifier:
    def __init__(self):
        self.pinch_threshold = 0.05
        self.movement_threshold = 0.01
        self.gesture_cooldown = 20  # frames
        self.last_gesture = None
        self.cooldown_counter = 0
        
    def check_consistent_gesture(self, history, required_state, frames=5):
        if len(history) < frames:
            return False
        return all(self.compare_fingers(gesture, required_state) 
                  for gesture in history[-frames:])
    
    def compare_fingers(self, gesture1, gesture2):
        return all(a == b for a, b in zip(gesture1, gesture2))
    
    def classify(self, features):
        if not features or self.cooldown_counter > 0:
            self.cooldown_counter = max(0, self.cooldown_counter - 1)
            return None
            
        fingers = features['fingers_extended']
        history = features['gesture_history']
        movement = features['movement']
        
        # Scroll activation (pinch gesture)
        if features['pinch_distance'] < self.pinch_threshold:
            if abs(movement['y']) > self.movement_threshold:
                return 'SCROLL_UP' if movement['y'] < 0 else 'SCROLL_DOWN'
        
        # Close page gesture (all fingers closed + horizontal swipe)
        close_page_gesture = [False, False, False, False, False]  # fist
        if (self.check_consistent_gesture(history, close_page_gesture, 10) and
            abs(movement['x']) > self.movement_threshold * 3):
            self.cooldown_counter = self.gesture_cooldown
            return 'CLOSE_PAGE'
        
        # Zoom in (peace sign + spread)
        zoom_in_gesture = [False, True, True, False, False]  # peace sign
        if self.check_consistent_gesture(history, zoom_in_gesture, 5):
            curr_distance = features['pinch_distance']
            if curr_distance > self.pinch_threshold * 4:  # spreading threshold
                self.cooldown_counter = self.gesture_cooldown
                return 'ZOOM_IN'
        
        # Zoom out (peace sign + pinch)
        if self.check_consistent_gesture(history, zoom_in_gesture, 5):
            if features['pinch_distance'] < self.pinch_threshold:
                self.cooldown_counter = self.gesture_cooldown
                return 'ZOOM_OUT'
                
        return None