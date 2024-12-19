# main.py
import cv2
from camera import CameraCapture
from preprocessing import ImagePreprocessor
from hand_detector import HandDetector
from feature_extractor import FeatureExtractor
from gesture_classifier import GestureClassifier
from command_executor import CommandExecutor


def main():
    camera = CameraCapture()
    preprocessor = ImagePreprocessor()
    # we will later need to replace those two parts which relay on media pipe with our own implementation
    detector = HandDetector()
    feature_extractor = FeatureExtractor()
    # end of media pipe
    classifier = GestureClassifier()
    executor = CommandExecutor()
    
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
            
        processed_frame = preprocessor.process(frame)
        hand_landmarks = detector.detect(processed_frame)
        features = feature_extractor.extract_features(hand_landmarks)
        gesture = classifier.classify(features)
        
        if gesture:
            executor.execute(gesture, features)
            # Visual feedback
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if hand_landmarks:
            detector.mp_draw.draw_landmarks(
                frame, hand_landmarks[0], 
                detector.mp_hands.HAND_CONNECTIONS
            )
        
        cv2.imshow("Hand Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()