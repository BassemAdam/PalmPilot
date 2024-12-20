import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from classifier_testV2 import process_segments, test_single_image
from command_executor import CommandExecutor

class GestureRecognitionGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Gesture Recognition System")
        
        # Initialize camera and executor
        self.cap = cv2.VideoCapture(0)
        self.executor = CommandExecutor()
        self.frames_count = 1
        
        # Create GUI elements
        self.create_widgets()
        
        # Start video loop
        self.update_video()
        
    def create_widgets(self):
        # Main container
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(padx=10, pady=10, expand=True, fill='both')
        
        # Video feed
        self.video_label = ttk.Label(self.main_frame)
        self.video_label.grid(row=0, column=0, padx=5, pady=5)
        
        # Segmentation view
        self.segment_label = ttk.Label(self.main_frame)
        self.segment_label.grid(row=0, column=1, padx=5, pady=5)
        
        # Controls panel
        self.controls_frame = ttk.LabelFrame(self.main_frame, text="Controls")
        self.controls_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky='ew')
        
        # Start/Stop button
        self.is_running = True
        self.toggle_button = ttk.Button(self.controls_frame, 
                                      text="Stop", 
                                      command=self.toggle_camera)
        self.toggle_button.pack(side='left', padx=5)
        
        # Gesture display
        self.gesture_label = ttk.Label(self.controls_frame, 
                                     text="Detected Gesture: None")
        self.gesture_label.pack(side='right', padx=5)
        
    def toggle_camera(self):
        self.is_running = not self.is_running
        self.toggle_button.config(text="Start" if not self.is_running else "Stop")
        
    def update_video(self):
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                
                # Process frame
                original, segmented, blue_part, red_part = process_segments(frame)
                
                # Update gesture recognition
                if red_part is not None and self.frames_count % 10 == 0:
                    gesture = test_single_image(red_part)
                    if gesture:
                        self.executor.execute(gesture)
                        self.gesture_label.config(text=f"Detected Gesture: {gesture}")
                        self.frames_count = 0
                
                self.frames_count += 1
                
                # Convert frames for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_tk = ImageTk.PhotoImage(frame_pil)
                self.video_label.configure(image=frame_tk)
                self.video_label.image = frame_tk
                
                if segmented is not None:
                    seg_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
                    seg_pil = Image.fromarray(seg_rgb)
                    seg_tk = ImageTk.PhotoImage(seg_pil)
                    self.segment_label.configure(image=seg_tk)
                    self.segment_label.image = seg_tk
        
        self.window.after(10, self.update_video)
        
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

def main():
    root = tk.Tk()
    app = GestureRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()