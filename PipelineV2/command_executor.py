# command_executor.py
import pyautogui
import time


# command_executor.py
class CommandExecutor:
    def __init__(self):
        self.scroll_speed = 50
        self.zoom_factor = 1.1
        self.last_command_time = time.time()
        self.command_cooldown = 0.5  # seconds
        pyautogui.FAILSAFE = False
        
    def execute(self, gesture):
        current_time = time.time()
        if current_time - self.last_command_time < self.command_cooldown:
            return
            
        if gesture in ['SCROLL_UP', 'SCROLL_DOWN']:
            scroll_amount = self.scroll_speed * (1 if gesture == 'SCROLL_UP' else -1)
            pyautogui.scroll(scroll_amount)
            
        elif gesture == 'ZOOM_IN':
            pyautogui.hotkey('ctrl', '+')
            self.last_command_time = current_time
            
        elif gesture == 'ZOOM_OUT':
            pyautogui.hotkey('ctrl', '-')
            self.last_command_time = current_time
            
        elif gesture == 'CLOSE_PAGE':
            # Require confirmation for close
            pyautogui.hotkey('ctrl', 'w')
            self.last_command_time = current_time