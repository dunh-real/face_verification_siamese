from face_detector import FaceDetector
from network import SiameseNetwork
from dataloader import ReferenceDatabase
from verify_thread import VerificationWorker
import pygame
import cv2
from collections import deque
import os
import time
import numpy as np

class RealTimeVerificationSystem:
    """Optimized real-time verification system"""
    def __init__(self, model_weights_path, reference_file, 
                 camera_id=0, similarity_threshold=0.5):
        print("Initializing Real-Time Face Verification System...")
        
        # Initialize components
        self.face_detector = FaceDetector(scale_factor=2)  # Downscale for speed
        self.siamese_network = SiameseNetwork(model_weights_path)
        self.reference_db = ReferenceDatabase(reference_file)
        
        # Background verification worker
        self.verification_worker = VerificationWorker(
            self.siamese_network,
            self.reference_db,
            similarity_threshold
        )
        self.verification_worker.start()
        
        # Camera setup
        self.camera_id = camera_id
        self.cap = None
        
        # Parameters
        self.similarity_threshold = similarity_threshold
        
        # Performance tracking
        self.fps_queue = deque(maxlen=30)
        self.detection_fps_queue = deque(maxlen=30)
        
        # Face detection
        self.last_faces = []
        self.frame_skip = 2  # Process every N frames for detection
        self.frame_count = 0
        
        # Sound alert
        self.load_alert_sound()
        self.last_alert_time = 0
        self.alert_cooldown = 2.0
        
        # Running flag
        self.running = False
        
        print("System initialized successfully!")
        print(f"Total reference persons: {len(self.reference_db.reference_data)}")
    
    def load_alert_sound(self):
        """Create or load alert sound"""
        try:
            if os.path.exists('alert.wav'):
                self.alert_sound = pygame.mixer.Sound('alert.wav')
            else:
                import numpy as np
                from scipy.io import wavfile
                
                sample_rate = 44100
                duration = 0.5
                frequency = 1000
                
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio = np.sin(2 * np.pi * frequency * t) * 0.3
                audio = (audio * 32767).astype(np.int16)
                
                wavfile.write('alert.wav', sample_rate, audio)
                self.alert_sound = pygame.mixer.Sound('alert.wav')
            
            print("Alert sound loaded")
        except Exception as e:
            print(f"Could not load alert sound: {e}")
            self.alert_sound = None
    
    def play_alert(self):
        """Play alert sound with cooldown"""
        current_time = time.time()
        if (self.alert_sound is not None and 
            current_time - self.last_alert_time > self.alert_cooldown):
            self.alert_sound.play()
            self.last_alert_time = current_time
    
    def process_frame(self, frame):
        """Process single frame with optimization"""
        detection_start = time.time()
        
        # Only detect faces every N frames
        if self.frame_count % self.frame_skip == 0:
            faces = self.face_detector.detect_faces(frame)
            self.last_faces = faces
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w].copy()
                
                # Submit to worker thread (non-blocking)
                self.verification_worker.submit_face(face_roi)
        
        self.frame_count += 1
        
        # Calculate detection FPS
        detection_time = time.time() - detection_start
        if detection_time > 0:
            detection_fps = 1.0 / detection_time
            self.detection_fps_queue.append(detection_fps)
        
        return frame
    
    def draw_results(self, frame):
        """Draw bounding boxes and labels"""
        if len(self.last_faces) > 0:
            x, y, w, h = self.last_faces[0]
            
            # Get latest verification result
            match = self.verification_worker.get_result()
            
            # Draw bounding box
            color = (0, 255, 0) if match is None else (0, 0, 255) if match['type'] == 'wanted' else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Add label
            if match:
                label = f"ID:{match['person_id']} ({match['type']}) - {match['similarity']:.2%}"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Play alert for wanted persons
                if match['type'] == 'wanted':
                    self.play_alert()
        
        return frame
    
    def create_display_frame(self, camera_frame):
        """Create split-screen display"""
        h, w = camera_frame.shape[:2]
        
        # Create canvas for split screen
        display = np.zeros((h, w*2, 3), dtype=np.uint8)
        
        # Left side: Camera feed with stats
        left_frame = camera_frame.copy()
        
        # Add FPS
        avg_fps = np.mean(self.fps_queue) if len(self.fps_queue) > 0 else 0
        cv2.putText(left_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add latency (inverse of detection FPS)
        avg_det_fps = np.mean(self.detection_fps_queue) if len(self.detection_fps_queue) > 0 else 0
        latency = (1000.0 / avg_det_fps) if avg_det_fps > 0 else 0
        cv2.putText(left_frame, f"Latency: {latency:.1f}ms", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        display[:, :w] = left_frame
        
        # Right side: Matching results
        right_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        match = self.verification_worker.get_result()
        
        if match:
            # Display reference image
            ref_img = cv2.cvtColor(match['image'], cv2.COLOR_RGB2BGR)
            ref_h, ref_w = ref_img.shape[:2]
            
            # Resize to fit
            scale = min(w * 0.8 / ref_w, h * 0.6 / ref_h)
            new_w = int(ref_w * scale)
            new_h = int(ref_h * scale)
            
            ref_img_resized = cv2.resize(ref_img, (new_w, new_h))
            
            # Center the image
            start_x = (w - new_w) // 2
            start_y = 50
            
            right_frame[start_y:start_y+new_h, start_x:start_x+new_w] = ref_img_resized
            
            # Add match information
            cv2.putText(right_frame, "MATCH FOUND", (w//2-100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            y_offset = start_y + new_h + 40
            
            info_color = (0, 0, 255) if match['type'] == 'wanted' else (0, 255, 0)
            
            cv2.putText(right_frame, f"Person ID: {match['person_id']}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, info_color, 2)
            
            cv2.putText(right_frame, f"Type: {match['type'].upper()}", 
                       (20, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, info_color, 2)
            
            cv2.putText(right_frame, f"Similarity: {match['similarity']:.2%}", 
                       (20, y_offset + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, info_color, 2)
            
            # Add warning for wanted persons
            if match['type'] == 'wanted':
                cv2.putText(right_frame, "!!! WANTED PERSON !!!", 
                           (w//2-150, h-30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 0, 255), 3)
        else:
            cv2.putText(right_frame, "NO MATCH", (w//2-80, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)
        
        display[:, w:] = right_frame
        
        return display
    
    def run(self):
        """Run the real-time verification system"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        print("\nReal-Time Face Verification System Running (OPTIMIZED)...")
        print("Press 'q' to quit")
        print("Press 's' to save current match")
        print(f"Similarity threshold: {self.similarity_threshold}")
        
        fps_frame_count = 0
        fps_start_time = time.time()
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process frame (async verification)
                self.process_frame(frame)
                
                # Draw results
                frame_with_results = self.draw_results(frame)
                
                # Calculate display FPS
                fps_frame_count += 1
                elapsed_time = time.time() - fps_start_time
                if elapsed_time > 0:
                    fps = fps_frame_count / elapsed_time
                    self.fps_queue.append(fps)
                
                # Create and show display
                display_frame = self.create_display_frame(frame_with_results)
                cv2.imshow('Real-Time Face Verification System', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"match_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved screenshot: {filename}")
                
                # Reset FPS counter periodically
                if elapsed_time > 1.0:
                    fps_frame_count = 0
                    fps_start_time = time.time()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.verification_worker.stop()
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        print("System shut down")