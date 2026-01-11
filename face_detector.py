import cv2

class FaceDetector:
    """Optimized face detector"""
    def __init__(self, scale_factor=2):
        # Load pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.scale_factor = scale_factor  # Downscale for faster detection
    
    def detect_faces(self, frame):
        """Detect faces with optimization"""
        # Downscale for faster detection
        small_frame = cv2.resize(frame, 
                                (frame.shape[1] // self.scale_factor, 
                                 frame.shape[0] // self.scale_factor))
        
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(40, 40)
        )
        
        # Scale back up
        faces = [(x * self.scale_factor, y * self.scale_factor, 
                 w * self.scale_factor, h * self.scale_factor) 
                for (x, y, w, h) in faces]
        
        return faces