import threading
from queue import Queue
import numpy as np

class VerificationWorker(threading.Thread):
    """Background worker thread for face verification"""
    def __init__(self, siamese_network, reference_db, similarity_threshold):
        threading.Thread.__init__(self)
        self.daemon = True
        
        self.siamese_network = siamese_network
        self.reference_db = reference_db
        self.similarity_threshold = similarity_threshold
        
        self.input_queue = Queue(maxsize=2)  # Limit queue size
        self.result = None
        self.result_lock = threading.Lock()
        self.running = True
    
    def find_best_match(self, face_image):
        """Find best matching person using batch processing"""
        # Batch verify against ALL references at once
        similarities = self.siamese_network.verify_batch(
            face_image,
            self.reference_db.all_reference_images
        )
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = float(similarities[best_idx])
        
        if best_similarity > self.similarity_threshold:
            person_data = self.reference_db.reference_data[best_idx]
            return {
                'person_id': person_data['person_id'],
                'type': person_data['type'],
                'similarity': best_similarity,
                'image': person_data['image'],
                'image_path': person_data['image_path']
            }
        
        return None
    
    def run(self):
        """Worker thread main loop"""
        while self.running:
            try:
                face_image = self.input_queue.get(timeout=0.1)
                if face_image is not None:
                    match = self.find_best_match(face_image)
                    with self.result_lock:
                        self.result = match
            except:
                continue
    
    def submit_face(self, face_image):
        """Submit face for verification (non-blocking)"""
        # Clear queue and add new face
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except:
                break
        
        if not self.input_queue.full():
            self.input_queue.put(face_image)
    
    def get_result(self):
        """Get latest verification result"""
        with self.result_lock:
            return self.result
    
    def stop(self):
        """Stop worker thread"""
        self.running = False