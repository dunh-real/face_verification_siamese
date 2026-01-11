import pandas as pd
import cv2
import os

class ReferenceDatabase:
    """Reference database with single image per person"""
    def __init__(self, reference_file):
        self.reference_data = self.load_reference_data(reference_file)
        print(f"Loaded {len(self.reference_data)} reference faces")
        
        # Pre-prepare batch data for fast verification
        self.all_reference_images = [person['image'] for person in self.reference_data]
        print(f"Prepared batch verification for {len(self.all_reference_images)} persons")
    
    def load_reference_data(self, reference_file):
        """Load reference faces from file"""
        df = pd.read_csv(reference_file, sep='\t')
        
        reference_list = []
        for idx, row in df.iterrows():
            person_id = row['person_id']
            person_type = row['type']
            image_path = row['image_path']
            
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    reference_list.append({
                        'person_id': person_id,
                        'type': person_type,
                        'image': img,
                        'image_path': image_path
                    })
                else:
                    print(f"Warning: Could not read image {image_path}")
            else:
                print(f"Warning: Image not found {image_path}")
        
        return reference_list