import tensorflow as tf
from keras import layers, Model
from keras.applications import MobileNetV2
import numpy as np
import cv2

class SiameseNetwork:
    """Optimized Siamese Network with batch processing"""
    def __init__(self, model_weights_path, input_shape=(128, 128, 3), embedding_dim=128):
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.model = self.build_model()
        self.model.load_weights(model_weights_path)
        print(f"Model loaded from: {model_weights_path}")
    
    def create_base_network(self):
        """Create base network matching training architecture"""
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape,
            alpha=1.0
        )
        
        for layer in base_model.layers:
            layer.trainable = False
        
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Dense(512, activation='relu', 
                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(self.embedding_dim, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        
        embeddings = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(x)
        
        model = Model(inputs=base_model.input, outputs=embeddings)
        return model
    
    def build_model(self):
        """Build complete Siamese network"""
        base_network = self.create_base_network()
        
        input_a = layers.Input(shape=self.input_shape, name='input_a')
        input_b = layers.Input(shape=self.input_shape, name='input_b')
        
        embedding_a = base_network(input_a)
        embedding_b = base_network(input_b)
        
        l1_distance = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([embedding_a, embedding_b])
        
        x = layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001))(l1_distance)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        model = Model(inputs=[input_a, input_b], outputs=output)
        return model
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        img = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        img = img.astype('float32') / 255.0
        return img
    
    def verify_batch(self, face_image, reference_images):
        """Batch verify face against all references - MUCH FASTER"""
        # Preprocess detected face once
        img1 = self.preprocess_image(face_image)
        
        # Preprocess all reference images
        batch_size = len(reference_images)
        img1_batch = np.array([img1] * batch_size)
        img2_batch = np.array([self.preprocess_image(ref) for ref in reference_images])
        
        # Single batch prediction - much faster than loop
        predictions = self.model.predict([img1_batch, img2_batch], verbose=0)
        
        return predictions.flatten()