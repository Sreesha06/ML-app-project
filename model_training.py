"""
Crop Disease Detection Model Training
Tamil Nadu Agricultural Hackathon
Uses PlantVillage dataset (publicly available)
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Disease classes - Tamil Nadu focus
DISEASE_CLASSES = {
    'Healthy': 0,
    'Rice_Blast': 1,
    'Rice_LeafSpot': 2,
    'Rice_SheatRot': 3,
    'Cotton_LeafCurl': 4,
    'Cotton_Wilt': 5,
    'Cotton_Anthracnose': 6,
    'Sugarcane_RedRot': 7,
    'Sugarcane_Smut': 8,
    'Sugarcane_LeafScald': 9,
}

class CropDiseaseModel:
    """Model builder for crop disease detection"""
    
    def __init__(self, num_classes=len(DISEASE_CLASSES), img_size=IMG_SIZE):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        
    def build_model(self):
        """Build transfer learning model using MobileNetV2"""
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom layers for disease classification
        inputs = tf.keras.Input(shape=(self.img_size, self.img_size, 3))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return self.model
    
    def create_data_generators(self, train_dir, val_dir):
        """Create data generators for training and validation"""
        
        # Training augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Validation (minimal augmentation)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        # Load validation data
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator, steps_per_epoch=100):
        """Train the model"""
        
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=val_generator,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_disease_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True
                )
            ]
        )
        
        return history
    
    def save_model(self, filepath='crop_disease_model.h5'):
        """Save trained model"""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='crop_disease_model.h5'):
        """Load pre-trained model"""
        self.model = tf.keras.models.load_model(filepath)
        return self.model


def predict_disease(model, image_path, class_names):
    """Predict disease from single image"""
    
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=(IMG_SIZE, IMG_SIZE)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx])
    
    return class_names[class_idx], confidence, predictions[0]


# Example usage for training
if __name__ == "__main__":
    print("🌾 Crop Disease Detection Model - Training Setup")
    print("=" * 50)
    
    # Initialize model
    model_builder = CropDiseaseModel(num_classes=len(DISEASE_CLASSES))
    model = model_builder.build_model()
    
    print(f"✅ Model built successfully")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output classes: {len(DISEASE_CLASSES)}")
    print(f"   Trainable params: {model.count_params():,}")
    
    print("\n📚 Disease Classes:")
    for disease, idx in DISEASE_CLASSES.items():
        print(f"   {idx}: {disease}")
    
    print("\n🔧 Training Configuration:")
    print(f"   Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Learning rate: {LEARNING_RATE}")
    
    print("\n⚠️  To train, you need:")
    print("   1. PlantVillage dataset (download from: https://github.com/spMohanty/PlantVillage-Dataset)")
    print("   2. Organize as: data/train/<disease>/ and data/val/<disease>/")
    print("   3. Run: train_generators = model_builder.create_data_generators('data/train', 'data/val')")
    print("   4. Run: model_builder.train(train_gen, val_gen)")
if __name__ == "__main__":
    model_builder = CropDiseaseModel(num_classes=len(DISEASE_CLASSES))
    model = model_builder.build_model()

    train_gen, val_gen = model_builder.create_data_generators('data/train', 'data/val')

    model_builder.train(train_gen, val_gen)

    model_builder.save_model("model.h5")
