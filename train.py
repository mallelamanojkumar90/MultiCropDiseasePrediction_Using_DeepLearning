import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.model import build_model
import os
import argparse
import json

def train_model(data_dir, epochs=10, batch_size=32):
    """
    Trains the model on the provided dataset directory.
    Expected structure: data_dir/class_name/images...
    """
    # Image Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Save class indices
    if not os.path.exists('models'):
        os.makedirs('models')
    
    with open(os.path.join('models', 'class_indices.json'), 'w') as f:
        json.dump(train_generator.class_indices, f)
    
    print(f"Class indices saved to models/class_indices.json")

    # Build model
    num_classes = train_generator.num_classes
    model = build_model(num_classes=num_classes)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join('models', 'crop_disease_weights.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=0.00001
    )

    # Train
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Crop Disease Model')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    args = parser.parse_args()

    if not os.path.exists('models'):
        os.makedirs('models')

    train_model(args.data, epochs=args.epochs)
