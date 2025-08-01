
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- Configuration for Training --- #
IMAGE_SIZE = (128, 128) # Smaller size for faster training and demonstration
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 2 # Weeds and Plants

# --- Data Preparation (Simulated) --- #
def create_dummy_dataset(base_dir="./data"):
    train_dir = os.path.join(base_dir, "train")
    validation_dir = os.path.join(base_dir, "validation")
    
    os.makedirs(os.path.join(train_dir, "weeds"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "plants"), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, "weeds"), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, "plants"), exist_ok=True)
    
    print(f"Dummy dataset structure created in {base_dir}")
    print("Please populate these directories with actual images for training.")
    print("Example: data/train/weeds/weed_1.jpg, data/train/plants/plant_1.jpg")

# --- Model Definition --- #
def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax') # Use softmax for multi-class classification
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', # Use sparse_categorical_crossentropy for integer labels
                  metrics=['accuracy'])
    return model

# --- Training Function --- #
def train_model(data_dir="./data", model_save_path="./weed_plant_detector.keras"):
    train_dir = os.path.join(data_dir, "train")
    validation_dir = os.path.join(data_dir, "validation")
    
    # Check if data directories are populated
    if not os.listdir(os.path.join(train_dir, "weeds")) or \
       not os.listdir(os.path.join(train_dir, "plants")):
        print("Training data not found. Please populate the dummy dataset directories.")
        create_dummy_dataset(data_dir)
        return
        
    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse') # Use sparse for integer labels
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse') # Use sparse for integer labels
        
    model = create_model(IMAGE_SIZE + (3,), NUM_CLASSES)
    model.summary()
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )
    
    # Save the model in the native Keras format
    model.save(model_save_path)
    print(f"Model trained and saved to {model_save_path}")
    
    return history

if __name__ == "__main__":
    create_dummy_dataset()
    # Uncomment the line below to run training after populating the dataset
    train_model()


