import os
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

mlflow.set_tracking_uri("file:../mlruns")
mlflow.set_experiment("Default")
# Enable MLflow autologging
mlflow.tensorflow.autolog()



# === Paths ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'cat_dog_model.h5')

# Create model output directory if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# === Image Generators ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

val_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# === Model: Transfer Learning ===
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === Train & Log with MLflow ===
with mlflow.start_run():
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=2
    )

    # Save model manually
    model.save(MODEL_PATH)
    print(f"✅ Model saved at: {MODEL_PATH}")
