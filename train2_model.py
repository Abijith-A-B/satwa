import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os


data_dir = '/home/abijith/Desktop/waste_data'  
img_size = (224, 224)    
batch_size = 16         


datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    classes=['bio-waste', 'plastic', 'paper']
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    classes=['bio-waste', 'plastic', 'paper']
)


print("Class indices:", train_gen.class_indices)


base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

model.save('waste_model.h5')
print("Model saved as waste_model.h5")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('waste_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Model saved as waste_model.tflite")


def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess the input image for classification."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Could not load image at {image_path}. Please check the path.")
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_trash(image_path, interpreter, input_details, output_details, labels):
    """Classify the trash image using the TFLite model."""
    img = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = labels[np.argmax(output_data[0])]
    return prediction


interpreter = tf.lite.Interpreter(model_path='waste_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
labels = list(train_gen.class_indices.keys())  


print("\nTrash Classification Tool")
print("Enter the full path to a trash image (or type 'exit' to quit):")
while True:
    user_input = input("Image path: ").strip()
    if user_input.lower() == 'exit':
        break
    if not os.path.exists(user_input):
        print("Error: File does not exist. Please provide a valid path.")
        continue
    try:
        prediction = classify_trash(user_input, interpreter, input_details, output_details, labels)
        print(f"Predicted class: {prediction}")
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

print("Classification tool terminated.")