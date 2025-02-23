import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
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


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.savefig('training_plot.png')
print("Training plot saved as training_plot.png")


img_path = '/home/abijith/Desktop/waste_data/test_image.jpg'  
img = cv2.imread(img_path)
if img is None:
    print("Error: Could not load test_image.jpg. Please check the path.")
else:
    img = cv2.resize(img, img_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    keras_pred = model.predict(img)
    labels = list(train_gen.class_indices.keys())  # Use class_indices for consistency
    keras_prediction = labels[np.argmax(keras_pred[0])]
    print(f"Keras model prediction: {keras_prediction}")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('waste_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Model saved as waste_model.tflite")


interpreter = tf.lite.Interpreter(model_path='waste_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


if img is not None:  
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    tflite_prediction = labels[np.argmax(output_data[0])] 
    print(f"TFLite model prediction: {tflite_prediction}")

print("Training and testing complete!")