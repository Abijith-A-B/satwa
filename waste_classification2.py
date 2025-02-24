import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image


data_dir = '/home/abijith/trashnet_data/dataset-resized'   
classes = ['plastic', 'metal', 'paper', 'cardboard', 'trash', 'glass']

for class_name in classes:
    class_path = os.path.join(data_dir, class_name)
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"Folder {class_path} not found! Check your dataset.")
    print(f"{class_name}: {len(os.listdir(class_path))} images")

train_datagen = ImageDataGenerator(
    rescale=1./255, validation_split=0.2, rotation_range=20,
    width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical',
    subset='training', classes=classes
)

val_generator = train_datagen.flow_from_directory(
    data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical',
    subset='validation', classes=classes
)


base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(6, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=12)
model.save('waste_classifier.h5')
print("Model saved as waste_classifier.h5")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('waste_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
print("Model converted and saved as waste_classifier.tflite")


interpreter = tf.lite.Interpreter(model_path='waste_classifier.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test_img_path = '/home/abijith/trashnet_data/dataset-resized/test_image.jpg'   
img = image.load_img(test_img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

predicted_class = classes[np.argmax(output_data)]
confidence = np.max(output_data)
print(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")