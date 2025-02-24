import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


print("Generating mock search history...")
data = {
    'user_id': [1, 1, 2, 2, 3, 4, 5],
    'waste_type': [0, 1, 2, 0, 3, 0, 1],
    'days_ago': [1, 3, 2, 5, 1, 4, 2],
    'frequency': [5, 2, 4, 1, 3, 6, 3]
}
df = pd.DataFrame(data)
df['weight'] = np.exp(-df['days_ago'] / 7)
df['weighted_freq'] = df['frequency'] * df['weight']
print(df)


user_encoder = LabelEncoder()
df['user_id_encoded'] = user_encoder.fit_transform(df['user_id'])
X = df[['user_id_encoded']].values
y = tf.keras.utils.to_categorical(df['waste_type'], num_classes=4)

print("\nTraining recommendation model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Embedding(input_dim=len(user_encoder.classes_), output_dim=8),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=4, verbose=1)

print("\nConverting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('waste_recommendation_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("TFLite model saved as 'waste_recommendation_model.tflite'")


interpreter = tf.lite.Interpreter(model_path='waste_recommendation_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

user_id = 1  
user_encoded = user_encoder.transform([user_id])
input_data = np.array([[user_encoded[0]]], dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])[0]
labels = ['Plastic', 'Metal', 'Paper', 'Mixed']
top_recommendations = np.argsort(output)[::-1][:2]
print(f"\nUser {user_id} Recommendations:")
for idx in top_recommendations:
    print(f"- {labels[idx]} (Probability: {output[idx]:.2f})")