import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import os


np.random.seed(42)
waste_types = ['plastic', 'metal', 'paper', 'cardboard', 'trash', 'glass']
base_prices = {'plastic': 15, 'metal': 40, 'paper': 10, 'cardboard': 12, 'trash': 3, 'glass': 8}


data = []
for _ in range(1000):
    waste = np.random.choice(waste_types)
    base_price = base_prices[waste]
    quality = np.random.uniform(0, 1)  
    
    price_variation = base_price * 0.2 * (quality - 0.5)  
    price = max(0, base_price + price_variation + np.random.normal(0, 2))
    data.append([waste, quality, price])

df = pd.DataFrame(data, columns=['waste_type', 'quality', 'price_inr'])
print("Dataset Preview:")
print(df.head())
print(f"Total samples: {df.shape[0]}")


encoder = OneHotEncoder(sparse_output=False)
waste_encoded = encoder.fit_transform(df[['waste_type']])
X = np.hstack((waste_encoded, df[['quality']].values))  
y = df['price_inr'].values


train_idx = int(0.8 * len(X))
X_train, X_test = X[:train_idx], X[train_idx:]
y_train, y_test = y[:train_idx], y[train_idx:]
print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(7,)),  
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)


train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTrain MAE: {train_mae:.2f} INR, Test MAE: {test_mae:.2f} INR")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('advanced_price_predictor.tflite', 'wb') as f:
    f.write(tflite_model)
print("\nModel saved as 'advanced_price_predictor.tflite'")


interpreter = tf.lite.Interpreter(model_path='advanced_price_predictor.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


test_input = np.hstack((encoder.transform([['plastic']]), [[0.7]])).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
price = interpreter.get_tensor(output_details[0]['index'])[0][0]
print(f"TFLite predicted price for 'plastic' (quality=0.7): ₹{price:.2f}")