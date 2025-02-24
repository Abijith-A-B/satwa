import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import joblib
import os


def load_hotspot_data(file_path= '/home/abijith/trashnet_data/dataset-resized/waste_uploads.csv'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    df = pd.read_csv(file_path)
    print(f"Loaded data from {file_path}:")
    print(df.head())
    return df


def prepare_data(df):
    feature_cols = ['plastic', 'metal', 'paper', 'cardboard', 'trash', 'glass']
    X = df[feature_cols]
    y = (df["hotspot_prob"] > 0.5).astype(int) if "hotspot_prob" in df.columns else (df["upload_count"] > df["upload_count"].median()).astype(int)
    print("\nFeatures:")
    print(X.head())
    print("\nTarget (is_hotspot):")
    print(y.head())
    return X, y


def train_rf_model(X, y, model_file="waste_hotspot_model.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nRandom Forest Accuracy: {accuracy:.2f}")
    joblib.dump(rf_model, model_file)
    print(f"Model saved as {model_file}")
    return rf_model


def create_tf_model(rf_model, X, input_shape):
   
    rf_probs = rf_model.predict_proba(X)[:, 1] 
    
    
    inputs = tf.keras.Input(shape=(input_shape[1],)) 
    x = tf.keras.layers.Dense(16, activation='relu')(inputs)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)  
    tf_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    
    tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    y_binary = (rf_probs > 0.5).astype(int)
    tf_model.fit(X, y_binary, epochs=10, batch_size=32, verbose=1)
    
    return tf_model


def save_as_tflite(tf_model, input_shape, output_file="waste_hotspot_model.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    tflite_model = converter.convert()
    with open(output_file, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {output_file}")


def train_and_convert_to_tflite(input_file="waste_hotspots.csv"):
    print("Starting training and conversion process...")
    df = load_hotspot_data(input_file)
    X, y = prepare_data(df)
    rf_model = train_rf_model(X, y)
    input_shape = (1, X.shape[1])  # e.g., (1, 6)
    tf_model = create_tf_model(rf_model, X, input_shape)
    save_as_tflite(tf_model, input_shape)


if __name__ == "__main__":
    train_and_convert_to_tflite()