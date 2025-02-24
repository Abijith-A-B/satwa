import cv2
import numpy as np
import tensorflow as tf


classes = ['plastic', 'metal', 'paper', 'cardboard', 'trash', 'glass']


model_path = '/home/abijith/Downloads/waste_classifier2.tflite'   
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Error: Could not open webcam. Check connection or index.")


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Webcam opened successfully. Press 'q' to quit.")

while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    
    
    input_frame = cv2.resize(frame, (224, 224))
    
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    
    input_frame = input_frame / 255.0
    
    input_frame = np.expand_dims(input_frame, axis=0).astype(np.float32)

    
    interpreter.set_tensor(input_details[0]['index'], input_frame)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    
    predicted_idx = np.argmax(output_data)
    predicted_class = classes[predicted_idx]
    confidence = output_data[0][predicted_idx] * 100  

    
    label = f"{predicted_class}: {confidence:.2f}%"
    cv2.putText(
        frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
    )

    
    cv2.imshow('Waste Detector', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")