import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


model_path = '/home/abijith/Documents/waste_model.tflite'
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


input_shape = input_details[0]['shape']
print(f"Model Input Shape: {input_shape}")  


if len(input_shape) != 4 or input_shape[0] != 1:
    raise ValueError("Model input shape should be [1, height, width, channels]")

height, width = input_shape[1], input_shape[2]


class_labels = ["plastic", "paper", "bio-waste"]


cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    
    img = cv2.resize(frame, (width, height))  
    img = img.astype(np.float32) / 255.0  
    img = np.expand_dims(img, axis=0) 

    
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    
    if output_data.ndim == 2: 
        output_data = output_data[0]

    
    predicted_class = np.argmax(output_data)
    confidence = np.max(output_data) * 100

    
    label = f"{class_labels[predicted_class]} ({confidence:.2f}%)"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Waste Detection", frame)

   
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
