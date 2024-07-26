# Hand-Gesture-Recognition

This code implements an American Sign Language (ASL) gesture recognition system using computer vision and deep learning. 
It starts by loading and preprocessing image data of hand gestures, then trains a convolutional neural network (CNN) to classify these gestures. 
The trained model is saved and later used for real-time prediction. The script uses OpenCV for image processing and webcam input, TensorFlow/Keras for building and training the neural network, and MediaPipe for hand landmark detection. In the real-time prediction phase, it captures video from the webcam of your machine, detects hand landmarks, extracts the hand region, and passes it through the trained model to predict the ASL gesture. The predicted gesture and confidence score are displayed on the video feed.

OUTPUT:


![WhatsApp Image 2024-07-26 at 14 28 31_6541d100](https://github.com/user-attachments/assets/60636008-439e-445b-8a5b-b28a3470d3b9)
![WhatsApp Image 2024-07-26 at 14 31 26_4d28ec55](https://github.com/user-attachments/assets/b1a51ced-87e9-40b8-96d8-f9189d6c5653)

Dataset:
https://www.kaggle.com/datasets/muhammadkhalid/sign-language-for-alphabets
