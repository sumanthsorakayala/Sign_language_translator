# Sign_language_translator
🧠 Sign Language to Speech Translator
This project is a real-time translator that converts ASL (American Sign Language) hand gestures into spoken words using computer vision and deep learning. It’s designed to bridge the communication gap between deaf or hard-of-hearing individuals and those unfamiliar with sign language.

📸 Features
🖐️ Detects ASL hand signs using MediaPipe

🤖 Classifies gestures using a trained TensorFlow Keras model

🔊 Converts recognized text to speech using pyttsx3

💻 Real-time webcam interface with prediction overlay

📝 Easily extensible: add new signs or retrain the model

📂 Project Structure
bash
Copy
Edit
basic_signlanguage_translator/
├── models/
│   ├── asl_model.h5           # Trained Keras model
│   └── class_names.npy        # List of class names
├── src/
│   ├── asl_gui.py             # Optional GUI interface
│   ├── asl_recognition.py     # Gesture recognition logic
│   ├── data_collection.py     # Tool to collect labeled data
│   ├── train_model.py         # Model training script
│   └── utils.py               # Helper functions
├── labels.txt                 # Human-readable class labels
├── main.py                    # Main real-time translation app
└── README.md
🚀 How to Run
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Ensure the following files exist:

models/asl_model.h5

models/class_names.npy

labels.txt

Run the application:

bash
Copy
Edit
python main.py
Show an ASL gesture in front of the webcam and hear it spoken aloud 🎤

🎯 Supported Gestures
The model currently supports the following signs (from labels.txt):

mathematica
Copy
Edit
A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
You can modify this list and retrain the model using your own gestures!

🛠️ Model Training (Optional)
If you want to collect your own dataset and retrain:

bash
Copy
Edit
python src/data_collection.py
python src/train_model.py
Make sure you update class_names.npy and labels.txt accordingly.


🔊 Text to Speech
This project uses pyttsx3 to convert recognized text to audio. You can customize the voice, rate, and volume easily inside the code.

💡 Future Improvements
Add a GUI using Tkinter

Include Indian Sign Language (ISL) support

Add gesture-based sentence construction

Optimize model for mobile deployment

🙌 Acknowledgments
MediaPipe

TensorFlow

ASL Dataset contributors

📄 License
MIT License — Feel free to use and modify for educational or research purposes.
