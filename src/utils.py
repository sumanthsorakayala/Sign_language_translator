import os
import numpy as np

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data(data_dir):
    X = []
    y = []
    class_names = os.listdir(data_dir)
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.npy'):
                file_path = os.path.join(class_dir, file_name)
                landmarks = np.load(file_path)
                X.append(landmarks)
                y.append(label)
    
    return np.array(X), np.array(y), class_names