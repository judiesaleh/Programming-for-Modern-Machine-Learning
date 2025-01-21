import os
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array

# Pfad zum geladenen Modell
model_path = './Model/my_license_plate_recognition_model.keras'

# Modell laden
model = tf.keras.models.load_model(model_path)

# Bild vorverarbeiten (Größe anpassen und Normalisierung)
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))  # Größe der Bilder anpassen
    image = img_to_array(image)
    image = image / 255.0  # Normalisieren
    return np.expand_dims(image, axis=0)  # Model erwartet ein Batch

# Bildpfad
image_path = r'C:\Users\judie\Desktop\GitHub\Programming-for-Modern-Machine-Learning\license-plates-dataset\Serbia\files\domain2\0001882_labelled.jpg'  # Hier das lokale Bildpfad angeben

# Bild vorverarbeiten
image = preprocess_image(image_path)

# Model vorschlagen
prediction = model.predict(image)

# Label aus der Vorhersage
predicted_label = np.argmax(prediction)
print(f"Vorgeschlagene Lizenznummer: {predicted_label}")
