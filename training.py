# Verbesserter Trainingscode

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from keras._tf_keras.keras.utils import to_categorical

# Label Encoder initialisieren
label_encoder = LabelEncoder()

# Funktion zum Laden und Vorverarbeiten von Bildern und Labels
def load_data(image_dirs, csv_files):
    images = []
    labels = []

    for image_dir, csv_file in zip(image_dirs, csv_files):
        if not os.path.exists(image_dir):
            print(f"Das Bildverzeichnis {image_dir} existiert nicht.")
            continue

        if not os.path.exists(csv_file):
            print(f"Die CSV-Datei {csv_file} existiert nicht.")
            continue

        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Fehler beim Laden der CSV-Datei: {csv_file}: {e}")
            continue

        print(f"CSV-Datei geladen: {csv_file}, Spalten: {df.columns}")

        for idx, row in df.iterrows():
            # Bildpfade
            raw_image_path = os.path.join(image_dir, row.get('file_name', ''))

            if not os.path.exists(raw_image_path):
                print(f"Fehlendes Bild: {raw_image_path}")
                continue

            # Bild laden und vorverarbeiten
            try:
                image = load_img(raw_image_path, target_size=(224, 224))  # Größe anpassen
                image = img_to_array(image) / 255.0  # Normalisierung
            except Exception as e:
                print(f"Fehler beim Verarbeiten des Bildes {raw_image_path}: {e}")
                continue

            # Label extrahieren
            label = row.get('license_plate.number', None)
            if label is None:
                print(f"Kein Label für {raw_image_path} gefunden.")
                continue

            images.append(image)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Daten laden
image_dirs = [
    './license-plates-dataset/Brazil/files/domain1',
    './license-plates-dataset/Finland/files/domain2',
    './license-plates-dataset/Estonia/files/domain2',
    './license-plates-dataset/Kazakhstan/files/domain1',
    './license-plates-dataset/Kazakhstan/files/domain2',
    './license-plates-dataset/Lithuania/files/domain2',
    './license-plates-dataset/Serbia/files/domain1',
    './license-plates-dataset/Serbia/files/domain2',
    './license-plates-dataset/UAE/files/domain1',
    './license-plates-dataset/UAE/files/domain2'
]

csv_files = [
    './license-plates-dataset/Brazil/Brazil_domain1_p1_samples.csv',
    './license-plates-dataset/Finland/Finland_domain2_p1_samples.csv',
    './license-plates-dataset/Estonia/Estonia_domain2_p1_samples.csv',
    './license-plates-dataset/Kazakhstan/Kazakhstan_domain1_p1_samples.csv',
    './license-plates-dataset/Kazakhstan/Kazakhstan_domain2_p1_samples.csv',
    './license-plates-dataset/Lithuania/Lithuania_domain2_p1_samples.csv',
    './license-plates-dataset/Serbia/Serbia_domain1_p1_samples.csv',
    './license-plates-dataset/Serbia/Serbia_domain2_p1_samples.csv',
    './license-plates-dataset/UAE/UAE_domain1_p1_samples.csv',
    './license-plates-dataset/UAE/UAE_domain2_p1_samples.csv'
]

images, labels = load_data(image_dirs, csv_files)

if len(images) == 0 or len(labels) == 0:
    raise ValueError("Keine Trainingsdaten gefunden. Bitte überprüfen Sie die Pfade und Dateien.")

# Labels encodieren
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)  # One-Hot-Encoding für Multi-Klassen-Klassifikation

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Modell definieren
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Modell kompilieren
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

# Modell speichern
model_path = './Model/my_license_plate_recognition_model.keras'
model.save(model_path)

print("Modelltraining abgeschlossen und gespeichert.")