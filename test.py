import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array

# Funktion, um Bilder und Labels für mehrere Länder zu laden
def load_data(image_dirs, csv_files):
    images = []
    labels = []

    # Iteriere über alle angegebenen Bildverzeichnisse und CSV-Dateien
    for image_dir, csv_file in zip(image_dirs, csv_files):
        # Prüfen, ob das Bildverzeichnis existiert
        if not os.path.exists(image_dir):
            print(f"Das Bildverzeichnis {image_dir} existiert nicht.")
            continue

        # Prüfen, ob die CSV-Datei existiert
        if not os.path.exists(csv_file):
            print(f"Die CSV-Datei {csv_file} existiert nicht.")
            continue

        # Laden der CSV-Datei
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Fehler beim Laden der CSV-Datei: {csv_file}")
            continue

        print(f"Spalten der CSV-Datei: {df.columns}")
        print(f"Erste Zeilen der CSV-Datei:\n{df.head()}")

        # Durch die DataFrame-Zeilen iterieren
        for idx, row in df.iterrows():
            image_path = os.path.join(image_dir, row['file_name'])

            # Prüfen, ob das Bild existiert
            if not os.path.exists(image_path):
                print(f"Fehlendes Bild: {image_path}")
                continue

            # Bild laden und preprocessen
            image = load_img(image_path, target_size=(224, 224))  # Größe der Bilder anpassen
            image = img_to_array(image)
            image = image / 255.0  # Normalisieren

            # Label extrahieren (hier nehmen wir die Lizenzplatte als Beispiel)
            label = row['license_plate.number']  # Beispiel: Die Nummer der Lizenzplatte als Label

            images.append(image)
            labels.append(label)

    # Konvertiere Listen in NumPy Arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Pfade zu den Bildverzeichnissen und den CSV-Dateien
image_dirs = [
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\Brazil\files\domain1',
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\Finland\files\domain2',
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\Estonia\files\domain2',
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\Kazakhstan\files\domain1',
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\Kazakhstan\files\domain2',
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\Lithuania\files\domain2',
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\Serbia\files\domain1',
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\Serbia\files\domain2',
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\UAE\files\domain1',
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\UAE\files\domain2'
]

csv_files = [
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\Brazil\Brazil_domain1_p1_samples.csv',
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\Finland\Finland_domain2_p1_samples.csv',
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\Estonia\Estonia_domain2_p1_samples.csv',
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\Kazakhstan\Kazakhstan_domain1_p1_samples.csv',
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\Kazakhstan\Kazakhstan_domain2_p1_samples.csv',
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\Lithuania\Lithuania_domain2_p1_samples.csv',
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\Serbia\Serbia_domain1_p1_samples.csv',
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\Serbia\Serbia_domain2_p1_samples.csv',
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\UAE\UAE_domain1_p1_samples.csv',
    r'C:\Users\maure\Desktop\Programming-for-Modern-Machine-Learning\license-plates-dataset\UAE\UAE_domain2_p1_samples.csv'
]

# Daten laden
print("Prüfe Dateien in den Bildverzeichnissen...")
images, labels = load_data(image_dirs, csv_files)

# Wenn keine Daten geladen wurden, beenden
if len(images) == 0 or len(labels) == 0:
    print("Es wurden keine Daten geladen. Bitte überprüfe die Pfade, die CSV-Dateien und die Bilder.")
    exit()

# Umwandlung der Labels in numerische IDs
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Aufteilen der Daten in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Definieren des Modells (Beispiel)
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(labels)), activation='softmax')  # Anzahl der Klassen
])

# Modell kompilieren
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modell trainieren
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
