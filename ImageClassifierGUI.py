import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tkinter import messagebox
import numpy as np
import os

class ImageClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier GUI")

        self.model = None
        self.train_data = None
        self.test_data = None

        # Load Data Section
        self.load_data_label = tk.Label(root, text="Load Data:")
        self.load_data_label.grid(row=0, column=0, pady=5)

        self.load_data_button = tk.Button(root, text="Load Data", command=self.load_data)
        self.load_data_button.grid(row=0, column=1, pady=5)

        # Train Model Section
        self.train_model_label = tk.Label(root, text="Train Model:")
        self.train_model_label.grid(row=1, column=0, pady=5)

        self.train_model_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_model_button.grid(row=1, column=1, pady=5)

        # Save Model Section
        self.save_model_label = tk.Label(root, text="Save Model:")
        self.save_model_label.grid(row=2, column=0, pady=5)

        self.save_model_button = tk.Button(root, text="Save Model", command=self.save_model)
        self.save_model_button.grid(row=2, column=1, pady=5)

        # Predict Section
        self.predict_label = tk.Label(root, text="Predict:")
        self.predict_label.grid(row=3, column=0, pady=5)

        self.load_model_button = tk.Button(root, text="Load Model", command=self.load_model_for_prediction)
        self.load_model_button.grid(row=3, column=1, pady=5)

        self.predict_button = tk.Button(root, text="Predict", command=lambda: [self.load_image(), self.predict()])
        self.predict_button.grid(row=3, column=2, pady=5)

        self.output_label = tk.Label(root, text="")
        self.output_label.grid(row=4, column=0, columnspan=3, pady=10)

        # Create an image label to display the loaded image
        self.image_label = tk.Label(root)
        self.image_label.grid(row=5, column=0, columnspan=3, pady=10)
        self.image_path = None

    def load_image(self):
        self.image_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if self.image_path:
            self.display_image(self.image_path)
            #self.image_path = image_path  # Set the image path
            messagebox.showinfo("Success", "Image loaded successfully!")
        else:
            messagebox.showwarning("Warning", "No image selected.")

    def display_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((300, 300))  # Adjust the size as needed
        photo = ImageTk.PhotoImage(image)

        # Assuming you have a Label widget to display the image
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def load_data(self):
        folder_path = filedialog.askdirectory(title="Select Data Folder")
        print("Selected Folder:", folder_path)  # Add this line for debugging
        if folder_path:
            train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
            test_datagen = ImageDataGenerator(rescale=1./255)

            train_data = train_datagen.flow_from_directory(folder_path,
                                                       target_size=(64, 64),
                                                       batch_size=32,
                                                       class_mode='binary',
                                                       subset='training')

            test_data = test_datagen.flow_from_directory(folder_path,
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='binary',
                                                     subset='validation')
            
            return train_data, test_data

            return None
            
            self.train_data = train_data
            self.test_data = test_data

            self.output_label.config(text="Data loaded successfully.")

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self):
        if self.train_data is None or self.test_data is None:
            self.output_label.config(text="Error: Load data first.")
            return

        self.model = self.create_model()

        history = self.model.fit_generator(self.train_data,
                                           steps_per_epoch=len(self.train_data),
                                           epochs=10,
                                           validation_data=self.test_data,
                                           validation_steps=len(self.test_data))

        # Plot training curves (you may customize this based on your needs)
        plt.plot(history.history['accuracy'], label='Train')
        if 'val_accuracy' in history.history:
          plt.plot(history.history['val_accuracy'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.show()

        self.output_label.config(text="Model trained successfully.")

    def save_model(self):
        if self.model is None:
            self.output_label.config(text="Error: Train the model first.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("Model files", "*.h5")])
        if file_path:
            self.model.save(file_path)
            self.output_label.config(text="Model saved successfully.")

    #def load_model(self):
       # file_path = filedialog.askopenfilename(title="Select a Model File", filetypes=[("Model files", "*.h5")])
      #  if file_path:
       #     self.model = load_model(file_path)
       #     self.output_label.config(text="Model loaded successfully.")
      
    def load_model_for_prediction(self):
      model_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("Keras Models", "*.h5")])
      if model_path:
        self.model = load_model(model_path)
        messagebox.showinfo("Success", "Model loaded successfully!")
      else:
        messagebox.showwarning("Warning", "No model selected.")

    def predict(self):
        #self.load_model_for_prediction()
        if self.model is None:
            self.output_label.config(text="Error: Load or train a model first.")
            return

        if self.image_path is None:
            self.output_label.config(text="Error: No image selected.")
            return

        img = image.load_img(self.image_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = self.model.predict(img_array)
        predicted_label = "Class 1" if prediction[0][0] > 0.5 else "Class 0"
        self.output_label.config(text=f"Predicted Label: {predicted_label}")

def main():
    root = tk.Tk()
    app = ImageClassifierGUI(root)

# Load data and set it to the instance variables
    app.train_data, app.test_data = app.load_data()
    root.mainloop()

if __name__ == "__main__":
    main()
