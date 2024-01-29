import os
import subprocess
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tkinter as tk
import time
from PIL import Image


def create_model(input_shape=(64, 64, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary() #Displays the model summary
    
    return model

def train_model(model, train_data, test_data, epochs=10):
    history = model.fit_generator(train_data,
                                  steps_per_epoch=len(train_data),
                                  epochs=epochs,
                                  validation_data=test_data,
                                  validation_steps=len(test_data))
    return model, history

def plot_curves_and_save(history, task_number):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label=f'Task {task_number} Train')
    plt.plot(history.history['val_accuracy'], label=f'Task {task_number} Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Task {task_number} Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label=f'Task {task_number} Train')
    plt.plot(history.history['val_loss'], label=f'Task {task_number} Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Task {task_number} Training and Validation Loss')
    plt.legend()

    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(f'task{task_number}_curves.png')
    plt.show()

def predict_and_plot_images(model, task_number, new_data, task_labels):
    print(f"\nPredictions for Task {task_number}\n{'='*30}")

    # Assuming binary classification, adjust accordingly if needed
    predictions = model.predict(new_data, steps=len(new_data), verbose=1)
    true_labels = new_data.classes
     
    #new_data.reset()  # Reset generator to the beginning

    for i in range(len(predictions)):
        input_data, true_label = next(new_data)
        predicted_label = (predictions[i] if len(predictions.shape) > 1 else predictions[i][0])
        
    # Plot the input image
    plt.imshow(input_data[0])
    plt.title(f'Task {task_number} - True Label: {true_label[0]}, Predicted Label: {predicted_label}')
    plt.show(block=False)
    plt.pause(2)  # Add a delay (in seconds) to give time for the plot window to render

        #print("Input Data:", input_data)
        #print("True Label:", true_label)
        #print("Predicted Label:", predicted_label)
        
    # Print additional information
        
    print("True Label:", true_labels[i])
    print("Predicted Label:", predicted_label)

    predicted_task_label = task_labels[task_number - 1]
    if predicted_label is not None and predicted_label >= 0.3:
            print("Predicted Task:", predicted_task_label)
    elif predicted_label is not None:
            print("Predicted Task: Not Recognized")
    else:
            print("Predicted Task: Not Available")

    print("--------------------")

# Reset generator after processing
    new_data.reset()

def launch_gui():
    print("Launching ImageClassifierGUI.py...")
    subprocess.run(["python", "ImageClassifierGUI.py"])

def main():
    initial_model = create_model()
    task_datasets = [
        ('A&F', 'Animals-and-Flowers'),
        ('C&A', 'Cars-and-Animals'),
        ('F&P', 'Flowers-and-Paintings'),
        ('C&D', 'Cats-and-Dogs'),
        # Add more tasks as needed
    ]

    all_histories = []
    all_models = []
    task_labels = []

    for task_number, (task_name, folder_name) in enumerate(task_datasets, start=1):
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_data = train_datagen.flow_from_directory(f'C:/Users/nithi/Individual Project/Chinmaya-Dimmiti/My Work/{folder_name}/train',
                                                       target_size=(64, 64),
                                                       batch_size=32,
                                                       class_mode='binary')

        test_data = test_datagen.flow_from_directory(f'C:/Users/nithi/Individual Project/Chinmaya-Dimmiti/My Work/{folder_name}/test',
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='binary')
        
        task_labels.append(task_name)

        initial_model, task_history = train_model(initial_model, train_data, test_data)
        all_histories.append(task_history)

        for layer in initial_model.layers:
            layer.trainable = False

        # Create a new model for each task to avoid catastrophic forgetting
        task_model = Sequential()
        task_model.add(initial_model)
        task_model.add(Dense(units=128, activation='relu'))
        task_model.add(Dense(units=1, activation='sigmoid'))
        task_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        task_model, task_history = train_model(task_model, train_data, test_data)

        task_model.save(f'task{task_number}_model.h5')

        print(f"Task {task_number} Training Accuracy:", task_history.history['accuracy'][-1])
        print(f"Task {task_number} Validation Accuracy:", task_history.history['val_accuracy'][-1])

        plot_curves_and_save(task_history, task_number)
        
        all_models.append(task_model)
        all_histories.append(task_history)

    plt.plot(all_histories[0].history['accuracy'], label='Task 1 Train')
    plt.plot(all_histories[0].history['val_accuracy'], label='Task 1 Validation')

    for task_number in range(2, len(task_datasets) + 1):
        task_history = all_histories[task_number - 1]
        plt.plot(task_history.history['accuracy'], label=f'Task {task_number} Train')
        plt.plot(task_history.history['val_accuracy'], label=f'Task {task_number} Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy for All Tasks')
    plt.legend()
    plt.show()


    for task_number, task_model in enumerate(all_models, start=1):
        new_data = test_datagen.flow_from_directory(f'C:/Users/nithi/Individual Project/Chinmaya-Dimmiti/My Work/{task_datasets[task_number-1][1]}/test',
                                                    target_size=(64, 64),
                                                    batch_size=1,  # Set batch size to 1 for plotting individual images
                                                    class_mode='binary',
                                                    shuffle=False)  # Disable shuffling to match predictions with inputs)
        predict_and_plot_images(task_model, task_number, new_data, task_labels)

 # Launch the GUI after training is completed
    launch_gui()

if __name__ == "__main__":
    main()
