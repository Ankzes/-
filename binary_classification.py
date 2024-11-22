import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import splitfolders
import os
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from PIL import Image
import random
import warnings
warnings.filterwarnings('ignore')

bike_path = './Car-Bike-Dataset/Bike'
car_path = './Car-Bike-Dataset/Car'

bike_files_len = len(os.listdir(bike_path))
car_files_len = len(os.listdir(car_path))
print(f'Total Number of Images in Bike folder are: {bike_files_len}')
print(f'Total Number of Images in Car folder are: {car_files_len}')

all_file = [f for f in os.listdir(bike_path) if os.path.isfile(os.path.join(bike_path, f))]
print(len(all_file))
random.sample(all_file, 5)

# Converting into function to load 5 sample images 
def load_sample_images(folder_path, sample_size = 5):
    # Getting the total files so we can sample it
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # Sampling the 5 files from total files
    sample_files = random.sample(all_files, sample_size)
    
    images = []
    for file in sample_files:
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path)
        images.append(img)
    
    return images

bike_sample_img = load_sample_images(bike_path)
car_sample_img = load_sample_images(car_path)

# Plotting the sample images from dataset
all_img_sample = [bike_sample_img, car_sample_img]
fig, axes = plt.subplots(2, len(bike_sample_img), figsize=(15,6))
title = ['Bike', 'Car']
for row, category in enumerate(all_img_sample):
    for i, img in enumerate(category):
        axes[row, i].imshow(img)
        axes[row, i].set_title(title[row])
        axes[row, i].axis('off')
plt.suptitle('Sample Images of Bikes,and Cars', fontweight='bold')
plt.tight_layout()
plt.show()

# Spliting the dataset into train and test
input_folder = './Car-Bike-Dataset'
output_folder = './output'

splitfolders.ratio(input_folder, output = output_folder, seed = 1337, ratio = (0.8, 0.2), group_prefix=None)

print("Data has been splited into Train and Test successfully!")

# Showing number of Car file in train and test folder (Demo code or kind a implementation technique so if its work will convert into a function)
train_car_path = './output/train/Car'
test_car_path = './output/val/Car'

train_car_files_len = len(os.listdir(train_car_path))
test_car_files_len = len(os.listdir(test_car_path))

print(f'Total Number of Car Images in Train folder are: {train_car_files_len}')
print(f'Total Number of Car Images in Test  folder are: {test_car_files_len}')

# Now creating a function which will help to count the numbers of file in all folders which are in train and test
def count_files_in_folders(base_path):
    folder_data = []
    if not os.path.exists(base_path):
        print(f"The path {base_path} does not exist.")
        return folder_data

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            num_files = 0
            for item in os.listdir(folder_path):
                if os.path.isfile(os.path.join(folder_path, item)):
                    num_files += 1
            folder_data.append((folder, num_files))

    return folder_data

train_path = './output/train'
test_path =  './output/val'

train_data_count = count_files_in_folders(train_path)
test_data_count = count_files_in_folders(test_path)

# Numbers of train test dataset for each category
train_df = pd.DataFrame(train_data_count, columns=['Folder', 'NumFiles'])
test_df = pd.DataFrame(test_data_count, columns=['Folder', 'NumFiles'])

# Adding a column
train_df['Type'] = 'Train'
test_df['Type'] = 'Test'

# Combine the data
combined_df = pd.concat([train_df, test_df])

combined_df

# Applying Data Generator 
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range=15)

test_datagen =  ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    directory = train_path,
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'binary'
)

validation_generator = test_datagen.flow_from_directory(
    directory= test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

train_generator.class_indices

def plitting_graph(history):
    # plitting line graph to see training and testing loss, accuracy curve
    fig, axes = plt.subplots(1,2 , figsize =(14, 5))

    sns.lineplot(ax = axes[0], data = history['loss'], label = 'training loss' )
    sns.lineplot(ax= axes[0], data = history['val_loss'], label = 'testing loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    sns.lineplot(ax = axes[1], data = history['accuracy'], label = 'training accuracy')
    sns.lineplot(ax = axes[1], data = history['val_accuracy'], label = 'testing accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

model_path = 'model.h5'

# Проверка, существует ли модель
if os.path.exists(model_path):
    # Загрузка сохраненной модели
    model = load_model('model.h5')
    print("Model has been loaded successfully.")

    # Загрузка сохраненной истории обучения
    with open('training_history.pkl', 'rb') as file:
        history = pickle.load(file)
    
    plitting_graph(history)
else:
    # Creating Model Architecture
    model = Sequential()

    model.add(Conv2D(56, kernel_size = (3,3), activation = 'relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(28, kernel_size = (3,3), activation = 'relu'))
    model.add(MaxPooling2D((3,3)))

    model.add(Conv2D(16, kernel_size = (3,3), activation = 'relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())

    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(14, activation = 'relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation = 'sigmoid'))

    model.summary()

    # Compiling the model wiht loss binary cross entropy and optimizer Adam
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    # Training the model and storing the information into history vairable
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=8
    )

    # Saving training history
    with open('training_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    # Saving trained model
    model.save('model_8.hs')

    plitting_graph(history)

def get_true_and_predicted_labels(generator, model, threshold=0.5):
    true_labels = []
    pred_labels = []
    for images, labels in generator:
        preds = model.predict(images)

        # Convert probabilities to binary class
        pred_labels.extend((preds > threshold).astype(int))

        # Convert labels to binary class
        if len(labels.shape) > 1:
            true_labels.extend((labels > threshold).astype(int))
        else:
            true_labels.extend(labels)
        if len(true_labels) >= generator.samples:
            break

    return np.array(true_labels), np.array(pred_labels)

# Get true and predicted labels
true_labels, pred_labels = get_true_and_predicted_labels(validation_generator, model)

# calculating the metrics
conf_matrix = confusion_matrix(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average='binary')
recall = recall_score(true_labels, pred_labels, average='binary')
f1 = f1_score(true_labels, pred_labels, average='binary')

print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# plotting confusion metrics
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Define class names
class_names = list(validation_generator.class_indices.keys())

# Plot confusion matrix
plot_confusion_matrix(conf_matrix, class_names)

# Preprocessing the image for prediction 
def process_image(img, target_size=(224, 224)):
    if isinstance(img, str):
        img = image.load_img(img, target_size=target_size)
    else:
        img = img.resize(target_size)
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

car_test_path = './output/val/Car'
bike_test_path = './output/val/Bike'

img_car = load_sample_images(car_test_path)
img_bike = load_sample_images(bike_test_path)
validation_generator.class_indices

# plotting the prediction with actual and predicted title
test_img = [img_car, img_bike]
fig, axes = plt.subplots(2, len(img_car), figsize=(20,12))
title = ['Actual: Car', 'Actual: Bike']
for row, category in enumerate(test_img):
  for i, img in enumerate(category):
    processed_img = process_image(img)
    pred = model.predict(processed_img)
    pred_title = ' | Predicted: Bike' if pred[0][0] < 0.5 else ' | Predicted: Car'
    axes[row, i].imshow(img)
    axes[row, i].set_title(title[row] + pred_title)
    axes[row, i].axis('off')

plt.suptitle('Predcition from Test Data', fontweight='bold')
plt.tight_layout()
plt.show()