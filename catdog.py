# Imports
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Plotting library
import warnings  # To handle warnings
import os  # Operating system interfaces
import random  # Generate random numbers
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator  # Image processing
import PIL  # Python Imaging Library
import seaborn as sns  # Statistical data visualization
from sklearn.model_selection import train_test_split  # Split arrays or matrices into random train and test subsets
from tensorflow.keras.models import Sequential  # Sequential model API
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization  # Neural network layers


# Ignoring warnings
warnings.filterwarnings('ignore')

# Defining image dimensions and color channels
Image_Width = 128
Image_Height = 128
Image_Size = (Image_Width, Image_Height)
Image_Channels = 3

# Defining paths and labels
input_path = []
label = []

# Looping through each class directory in 'PetImages'
for class_name in os.listdir("PetImages"):
    # Looping through each image in class directory
    for path in os.listdir(f"PetImages/{class_name}"):
        # Checking for '.jpg' extension
        if '.jpg' in path:
            # Appending 0 for 'Cat' and 1 for 'Dog' to the label list
            label.append(0 if class_name == 'Cat' else 1)
            # Appending the image path to the input_path list
            input_path.append(os.path.join("PetImages", class_name, path))

# Creating a DataFrame with image paths and labels
df = pd.DataFrame({'images': input_path, 'label': label})
# Shuffling the DataFrame and resetting index
df = df.sample(frac=1).reset_index(drop=True)

# Removing unwanted files from the DataFrame
df = df[~df['images'].str.contains('Thumbs.db')]
df = df[~df['images'].str.contains('666.jpg')]
df = df[~df['images'].str.contains('11702.jpg')]

# Function to display images
def display_images(df, label, title):
    plt.figure(figsize=(25, 25))
    # Filtering images by label
    temp = df[df['label'] == label]['images']
    # Random starting index for displaying images
    start = random.randint(0, len(temp) - 25)
    # Selecting 25 images from the starting index
    files = temp[start:start + 25]

    # Displaying each image
    for index, file in enumerate(files):
        plt.subplot(5, 5, index + 1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.show()

# Displaying images of Dogs and Cats
display_images(df, 1, 'Dogs')
display_images(df, 0, 'Cats')

# Plotting count graph for labels
sns.countplot(df['label'])
plt.show()

# Converting labels to string type
df['label'] = df['label'].astype('str')

# Splitting data into train and test sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Data generators with data augmentation for training
train_generator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data generator for validation
val_generator = ImageDataGenerator(rescale=1. / 255)

# Creating iterators for training and validation
train_iterator = train_generator.flow_from_dataframe(
    train, x_col='images', y_col='label',
    target_size=(128, 128), batch_size=512, class_mode='binary'
)

val_iterator = val_generator.flow_from_dataframe(
    test, x_col='images', y_col='label',
    target_size=(128, 128), batch_size=512, class_mode='binary'
)

# Building the model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(Image_Width, Image_Height, Image_Channels)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compiling and training the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Printing model summary
model.summary()

# Training the model
history = model.fit(train_iterator, epochs=10, validation_data=val_iterator)

# Function to plot accuracy and loss graphs
def plot_metrics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    # Plotting training and validation accuracy
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Accuracy Graph')
    plt.legend()
    plt.figure()

    # Plotting training and validation loss
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Loss Graph')
    plt.legend()
    plt.show()

# Plotting metrics
plot_metrics(history)

# Saving the model
model.save('cat_dog_classifier2.h5')
