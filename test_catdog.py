from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np

# Modelo de Keras cargado
model = load_model('cat_dog_classifier.h5')

# Código para realizar una predicción individual
def predict_image(model, image_path):
    img = load_img(image_path, target_size=(128, 128))
    img = np.array(img) / 255.0
    img = img.reshape(1, 128, 128, 3)
    pred = model.predict(img)
    return 'Dog' if pred[0] > 0.5 else 'Cat'

# Llamado de función de predicción
image_path = "./dogs-vs-cats/test1/5.jpg"
print(predict_image(model, image_path))