import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import streamlit as st

uploaded_img = st.file_uploader("Upload Your X-Ray Here")
test_image = image.load_img('uploaded_img', target_size = (600, 600))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
model = keras.models.load_model('https://www.webdesign-hosting.eu/ml4b/saved_xray_classifier.h5')
result = model.predict(test_image)
#training_set.class_indices
st.write(print(result))
if result[0][0] >= 0.5:
    prediction = 'PNEUMONIA'
else:
    prediction = 'NORMAL'
st.write(print(prediction))
