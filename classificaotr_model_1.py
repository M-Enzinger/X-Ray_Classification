import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow import keras

test_image = image.load_img('chest-pneumoia.jpeg', target_size = (512, 512))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
model = keras.models.load_model('models/')
result = model.predict(test_image)
#training_set.class_indices
print(result)
if result[0][0] >= 0.5:
    prediction = 'PNEUMONIA'
else:
    prediction = 'NORMAL'
print(prediction)