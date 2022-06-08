import os

import numpy as np
import tensorflow_hub as hub
# from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# import urllib.request
# import tensorflow.keras.preprocessing

class_name = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

os.environ["TFHUB_CACHE_DIR"] = "/tmp/model"
hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")
model = keras.models.load_model('garbage_classifier.h5', custom_objects={'KerasLayer':hub.KerasLayer})



loaded_img = image.load_img('sampel5.jpg', target_size=(224, 224))
loaded_img = image.img_to_array(loaded_img) / 255
loaded_img = np.expand_dims(loaded_img, 0)

predictions = model.predict(loaded_img)
result = np.argmax(predictions)
if result == 1:
    print("Kategori sampah adalah {}, cuci bersih dan keringkan".format(class_name[result]))
elif result == 2:
    print("Kategori sampah adalah {}, cuci bersih dan keringkan".format(class_name[result]))
elif result == 3:
    print("Kategori sampah adalah {}, cuci bersih dan keringkan".format(class_name[result]))
elif result == 4:
    print("Kategori sampah adalah {}, cuci bersih dan keringkan".format(class_name[result]))
elif result == 5:
    print("Kategori sampah adalah {}, cuci bersih dan keringkan".format(class_name[result]))
elif result == 6:
    print("Kategori sampah adalah {}, cuci bersih dan keringkan".format(class_name[result]))

print(result)
print(class_name[result])

