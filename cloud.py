import io
import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow import keras

class_name = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

os.environ["TFHUB_CACHE_DIR"] = "/tmp/model"
hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")
model = keras.models.load_model('garbage_classifier.h5', custom_objects={'KerasLayer':hub.KerasLayer})

# process the image before performing inference
def process_img(data):
    read_data = data
    loaded_img = np.asarray(read_data)
    loaded_img = loaded_img / 255.0
    loaded_img = np.expand_dims(loaded_img, 0)
    loaded_img = tf.image.resize(loaded_img, [224, 224])
    return loaded_img

# run the predictions
def run_inference(img_path):
    inference_input = img_path
    predictions = model.predict(inference_input)
    result = np.argmax(predictions)
    return result

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def predictions():
    if request.method == 'POST':

        file = request.files.get('file')

        if file is None or file.filename == "":
            return jsonify({"error": "no file uploaded"})

        try:
            load_img = file.read()
            conv_img = Image.open(io.BytesIO(load_img)).convert('RGB')

            input_image = process_img(conv_img)
            result = run_inference(input_image)

            predictions_result = ""

            if result == 0:
                predictions_result = "Garbage categorized as: {} \n" \
                                     "Dry and flatten it, stack it together and tie properly".format(class_name[result])
                # print("Garbage categorized as: {} clean thoroughly and sun-dry".format(class_name[result]))
            elif result == 1:
                predictions_result = "Garbage categorized as: {} \n" \
                                     "Clean and wash with soap, put it in secure box \n " \
                                     "If broken put in separated box".format(class_name[result])
                # print("Garbage categorized as: {}, clean thoroughly and sun-dry".format(class_name[result]))
            elif result == 2:
                predictions_result = "Garbage categorized as: {} \n" \
                                     "Clean and dry \n" \
                                     "DO NOT INCLUDE HAZARDOUS MATERIAL SUCH AS BATTERIES!".format(class_name[result])
                # print("Garbage categorized as: {}, clean thoroughly and sun-dry".format(class_name[result]))
            elif result == 3:
                predictions_result = "Garbage categorized as: {} \n" \
                                     "Dry and flatten it, stack it together and tie properly".format(class_name[result])
                # print("Garbage categorized as: {}, clean thoroughly and sun-dry".format(class_name[result]))
            elif result == 4:
                predictions_result = "Garbage categorized as: {} \n" \
                                     "Clean with soap and dry \n" \
                                     "Place in one box or container".format(class_name[result])
                # print("Garbage categorized as: {}, clean thoroughly and sun-dry".format(class_name[result]))
            elif result == 5:
                predictions_result = "Garbage categorized as: {} \n" \
                                     "Place in plastic bag and tie properly".format(class_name[result])
                # print("Garbage categorized as: {}, clean thoroughly and sun-dry".format(class_name[result]))

            return jsonify(predictions_result)

        except Exception as e:
            return jsonify({"error": str(e)})
    return "IT'S UP AND READY TO GO!"


if __name__ == "__main__":
    app.run(debug=True)
