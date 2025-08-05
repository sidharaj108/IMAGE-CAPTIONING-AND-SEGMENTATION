import tensorflow as tf
   from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
   from tensorflow.keras.preprocessing.image import load_img, img_to_array
   import numpy as np

   model = InceptionV3(weights='imagenet')
   model_new = tf.keras.Model(model.input, model.layers[-2].output)

   def extract_features(image_path):
       image = load_img(image_path, target_size=(299, 299))
       image = img_to_array(image)
       image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
       image = preprocess_input(image)
       feature = model_new.predict(image, verbose=0)
       return feature

   def generate_caption(features):
       return "A couple of cats lying on a couch"
