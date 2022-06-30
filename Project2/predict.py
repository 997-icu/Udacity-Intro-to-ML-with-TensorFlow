import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import time
import warnings
import argparse
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(1)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


parser = argparse.ArgumentParser(description='Image Classifier')
parser.add_argument(action="store", dest="path",help = "Path to the Image",type = str)
parser.add_argument(action="store", dest="model",help = "The Saved Model",type = str)
parser.add_argument('--top_k', action="store", default=None,type = int, help = "Top K values")
parser.add_argument('--category_names', action="store", default=None,type = str, help = "The Name Map File")


args = parser.parse_args()
img_path = args.path
model_name = args.model
top_k = args.top_k
category_names = args.category_names
#print(img_path,model_name,top_k,category_names)

image_size = 224

def process_image(img):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (image_size, image_size))
    img /= 255
    
    img = img.numpy()
    return img

def predict(image_path,model,top_k):
    
    im = Image.open(image_path)
    im = np.asarray(im)
    
    im = process_image(im)
    im = np.expand_dims(im,axis = 0)
    probs_all = model.predict(im)
    if top_k != None:
        probs = probs_all[0][np.argsort(probs_all[0],axis = 0)][-top_k:][::-1]
        classes = np.array(range(1,103))[np.argsort(probs_all[0],axis = 0)][-top_k:][::-1]
    
        return list(probs),list(classes)
    else:
        probs = probs_all[0][np.argsort(probs_all[0],axis = 0)][::-1]
        classes = np.array(range(1,103))[np.argsort(probs_all[0],axis = 0)][::-1]
    
        return list(probs),list(classes)
        




reloaded_keras_model = tf.keras.models.load_model((model_name),custom_objects={'KerasLayer':hub.KerasLayer},compile = False)
probs, classes = predict(img_path,reloaded_keras_model,top_k)

if category_names != None:
    with open(category_names, 'r') as f:
        class_names = json.load(f)
        
        names = [class_names[str(i)] for i in classes]
        print("\n")
        for i,j in zip(names,probs):
            print(i,":",j)
else:
    print("\n")
    for i,j in zip(classes,probs):
       print(i,":",j)
        











