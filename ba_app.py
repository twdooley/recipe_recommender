import streamlit as st
import keras
import tensorflow as tf 
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image as imgx
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import pickle
from numpy.linalg import norm
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


st.title("I'll Bake What She's Having!")
st.markdown("### Scan or upload an image below by clicking the browse files button. You'll then be able to take a photo if on a phone.")


feature_list = pickle.load(open('data/featuresRESNET_mymodel.pickle', 'rb'))
filenames = pickle.load(open('data/filenames.pickle', 'rb'))
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


def extract_features(imgin, model):
    input_shape = (224, 224, 3)
    #img = imgx.load_img(imgin, target_size=(input_shape[0], input_shape[1]))
    img_array = imgx.img_to_array(imgin)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features

def get_knn(img_array):
    neighbors = NearestNeighbors(n_neighbors=20, algorithm='brute', metric='euclidean').fit(feature_list)
    distances, indices = neighbors.kneighbors([extract_features(img_array,model)])
    return distances, indices


uploaded_file = st.file_uploader("Choose an image...")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    #t_im = cv2.imread('/Users/timothydooley/Documents/ds/metis/repos/ba_scrape/ba_images/“tater_tots”_with_spicy_mayonnaise.jpg')
    #rsized = cv2.resize(src =t_im, dsize=(224,224))
    rsized = image.resize((224,224))
    distances, indices = get_knn(rsized)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("") 
    listed = list(indices[0])
    for num in range(10):
        st.image(filenames[listed[num]], use_column_width = True)


    st.write()