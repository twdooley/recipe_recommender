import streamlit as st
import keras
import tensorflow as tf 
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image as imgx
import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import pickle
from numpy.linalg import norm
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


df = pd.read_csv('w_urls.csv')

st.title("I'll Bake What She's Having!")
st.markdown("### Scan or upload an image below by clicking the browse files button. You'll then be able to take a photo if on a phone.")


feature_list = pickle.load(open('data/featuresRESNET_mymodel.pickle', 'rb'))
filenames = pickle.load(open('data/filenames.pickle', 'rb'))
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Credit to Koul, Ganju, Kassam Ch. 4 'Practical Deep Learning for Cloud, Mobile, and Edge
# For a very thorough and helpful tutorial on reverse image search. 
# The brute KNN method I follow to build this model owes much to their help. 

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

st.write("")

sty_choose = st.sidebar.radio('Optional: Pick a cooking style that you would like.', ('None/Default','Baked Goods, Sweets', 'Cocktails', 'Mediterranean', 'Onion and Garlic', 'Shallots, Chiles, Garlic', 'Asian Inspired', 'Wholesome, Buttery, Herbs', 'Spicy', 'French-ish'))
# for debug, print initialized style
style = None
#st.write(style)
if sty_choose == 'Baked Goods, Sweets':
    style = 'baked'
elif sty_choose ==  'Cocktails':
    style = 'cocktails'
elif sty_choose == 'Mediterranean':
    style = 'evo_greek_tomatoes'
elif sty_choose == 'Onion and Garlic':
    style = 'onions_garlic'
elif sty_choose == 'Shallots, Chiles, Garlic':
    style = 'gar_chick_shall_chile'
elif sty_choose == 'Asian Inspired':
    style = 'asian'
elif sty_choose == 'Wholesome, Buttery, Herbs':
    style = 'chicken_butter_herbs'
elif sty_choose == 'Spicy':
    style = 'spicy'
elif sty_choose == 'French-ish':
    style = 'dijon_parsley_snp'
elif sty_choose == 'None/Default':
    style = None
# for debug, show style
#st.write(style)

veg = st.sidebar.checkbox("Vegetarian")
num_opts = st.sidebar.selectbox('How many options would you like?', (10,11,12,13,14,15,16,17,18,19,20))







uploaded_file = st.file_uploader("Choose an image...")
if uploaded_file is not None:
    st.spinner("Thinking...")
    image = Image.open(uploaded_file)
    rsized = image.resize((224,224))
    distances, indices = get_knn(rsized)
    break_line = '<hr style="border:2px solid gray"> </hr>'
    st.markdown(break_line, unsafe_allow_html = True)
    st.markdown("## Your Image: ")
    st.image(image, use_column_width=True)
    st.write("") 
    listed = list(indices[0])
    for num in range(num_opts):
        if veg == True:
            title = df[df.file == filenames[listed[num]].replace('/Users/timothydooley/Documents/ds/metis/repos/ba_scrape/ba_images/', '').replace('.jpg', '')].name.to_string(index = False).strip()
            get_url = df[df.file == filenames[listed[num]].replace('/Users/timothydooley/Documents/ds/metis/repos/ba_scrape/ba_images/', '').replace('.jpg', '')].url.to_string(index = False).strip()
            meat = df[df.file == filenames[listed[num]].replace('/Users/timothydooley/Documents/ds/metis/repos/ba_scrape/ba_images/', '').replace('.jpg', '')].meat.to_string(index = False).strip()
            label_style = df[df.file == filenames[listed[num]].replace('/Users/timothydooley/Documents/ds/metis/repos/ba_scrape/ba_images/', '').replace('.jpg', '')].nmf_label.to_string(index = False).strip()
            if meat == '0': 
                if style == None:
                    final_url = f"https://www.bonappetit.com{get_url}"
                    concat_url = f"[{title}]({final_url})"
                    break_line = '<hr style="border:2px solid gray"> </hr>'
                    st.markdown(break_line, unsafe_allow_html = True)
                    st.markdown(f"## {concat_url}", unsafe_allow_html=True)
                    #st.write(label_style)
                    st.write(final_url)
                    st.image(filenames[listed[num]], use_column_width = True)
                elif style == label_style:
                    final_url = f"https://www.bonappetit.com{get_url}"
                    concat_url = f"[{title}]({final_url})"
                    break_line = '<hr style="border:2px solid gray"> </hr>'
                    st.markdown(break_line, unsafe_allow_html = True)
                    st.markdown(f"## {concat_url}", unsafe_allow_html=True)
                    #st.write(label_style)
                    st.write(final_url)
                    st.image(filenames[listed[num]], use_column_width = True)
        elif veg == False:
            title = df[df.file == filenames[listed[num]].replace('/Users/timothydooley/Documents/ds/metis/repos/ba_scrape/ba_images/', '').replace('.jpg', '')].name.to_string(index = False).strip()
            get_url = df[df.file == filenames[listed[num]].replace('/Users/timothydooley/Documents/ds/metis/repos/ba_scrape/ba_images/', '').replace('.jpg', '')].url.to_string(index = False).strip()
            label_style = df[df.file == filenames[listed[num]].replace('/Users/timothydooley/Documents/ds/metis/repos/ba_scrape/ba_images/', '').replace('.jpg', '')].nmf_label.to_string(index = False).strip()
            if style == None:
                final_url = f"https://www.bonappetit.com{get_url}"
                concat_url = (f"[{title}]({final_url})")
                break_line = '<hr style="border:2px solid gray"> </hr>'
                st.markdown(break_line, unsafe_allow_html = True)
                st.markdown(f"## {concat_url}", unsafe_allow_html=True)
                #st.write(label_style)
                st.write(final_url)
                st.image(filenames[listed[num]], use_column_width = True)
            elif style == label_style:
                final_url = f"https://www.bonappetit.com{get_url}"
                concat_url = f"[{title}]({final_url})"
                break_line = '<hr style="border:2px solid gray"> </hr>'
                st.markdown(break_line, unsafe_allow_html = True)
                st.markdown(f"## {concat_url}", unsafe_allow_html=True)
                #st.write(label_style)
                st.write(final_url)
                st.image(filenames[listed[num]], use_column_width = True)
        #st.info(f"Your closest match is {df[df.file == filenames[listed[0]].replace('/Users/timothydooley/Documents/ds/metis/repos/ba_scrape/ba_images/', '').replace('.jpg', '')].name.to_string(index = False).strip()}")

