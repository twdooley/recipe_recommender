import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
from io import BytesIO




df = pd.read_csv('ingred.csv')
df2 = pd.read_csv('super.csv')

st.title("Bon Appetit Ingredient Genie")
st.markdown("### Enter ingredients (as descriptive and in depth as you'd like!) and get a recipe recommendation!")

dtm_tf = pickle.load(open('dtm_tf.pickle', 'rb'))
tf_vectorizer = pickle.load(open('tf_vec.pickle', 'rb'))


num = 10 #st.selectbox("How many options would you like?", range(0,21))
user_input = st.text_input("Enter ingredients")
warns = st.empty()


vec_user = tf_vectorizer.transform([user_input])


locs = cosine_similarity(vec_user, dtm_tf)#.argsort()#[::-1]
idxs = (-locs).argsort()[:num]
scores = -np.sort(-cosine_similarity(vec_user, dtm_tf))[0][:num+1]
idxs = zip(idxs[0][:num], scores) 
thresh = st.slider(key='Threshold', label="Threshold", min_value=0.20, max_value=0.80, step=0.01, value = 0.30)
st.write("A lower threshold will include more interesting, diverse options.")
st.write("A higher threshold will demand closer accuracy to your input.")
for idx, score in idxs:
    flag_small = 0
    #st.write(idx, score)
    if score >= thresh:
        break_line = '<hr style="border:2px solid gray"> </hr>'
        st.markdown(break_line, unsafe_allow_html = True)
        answer = df.iloc[idx]
        name = answer[1]
        st.markdown(f"## {name}")
        url = df2[df2.name == name].url.to_string(index = False).strip()
        concat_url = f"https://www.bonappetit.com{url}"
        st.write(concat_url)
        #st.write(f"Score: {score:.02f}")
        try:
            im_url = answer.img_url
            response = requests.get(im_url)
            img = Image.open(BytesIO(response.content))
            st.image(img, use_column_width = True)
        except:
            st.write('Mmmm... something went wrong. Try another recipe on the list!')
            continue
    else: 
        flag_small += 1
        continue
if flag_small > 0:
    warns.warning(f"You should add more description and ingredients and/or lower threshold to get more results!")
    flag_small = 0
