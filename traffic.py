from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
from labels import label
from os import listdir
from os.path import isfile, join
from PIL import Image,ImageOps
import io


model = load_model('model_traffic_data.h5')
# print(model.summary())



def predict(file):
    img_width, img_height = 33,33
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    # print(x.shape)
    x = (0.21 * x[:,:,:1]) + (0.72 * x[:,:,1:2]) + (0.07 * x[:,:,-1:])
    # print(x.shape)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)
    return answer

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def import_predict(image_data):
    
     
        image = ImageOps.fit(image_data, (33,33), Image.ANTIALIAS)
        image_array = rgb2gray(np.asarray(image))
        # print(image_array)

        # normalize
        image_array = (image_array.astype(np.float32) / 255.0)
        image_array = image_array.reshape(1,33,33,1)
       
        prediction = model.predict(image_array)
        res = np.argmax(prediction[0])
        return res,

def run():

   
    # image_hospital = Image.open('hospital.jpg')

    st.sidebar.info('This app is created for Traffic Symbol Classification')

    st.title("Traffic Symbol Prediction App")

    # st.sidebar.image(image_hospital)

    st.set_option('deprecation.showfileUploaderEncoding', False)
    onlyfiles = [f for f in listdir("Meta/") if isfile(join("Meta/", f))]
    if st.sidebar.checkbox("Example Run",False,key='1'):
        file_upload = st.sidebar.selectbox("Choose an image for classification", onlyfiles)
          
        if file_upload is not None:
            path="Meta/"+file_upload
            image = Image.open(path)
            st.sidebar.image(image,caption="Uploaded Image",use_column_width=True)
            if st.sidebar.button('Predict'): 
                st.sidebar.write("Classifying...")
                lab=predict(path)
                output=label[lab]
                st.success('The Traffic-Sign is {}'.format(output))

    img_upload = st.file_uploader("Choose an image for classification", type=["jpg","png"])


    if img_upload is not None:
        path=img_upload
        image = Image.open(path)
        st.image(image,caption="Uploaded Image",use_column_width=True)
        if st.button('Predict',key=2): 
            st.write("Classifying...")
            lab=import_predict(image)
            output=label[lab]
            st.success('The Traffic-Sign is {}'.format(output))
    

if __name__ == '__main__':
    run()