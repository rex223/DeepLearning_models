import streamlit as st

def display_image(img):
    st.image(img, caption="Input Image", use_column_width=True)
    st.write("Image displayed successfully.")