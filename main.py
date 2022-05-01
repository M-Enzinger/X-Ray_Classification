import streamlit as st
import pandas as pd
import numpy as np

st.title('X-Ray Classification Team - A Brief Introduction')
st.write("Hello, we are Ilayda and Maximilian, the X-Ray Classifiaction team. Subsequently we will introduce our project and motivation.")
st.write("In the picture below you can see the X-RAY of a chest. You recognize something unusual? Neither we.")
st.write("Inspect the picture carefully, then click the Button below.")
st.image(image="chest-pneumoia.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")


col1, col2, col3 , col4, col5 = st.columns(5)

with col1:
    pass
with col2:
    pass
with col3:
    button1 = st.button("What do I have missed?")
with col4:
    pass
with col5:
    pass

if button1:
  st.image(image="chest-pneumoia_prepared.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  



#st.balloons()
