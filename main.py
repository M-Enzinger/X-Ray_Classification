import streamlit as st
import pandas as pd
import numpy as np

st.title('X-Ray Classification Team - A Brief Introduction')
st.write("Hello, we are Ilayda and Maximilian, the X-Ray Classifiaction team. Subsequently we will introduce our project and motivation.")
st.write("In the picture below you can see the X-RAY of a chest. You recognize something unusual? Neither we.")
st.write("Inspect the picture carefully, then click the Button below.")

button1 = st.button("What do I have missed?", key=None, "Click me", on_click=None, args=None, kwargs=None, *, disabled=False)
if button1:
  st.image(image="Normal-chest.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  



st.balloons()
