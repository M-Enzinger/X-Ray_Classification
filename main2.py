import streamlit as st
import pandas as pd
import numpy as np

st.title('X-Ray Classification Team - A Brief Introduction')
st.write("Hello, we are Maxi and Ilayda.")
st.balloons()

resultnormal=st.button("Click here to see the x-ray image of a normal chest")
if resultnormal:
  st.image('Normal-chest.jpeg', caption="this is what a normal chest looks like", width=None, channels="RGB", output_format="auto")
  
resultpneumonia=st.button("Click here to see the x-ray image of a chest with pneumonia")
if resultpneumonia:
  st.image('chest-pneumoia.jpeg', caption="this is what a chest with pneumonia looks like", width=None, channels="RGB", output_format="auto")
 
 
