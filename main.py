import streamlit as st
import pandas as pd
import numpy as np

#First part - introduction and question - visible
st.title('X-Ray Classification Team - A Brief Introduction')
st.write("Hello, we are Ilayda and Maximilian, the X-Ray Classifiaction team. Subsequently we will introduce our project and motivation.")
st.write("In the picture below you can see the X-RAY of a chest. You recognize something unusual? Neither we.")
st.write("Inspect the picture carefully, then click the Button.")


col1, col2, col3 , col4, col5 = st.columns(5)

with col1:
    pass
with col2:
    pass
with col3:
    button1 = st.button("What have I missed?")
with col4:
    pass
with col5:
    pass

#Second part - solution and motivation - firstly unvisible
if button1:
    st.image(image="chest-pneumoia_prepared.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.write("Probably you have not detected it, but this person has a dangerous pneumonia, highlighted on the appeared picture above.")
    st.write("You can compare both, a healthy lung and one with pneumonia by clicking these radio buttons:")
    
    
    col1, col2, col3 , col4, col5 = st.columns(5)

    with col1:
        pass
    with col2:
        pass
    with col3:
        radio_button1 = st.radio("I want to see the X-RAY of...",
        ('a healthy lung', 'a lung with pneumonia'))
    with col4:
        pass
    with col5:
        pass


    if radio_button1 == 'a healthy lung':
        button1 = true
        st.image(image="Normal-chest.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        button1 = true
    else:
        st.image(image="chest-pneumoia.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

else:
    st.image(image="chest-pneumoia.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")




#st.balloons()
