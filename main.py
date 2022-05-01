import streamlit as st
import pandas as pd
import numpy as np
import pyowm

#First part - introduction and question - visible
st.title('X-Ray Classification Team - A Brief Introduction')
st.write("Hello, we are Ilayda and Maximilian, the X-Ray Classifiaction team. Subsequently we will introduce our project and motivation.")
st.write("In the picture below you can see the X-RAY of a chest. You recognize something unusual? Neither we.")
st.write("Inspect the picture carefully, then click the Button.")

col1, col2, col3 , col4, col5 = st.columns(5)
with col3:
    button1 = st.button("What have I missed?")
    

#Second part - solution and motivation - firstly unvisible
if button1:
    #Solution what user has missed
    st.image(image="chest-pneumoia_prepared.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.write("Probably you have not detected it, but this person has a dangerous pneumonia, highlighted on the appeared picture above.")
    st.write("You can compare both, a healthy lung and one with pneumonia by clicking these radio buttons:")
    
    
    #Raadio Button to switch between a healthy lung abnd one with pneumonia
    col1, col2, col3 , col4, col5 = st.columns(5)
    with col3:
        radio_button1 = st.radio("I want to see the X-RAY of...",
        ('a healthy lung', 'a lung with pneumonia'))

    if radio_button1 == 'a healthy lung':
        st.image(image="Normal-chest.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    else:
        st.image(image="chest-pneumoia.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        
 
    #Motivation
    st.write("As demonstrated, it is not possible to identify a pneumonia as a non medical. Even physicians sometimes fail in recognising dangerous lung deseases.")
    st.write("Therefore we want to create a machine learned based model as a possible solution. This model will be capable of identifying pneumonia on the bases of X-RAYs.")
    st.write("To state how important the recognition of lung deseases is, we built a chart of all deaths caused by pneumonia per 100th citizens per year in specific EU countries:")
    
    chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])
    st.line_chart(chart_data)
    
    
    #Weather forecast
    st.write("To make sure that you are dressed approriate to avoid a pneumonia, you can check today`s Weather in nuremberg below:")
    api_key = "41c76f28ad89e9493b1aa62dac513ba2"
    lat = "48.208176"
    lon = "16.373819"
    url = "https://api.openweathermap.org/data/2.5/onecall?lat=%s&lon=%s&appid=%s&units=metric" % (lat, lon, api_key)
    response = requests.get(url)
    data = json.loads(response.text)
    
    current = data["current"]["dt"]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", current, "1.2 °F")
    col2.metric("Wind", "9 mph", "-8%")
    col3.metric("Humidity", "86%", "4%")

else:
    st.image(image="chest-pneumoia.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
             




#ideen: wetter standort dropdown menü; 
#st.balloons()
             
