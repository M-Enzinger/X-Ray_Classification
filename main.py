import streamlit as st
import pandas as pd
import numpy as np
import requests
import json


def first_part():
     #First part - introduction and question - visible
     st.title('X-Ray Classification Team - A Brief Introduction')
     st.write("Hello, we are Ilayda, Jan and Maximilian, the X-Ray Classifiaction team. Subsequently we will introduce our project and motivation.")
     st.write("In the picture below you can see the X-RAY of a chest. You recognize something unusual? Neither we.")
     st.write("Inspect the picture carefully, then click the Button.")
     st.image(image="chest-pneumoia.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")   
     
     #Button1 to show answer, motivation, chart
     col1, col2, col3 , col4, col5 = st.columns(5)
     with col3:
          button1 = st.button("What have I missed?")   
        
     if button1:
          #Executes second part to show answer, motivation, chart
          second_part()

          
     #Weather:
     #Weather forecast text
     st.write("To make sure that you are dressed approriate to avoid a pneumonia, you can check today`s Weather by choosing your city below:")

     #User chooses location
     loc = st.selectbox(
          'What is your home town?',
          ('Erlangen', 'Nuremberg', 'Forchheim'))

     if loc == 'Erlangen':
          lat = "49.599941"
          lon = "11.006300"
          button1 = True
     elif loc == 'Nuremberg':
          lat = "49.452103"
          lon = "11.076665"
          button1 = True
     elif loc == 'Forchheim':
          lat = "49.719910"
          lon = "11.058220"
          button1 = True
     
     #Get Data from API
     api_key = "41c76f28ad89e9493b1aa62dac513ba2"
     url = "https://api.openweathermap.org/data/2.5/onecall?lat=%s&lon=%s&appid=%s&units=metric" % (lat, lon, api_key)
     response = requests.get(url)
     data = json.loads(response.text)
    
     #Extract Data
     tempr = data["current"]["temp"]
     wind = data["current"]["wind_speed"]
     hum = data["current"]["humidity"]
    
     #Use the Data in Graphics
     col1, col2, col3 = st.columns(3)
     col1.metric("Temperature", str(tempr) + " °C")
     col2.metric("Wind", str(wind) + " m/s")
     col3.metric("Humidity", str(hum) + " %")
     st.write("Source: api.openweathermap.org")
          
     
def second_part():  
    #Second part - solution and motivation - firstly unvisible - triggered by Button1
    #Solution what user has missed
    st.image(image="chest-pneumoia_prepared.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.write("Probably you have not detected it, but this person has a dangerous pneumonia, highlighted on the appeared picture above.")
    st.write("You can compare both, a healthy lung and one with pneumonia by clicking these radio buttons:") 
        
    #Motivation
    st.write("As demonstrated, it is not possible to identify a pneumonia as a non medical. Even physicians sometimes fail in recognising dangerous lung deseases.")
    st.write("Therefore we want to create a machine learned based model as a possible solution. This model will be capable of identifying pneumonia on the bases of X-RAYs.")
    st.write("To state how important the recognition of lung deseases is, we provide a chart of all deaths caused by pneumonia per 100th citizens per year in specific EU countries:")
    
    #Chart
    st.image("pneumonia_chart.PNG", caption="Deaths caused by Pneumonia in Germany between 1998 and 2019; Source: Federal Statistical Office of Germany", width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    
     
#main - switching between parts
first_part()


#st.balloons()
