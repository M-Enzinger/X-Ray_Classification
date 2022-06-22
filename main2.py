import streamlit as st
import requests
import json

with st.expander("Chapter 1: Business Understanding"):
    st.write("Lung disease is one of the most common forms of illness and can come in a variety of forms. Common chronic diseases are, for example, chronic obstructive pulmonary disease (COPD), or asthma. An acute lung disease, from which 15,287 people died in Germany in 2020, is pneumonia [1]. Common symptoms besides fever are chest pain when coughing and rapid breathing [2]. ")
    st.write("How can pneumonia be diagnosed?")
    st.write("In addition to checking the levels of inflammation in the blood, X-rays are an important tool for diagnosis. When the disease is present, white spots stand out in the affected areas of the lungs [2].")
    st.write("[1]: https://de.statista.com/statistik/daten/studie/1042795/umfrage/todesfaelle-aufgrund-der-haeufigsten-diagnosen-von-krankheiten-des-atmungssystems/, [2]: https://www.navigator-medizin.de/krankheiten/lungenentzuendung/behandlung-und-prognose.html?9-sieht-man-eine-lungenentz%c3%bcndung-immer-im-r%c3%b6ntgenbild.html#:~:text=Im%20Blut%20zeigen%20sich%20erh%C3%B6hte%20Entz%C3%BCndungswerte%20von%20CRP,schwarz%20erscheint.%20Neben%20typischen%20gibt%20es%20atypische%20Lungenentz%C3%BCndungen.")
