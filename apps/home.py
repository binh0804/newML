import streamlit as st
import pandas as pd
import numpy as np
import os

def app():
    with open('apps/static/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    st.write("# MACHINE LEARNING PROJECT")
    st.write("## 1. Topics and Members :male-scientist: ")
    st.write("### SINH VIÊN THỰC HIỆN")
    st.write("Phạm Phúc Bình - 20110252")
    st.write("### CHỦ ĐỀ")
    st.write('Poker card detection')
    st.write('## 2. Thực thi chương trình:arrows_counterclockwise: :+1: :white_check_mark: ')
    st.write('### Cài đặt môi trường:')
    st.write('* `cd ./streamlit-multiapps`')
    st.write('* `pip install -r requirements.txt`')
    st.write('### Khởi chạy:')
    st.write('* `streamlit run app.py`')
    
    st.write('## 2. References')
    st.write('  1. Python')
    st.write('  2. YoloV5')
    st.write('  3. OpenCV')
    st.write('')

    




    


