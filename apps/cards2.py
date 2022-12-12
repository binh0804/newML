from io import StringIO
from pathlib import Path
import streamlit as st
import cv2
from  yolov5.test2 import detect
import os
import numpy as np
import pandas as pd
import av
import argparse
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import json
def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


def app():
    with open('apps/static/style_card.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    


    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='weights/playing_cards.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    source = ("Hình ảnh", "Video")
    source_index = st.sidebar.selectbox("Chọn đầu vào", range(
        len(source)), format_func=lambda x: source[x])
    st.title('Poker Cards Recognization')
    first = st.container()
    with first:
        if source_index == 0:
            uploaded_file = st.sidebar.file_uploader(
                "   ### Load an image", type=['png', 'jpeg', 'jpg'])
            if uploaded_file is not None:
                is_valid = True
                with st.spinner(text='Loading...'):
                    st.sidebar.image(uploaded_file)
                    picture = Image.open(uploaded_file)
                    picture = picture.save(f'yolov5/data/images/{uploaded_file.name}')
                    opt.source = f'yolov5/data/images/{uploaded_file.name}'
            else:
                is_valid = False
        elif source_index==1:
            uploaded_file = st.sidebar.file_uploader("Load a video", type=['mp4'])
            if uploaded_file is not None:
                is_valid = True
                with st.spinner(text='Loading...'):
                    st.sidebar.video(uploaded_file)
                    with open(os.path.join("yolov5","data", "video",uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    opt.source = f'yolov5/data/video/{uploaded_file.name}'
            else:
                is_valid = False
        else:
            is_valid = True
            # def process(img):
            #     opt.source = img
            class VideoProcessor:
                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")        
                    img = cv2.flip(img,1)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

        if is_valid:
            if source_index ==2:
                    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
                    webrtc_ctx = webrtc_streamer(
                    key="WYH",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    media_stream_constraints={"video": True, "audio": False},
                    video_processor_factory=VideoProcessor,
                    async_processing=True,
                    )
            else:
                processed = False
                if st.button('Bắt đầu nhận diện'):
                    MSG_POKER = detect(opt)
                    print(MSG_POKER)
                    with open('apps/data.txt',"r", encoding='utf8') as fd:
                            if fd.mode == 'r':
                                data = fd.read()
                    newDict = json.loads(data)
                    with open('apps/data2.txt',"r", encoding='utf8') as f:
                        if f.mode == 'r':
                            data2 = f.read()
                    dataDict = json.loads(data2)
                    
                    if source_index == 0:
                        with st.spinner(text='Xử lý Images'):
                            for img in os.listdir(get_detection_folder()):
                                st.image(str(Path(f'{get_detection_folder()}') / img))
                            st.balloons()
                           
                            if MSG_POKER == {}:
                                st.write("Kết quả nhận diện: Không detect được tay bài nào")
                            else:
                                tempResText = str(list(MSG_POKER.keys())[0])
                                st.write("Kết quả nhận diện: "+ newDict[tempResText])
                                st.text(dataDict[tempResText])
                                st.title('')
                                st.title('')
                                st.title('')
                                st.title('')

                    elif source_index == 1:
                        
                        with st.spinner(text='Xử lý Video'):
                            for vid in os.listdir(get_detection_folder()):
                                st.video(str(Path(f'{get_detection_folder()}') / vid))
                            st.balloons()
                          
                            processed = True

                # if st.button("Load")         
                    if processed:
                        #khai báo biến sử dụng và gọi từ txt
                        resText = ''
                        
                        #In số lần xuất hiện
                        for text,times_appear in MSG_POKER.items():
                            resText = newDict[text] + " xuất hiện " + str(times_appear) + " lần.\n"
                            st.write(resText)
                        #vẽ đồ thị số lần xuất hiện
                        pvalue = []
                        for key in MSG_POKER.keys():
                            pvalue.append(MSG_POKER[key])
                        
                        def getMaxText():
                            if pvalue==[]:
                                return ''
                            else:
                                maxVal = max(pvalue)
                                for text,times_appear in MSG_POKER.items():
                                    if times_appear == maxVal:
                                        return text
                        maxText = getMaxText()    
                        if maxText == '': 
                            st.write("Kết quả nhận diện: Không detect được tay bài nào")
                        else:
                            chart_data = pd.DataFrame(pvalue,MSG_POKER.keys())
                            st.bar_chart(chart_data)

                            #In thông tin của lần nhiều nhất
                            st.write(newDict[maxText] + " xuất hiện nhiều nhất. Hãy cùng tìm hiểu về " + newDict[maxText])
                            st.text(dataDict[maxText])

    st.write('#### What is cards?')
    st.write('## 1. Overview :icecream: :tada: :star-struck: ')
    st.write('  * Một bộ bài có 52 lá và chia thành 4 chất: HEART :heart:, DIAMOND :diamonds: , CLUBS :clubs: , SPADES :spades:.')
    cols = st.columns(4) # number of columns in each row! = 2
    # first column of the ith row
    cols[0].image("apps/static/images/spade.png", use_column_width=True)
    cols[1].image("apps/static/images/cludes.png", use_column_width=True)
    cols[2].image("apps/static/images/diamond.png", use_column_width=True)
    cols[3].image("apps/static/images/heart.png", use_column_width=True)

    st.write('  * Mỗi chất có 13 lá với giá trị từ 2 tới 10 và 4 lá đặc biệt KING 👑, QUEEN👸, JACK 👦, ACES 🅰️')
    st.write('#### Poker rule')
    st.write('  * Mỗi người chơi sẽ được phát 2 lá bài và phải kết hợp với 5 lá bài chung trên bàn để chọn ra một bộ 5 lá mạnh nhất có thể.')
    st.title('Hãy chọn ảnh/video và xem kết quả')
    # st.write('  STRAIGHT (Sảnh): Any five consecutive cards of different suits')
    # st.write('  FLUSH (Thùng): Five cards with the same suit')
    # st.write('  STRAIGHT FLUSH (Thùng phá sảnh): Any straight with all five cards of the same suit')
    # st.write('  FOUR OF A KIND (Tứ quý): Four cards of one rank and one card of another rank')
    # st.write('  THREE OF A KIND (Bộ ba): any three cards of the same number or face value')
    # st.write('  FULL HOUSE (Cù lũ): THREE OF A KIND plus any other two cards of the same number or face')
    # st.write('  PAIR (Đôi): two cards of one rank')
    # st.write('  TWO PAIR (Hai đôi): two cards of one rank, two cards of another rank')
    # st.write('  HIGH CARD (Mậu thầu): ACES with a 10, J, Q or K')

   