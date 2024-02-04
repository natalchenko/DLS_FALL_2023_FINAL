# https://habr.com/ru/articles/691598/ -- Дообучение EasyOCR
# https://habr.com/ru/articles/553096/ -- Как я Лигу Легенд парсил
# https://habr.com/ru/companies/ods/articles/681718/ -- Data Science Pet Projects. FAQ
# Полезные инструменты: DVC, Hydra, MLFlow, WandB
# https://habr.com/ru/articles/696820/ -- ML | Hydra
# https://habr.com/ru/articles/718980/ -- 10 полезных сочетаний клавиш в PyCharm
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# pip install -U label-studio

# http://localhost:8080/projects/
# pip install opencv-python scikit-learn pyyaml
# pip install -U label-studio
# pip install notebook
# jupyter-notebook
# training.ipynb
# https://share.streamlit.io/
# pip freeze > requirements.txt
# http://213.171.12.97:8501

import streamlit as st
from ultralytics import YOLO
from easyocr import Reader
import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect import detect_number_plates, recognize_number_plates

# pip install -U label-studio
# streamlit run run_app.py

st.set_page_config(page_title="CV Demo", page_icon=":car:", layout="wide")
st.title('Система контроля движения транспортных средств по территории')
st.markdown("---")

uploaded_file = st.file_uploader("Загрузить изображение", type=["png","jpg", "jpeg"])
upload_path = "tmp"

if uploaded_file is not None:
    # construct the path to the uploaded image
    # and then save it in the `uploads` folder
    image_path = os.path.sep.join([upload_path, uploaded_file.name])
    with open(image_path,"wb") as f:
        f.write((uploaded_file).getbuffer())
    with st.spinner("Идет обработка ..."):
        # load the model from the local directory
        model = YOLO("best.pt")
        # initialize the EasyOCR reader
        reader = Reader(['en','ru'], gpu=True)

        # convert the image from BGR to RGB
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # make a copy of the image to draw on it
        image_copy = image.copy()

        # split the page into two columns
        col1, col2 = st.columns(2)

        # display the original image in the first column
        with col1:
            st.subheader("Исходное изображение")
            st.image(image)

        number_plate_list = detect_number_plates(image, model)

        col2_flag = False

        if number_plate_list!=[]:

            number_plate_list = recognize_number_plates(image_path, reader, number_plate_list)

            for box, text in number_plate_list:

                cropped_number_plate = image_copy[box[1]:box[3], box[0]:box[2]]

                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(image, text, (box[0], box[3] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if not col2_flag:
                    col2_flag = True
                    # display the number plate detection in the second column
                    with col2:
                        st.subheader("Определение номера ТС")
                        st.image(image)

                st.subheader("Детектированный номер ТС")
                st.image(cropped_number_plate, width=300)
                st.success("Текстовый номер ТС: **{}**".format(text))

        else:
            st.error("Номер ТС на изображении не обнаружен!")

else:
    st.info("Необходимо загрузить исходное изображение!")

st.markdown("<br><hr><center>Итоговый проект Натальченко А.В. по курсу DLS Осень 2023, 1-ый поток</center><hr>",
            unsafe_allow_html=True)