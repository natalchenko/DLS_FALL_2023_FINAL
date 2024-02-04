from ultralytics import YOLO
from easyocr import Reader
import time
import torch
import cv2
import os

# Пороговый уровень уверенности в правильности детектирования объектов
CONFIDENCE_THRESHOLD = 0.6
# Цвет отрисовки рамки детектированного объекта
COLOR = (0, 255, 0)


def detect_number_plates(image, model):

    start = time.time()

    # Прогоняем изображение через модель
    detections = model.predict(image)[0].boxes.data

    # Проверяем, что результат детектирования не пустой
    if detections.shape != torch.Size([0, 6]):

        boxes = []        # Список рамок детектированных объектов
        confidences = []  # Список степеней уверенности (вероятностей) для каждого
                          # детектированного объекта

        for detection in detections:

            # Извлечем величину уверенности (вероятности) результатах детектирования
            confidence = detection[4]

            # Пропускаем результаты детектирования с низкой вероятностью
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue
            boxes.append(detection[:4])
            confidences.append(detection[4])

        print(f"Детектировано {len(boxes)} номерных знаков")

        # Список координат рамок и вероятностей для детектированных объектов
        number_plate_list = []

        for i in range(len(boxes)):
            # Получим координаты рамки детектированного объекта
            xmin, ymin, xmax, ymax = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
            number_plate_list.append([[xmin, ymin, xmax, ymax]])

            # Нарисуем рамку детектированного объекта
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR, 2)
            text = "CONFIDENCE: {:.2f}%".format(confidences[i] * 100)
            cv2.putText(image, text, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

        end = time.time()

        print(f"На детекцию затрачено: {(end - start) * 1000:.0f} миллисекунд")

        return number_plate_list

    else:
        print("Номерных знаков не обнаружено")
        return []

def recognize_number_plates(image_or_path, reader, number_plate_list):

    start = time.time()

    if isinstance(image_or_path, str):
        image = cv2.imread(image_or_path)
    else:
        image_or_path

    for i, box in enumerate(number_plate_list):
        # Вырежем изображение региона номера ТС
        np_image = image[box[0][1]:box[0][3], box[0][0]:box[0][2]]

        # Пробуем распознать номер ТС
        detection = reader.readtext(np_image, paragraph=True)

        if len(detection) == 0:
            text = ""
        else:
            text = str(detection[0][1])
            # В силу не очень высокой точности выбранной бесплатной библиотеки OCR
            # для русского языка, в качестве "заплатки" временно будем использовать
            # "очистку" распознанной строки от технических символов
            text = text.replace("[", "")
            text = text.replace("]", "")
            text = text.replace(":", "")
            text = text.replace("'", "")
            text = text.replace('"', "")
            text = text.replace("?", "")
            text = text.replace("/", " ")
            text = text.replace("\\", " ")

        number_plate_list[i].append(text)

    end = time.time()
    print(f"На распознавание затрачено: {(end - start) * 1000:.0f} миллисекунд")

    return number_plate_list


if __name__ == "__main__":

    # ----- Код для тестирования процедуры детектирования номера ТС -----

    # Инициализируем модель и модуль распознавания
    model = YOLO("best.pt")
    reader = Reader(['en', 'ru'], gpu=True)
    file_path = "DATASET/IMAGES/TEST/frame675.jpeg"
    _, file_extension = os.path.splitext(file_path)

    if file_extension in ['.jpg', '.jpeg', '.png']:

        print("Обработка изображения...")
        image = cv2.imread(file_path)
        number_plate_list = detect_number_plates(image, model)
        cv2.imshow('Image', image)
        cv2.waitKey(0)

        if number_plate_list:
            number_plate_list = recognize_number_plates(file_path, reader,
                                                        number_plate_list)

        for box, text in number_plate_list:
            cv2.putText(image, text, (box[0], box[3] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
            cv2.imshow('Image', image)
            cv2.waitKey(0)

        print("Выполнено!")

    else:
        print("Неверный формат файла!")

