from sklearn.model_selection import train_test_split
import cv2
import os
import yaml

# ---------- Разделим исходные изображения на выборки -----------

root_dir = "DATASET/CAR_PLATES/"
valid_formats = [".jpg", ".jpeg", ".png", ".txt"]

def file_paths(root, valid_formats):
    file_paths = []

    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            extension = os.path.splitext(filename)[1].lower()

            # Проверим расширение входного файла
            if extension in valid_formats:
                file_path = os.path.join(dirpath, filename)
                file_paths.append(file_path)

    return file_paths


image_paths = file_paths(root_dir + "IMAGES", valid_formats[:3])
label_paths = file_paths(root_dir + "LABELS", valid_formats[-1])

X_train, X_val_test, y_train, y_val_test = train_test_split(image_paths, label_paths, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.7, random_state=42)


def write_to_file(images_path, labels_path, X):

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    for img_path in X:
        img_name = img_path.split("/")[-1].split(".")[0]
        img_ext = img_path.split("/")[-1].split(".")[-1]
        image = cv2.imread(img_path)
        cv2.imwrite(f"{images_path}/{img_name}.{img_ext}", image)

        f = open(f"{labels_path}/{img_name}.txt", "w")
        label_file = open(f"{root_dir}/LABELS/{img_name}.txt", "r")
        f.write(label_file.read())
        f.close()
        label_file.close()

write_to_file("DATASET/IMAGES/TRAIN", "DATASET/LABELS/TRAIN", X_train)
write_to_file("DATASET/IMAGES/VALID", "DATASET/LABELS/VALID", X_val  )
write_to_file("DATASET/IMAGES/TEST",  "DATASET/LABELS/TEST",  X_test )


# ---------- Создаем конфигурационный YAML файл ----------

# Словарь с настройками каталогов
data = {
    "path" : "../DATASET",   # Корневой каталог
    "train": "IMAGES/TRAIN", # Тренировочные изображения
    "val"  : "IMAGES/VALID", # Изображения из валидационной выборки
    "test" : "IMAGES/TEST",  # Изображения из тестовой выборки

    # Детектируемые классы
    "names":["number plate"]
}

with open("model_config.yaml", "w") as f:
    yaml.dump(data, f)
