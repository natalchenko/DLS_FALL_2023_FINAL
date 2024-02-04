import asyncio
import cv2
import os

from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.types import FSInputFile
from config import bot_token
from aiogram import F

from ultralytics import YOLO
from easyocr import Reader
from detect import detect_number_plates, recognize_number_plates

# Инициализация экземпляра Bot с режимом разбора по умолчанию, который будет передан всем вызовам API
bot = Bot(bot_token, parse_mode=ParseMode.HTML)
dp = Dispatcher()


@dp.message(F.photo)
async def echo_handler(message: types.Message) -> None:

    tmp_file_name = f"./tmp/{message.photo[-1].file_id[:40]}"
    tmp_file_name_full = tmp_file_name  + ".jpg"
    tmp_file_name_full2 = tmp_file_name + "_2.jpg"

    print(tmp_file_name_full)
    await bot.download(message.photo[-1], destination=tmp_file_name_full)

    # Загружаем обученную модель
    model = YOLO("best.pt")
    # Инициализируем библиотеку распознавания EasyOCR, модуль reader
    reader = Reader(['en', 'ru'], gpu=True)

    # Меняем формат с BGR на RGB
    image = cv2.cvtColor(cv2.imread(tmp_file_name_full), cv2.COLOR_BGR2RGB)
    image_copy = image.copy()

    number_plate_list = detect_number_plates(image, model)

    if number_plate_list:

        number_plate_list = recognize_number_plates(tmp_file_name_full, reader, number_plate_list)

        cv2.imwrite(tmp_file_name_full2, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        picture = FSInputFile(tmp_file_name_full2)

        await message.answer("<b>РЕЗУЛЬТАТ ДЕТЕКТИРОВАНИЯ:</b>")
        await message.answer_photo(picture)
        os.remove(tmp_file_name_full2)

        for box, text in number_plate_list:


            cropped_number_plate = image_copy[box[1]: box[3], box[0]: box[2]]

            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(image, text, (box[0], box[3] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if text:
                cv2.imwrite(tmp_file_name_full2, cv2.cvtColor(cropped_number_plate, cv2.COLOR_RGB2BGR))
                cropped_number_picture = FSInputFile(tmp_file_name_full2)

                await message.answer(f"<b>Номер ТС:</b> {text}")
                await message.answer_photo(cropped_number_picture)
                os.remove(tmp_file_name_full2)

    else:
        await message.reply("Номеров ТС на изображении не обнаружено!")

    os.remove(tmp_file_name_full)


@dp.message(F.text)
async def echo_handler(message: types.Message) -> None:
    await message.reply("Я Вас не понимаю, пришлите, пожалуйста, изображение с номером ТС!")


async def main() -> None:
    # Запуск обработки событий
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
