import cv2
from ultralytics import YOLO
import face_recognition
import pickle
import numpy as np
import pyttsx3  # Библиотека для озвучивания текста

# Загрузка предобученной модели YOLOv8 для детекции объектов
model = YOLO(r"C:\Users\Nikitka\Desktop\Programing\Camera\yolo11n.pt")  # Общая модель для детекции объектов

# Загрузка базы данных эмBEDding'ов
with open("face_database.pkl", "rb") as f:
    data = pickle.load(f)
known_encodings = data["encodings"]
known_names = data["names"]

# Инициализация текст-to-речи
engine = pyttsx3.init()

# Функция для сравнения эмBEDding'ов
def recognize_face(face_encoding, known_encodings, known_names, tolerance=0.5):
    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
    if True in matches:
        index = matches.index(True)
        return known_names[index]
    return "Noname"

# Функция для озвучивания текста
def speak(text):
    """Произносит заданный текст."""
    engine.say(text)
    engine.runAndWait()

# Открываем доступ к веб-камере
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Ширина кадра
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Высота кадра

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

# Переменная для отслеживания последнего распознанного человека
last_person = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break

    # Выполняем детекцию на кадре
    results = model(frame, conf=0.5)  # Установим порог уверенности в 50%

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls.item())
            confidence = box.conf.item()

            if cls_id == 0 and confidence >= 0.3:  # Класс "person"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_frame = frame[max(y1 - 50, 0):y2 + 50, max(x1 - 50, 0):x2 + 50]

                try:
                    # Находим лицо внутри рамки человека
                    face_locations = face_recognition.face_locations(person_frame)
                    if len(face_locations) > 0:
                        print("Лицо найдено.")
                        face_encoding = face_recognition.face_encodings(person_frame, face_locations)[0]
                        person_name = recognize_face(face_encoding, known_encodings, known_names)

                        color = (0, 255, 0) if person_name != "Noname" else (0, 0, 255)
                        top, right, bottom, left = face_locations[0]

                        # Проверяем координаты рамки
                        if x1 + left >= 0 and y1 + top >= 0 and x1 + right <= frame.shape[1] and y1 + bottom <= frame.shape[0]:
                            cv2.rectangle(frame, (x1 + left, y1 + top), (x1 + right, y1 + bottom), color, 2)
                            cv2.putText(frame, person_name, (x1 + left, y1 + top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                            # Если распознан конкретный человек ("Nikitka"), произносим приветствие
                            if person_name == "Nikita" and last_person != "Nikita":
                                speak("Здравствуйте, Никита!")
                                last_person = "Nikita"
                            elif person_name != "Nikita":
                                last_person = None  # Сброс, если распознан другой человек
                        else:
                            print("Координаты лица выходят за пределы кадра.")
                    else:
                        print("Лицо не найдено.")
                except Exception as e:
                    print(f"Ошибка при обработке лица: {e}")

    # Показываем кадр
    cv2.imshow("Детекция людей", frame)

    # Выход из цикла по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()