import face_recognition
import pickle
import os

# Путь к папке с фотографиями лиц
image_folder = "C:/Users/Nikitka/Desktop/Programing/Camera/known_faces"  # Укажите путь к вашей папке с фотографиями

# База данных эмбеддингов
known_encodings = []
known_names = []

# Проверяем, существует ли основная папка
if not os.path.exists(image_folder):
    print(f"Папка {image_folder} не найдена.")
    exit()

# Флаг для проверки наличия файлов
files_found = False

# Генерация эмбеддингов для каждой фотографии в подпапках
for person_folder in os.listdir(image_folder):
    person_folder_path = os.path.join(image_folder, person_folder)
    
    # Проверяем, является ли это директорией
    if os.path.isdir(person_folder_path):
        print(f"Обработка папки: {person_folder}")
        
        # Имя человека берется из названия папки
        name = person_folder
        
        # Обработка всех изображений в подпапке
        for filename in os.listdir(person_folder_path):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                files_found = True
                try:
                    image_path = os.path.join(person_folder_path, filename)
                    print(f"Обработка файла: {image_path}")  # Отладочное сообщение
                    image = face_recognition.load_image_file(image_path)  # Загрузка изображения
                    
                    # Проверяем, найдены ли лица на изображении
                    face_locations = face_recognition.face_locations(image)
                    if len(face_locations) == 0:
                        print(f"На изображении {filename} не найдено лиц. Файл пропущен.")
                        continue
                    
                    # Получаем эмбеддинг первого лица
                    encoding = face_recognition.face_encodings(image)[0]
                    
                    # Добавляем эмбеддинг и имя в базу данных
                    known_encodings.append(encoding)
                    known_names.append(name)
                    print(f"Эмбеддинг для {filename} успешно создан для {name}.")
                
                except Exception as e:
                    print(f"Ошибка при обработке {filename}: {e}")

if not files_found:
    print("В папке нет подходящих изображений (.jpg, .jpeg, .png).")
else:
    # Сохраняем базу данных в файл
    data = {"encodings": known_encodings, "names": known_names}
    with open("face_database.pkl", "wb") as f:
        pickle.dump(data, f)
    print("База данных успешно создана!")