# grades = {
#     "5": "Отлично",
#     "4": "Хорошо",
#     "3": "Удовлетворительно",
#     "2": "Неудовлетворительно"
# }

# # Получаем оценку от пользователя
# grade_input = input("Введите оценку (5, 4, 3 или 2): ")

# # Проверяем, есть ли введенная оценка в словаре и выводим соответствующее значение
# if grade_input in grades:
#     print(f'Ваша оценка: {grade_input}, значение оценки: {grades[grade_input]}')
# else:
#     print('Оценка не найдена :(')
# --------------------------------------------------------------------
# ----------------------ВЫВОД ИЗОБРАЖЕНИЯ/ВИДЕО
# import cv2

# img = cv2.imread('images/peoples.jpg')
# cv2.imshow('Photo', img)

# cv2.waitKey(0)
# ------------------------РАБОТА С КАМЕРОЙ
# import cv2

# cap = cv2.VideoCapture(0)

# cap.set(3, 500)
# cap.set(4, 500)
# while True:
#     success, img = cap.read()
#     cv2.imshow('Result', img)
#     if cv2.waitKey(1) and 0xff == ord('q'):
#         break
# ---------------------------------------------------------------------------
# import cv2
# import numpy as np
# -----------------------РАЗМЫТИЕ, ДЕДЕНИЕ ПОПОЛАМ И Т.Д
# img = cv2.imread('img/peoples.jpg')
# img = cv2.resize(img, (img.shape[1] // 2, img.shape[0]))
# img = cv2.GaussianBlur(img, (9, 9), 0)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.Canny(img, 200, 200)

# kernel = np.ones((5, 5), np.uint8)
# img = cv2.dilate(img, kernel, iterations=1)
# img = cv2.erode(img, kernel, iterations=1)



# # [0:100, 0:300] ----- ОБРЕЗКА
# cv2.imshow('result', img)

# # print(img.shape)


# cv2.waitKey(0)
# ----------------------------------------------------------------------------
# ----------------------------ЗАКРАСКА
# import cv2
# import numpy as np

# photo = np.zeros((300, 300, 3), dtype='uint8')

# # RGB - стандарт
# # BRG - формат в Open-cv
# # photo[100:150, 100:150 ] = 0, 0, 255 -цвет
# # thickness = cv2.FILLED - полностью закрашено
# cv2.rectangle(photo, (0, 0), (100,100), (0, 0, 255), thickness=cv2.FILLED)
# # Если до конца поля, то photo.shape[1]
# cv2.line(photo, (0, photo.shape[1] // 2), (photo.shape[1], photo.shape[0] // 2), (0, 0, 255,), thickness=3)
# cv2.circle(photo, (photo.shape[1] // 2, photo.shape[0] // 2), 50, (0, 0, 225), thickness=2)
# cv2.putText(photo, 'Dan1lk_ka', (0, photo.shape[1] // 2), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)

# cv2.imshow('Photo', photo)
# cv2.waitKey(0)
# ---------------------------------------------------------------------------
# ----------------------------СМЕЩЕНИЕ
# import cv2
# import numpy as np

# img = cv2.imread('images/peoples.jpg')

# # img = cv2.flip(img, 1)
# def rotate(img_param, angle):
#     height, width = img_param.shape[:2]
#     point = (width // 2, height // 2)

#     mat = cv2.getRotationMatrix2D(point, angle, 1)
#     return cv2.warpAffine(img_param, mat, (width, height))

# # img = rotate(img, -90)

# def transform(img_param, x, y):
#     mat = np.float32([[1, 0, x], [0, 1, y]])
#     return cv2.warpAffine(img_param, mat, (img_param.shape[1], img_param.shape[0]))

# img = transform(img, 30, 40)

# cv2.imshow('Result', img)

# cv2.waitKey(0)
# ------------------------------------------------------------------------
# -------------------------------КОНТУРЫ
# import cv2
# import numpy as np

# img = cv2.imread('images/peoples.jpg')

# new_img = np.zeros(img.shape, dtype='uint8')

# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# img = cv2.GaussianBlur(img, (5, 5), 0)

# # (img, 100, 140) ---цвета до 100 -проигнорированы (черные), от 140 тоже проигнорированы(былые)
# img = cv2.Canny(img, 100, 140)

# con, hir = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# cv2.drawContours(new_img, con, -1, (230, 111, 148), 1)


# cv2.imshow('Result', new_img)

# cv2.waitKey(0)
# ----------------------------------------------------------------------
# ----------------------------ЦВЕТОВЫЕ ФОРМАТЫ
# import cv2
# import numpy as np

# img = cv2.imread('images/peoples.jpg')

# img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
# img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# r, g ,b = cv2.split(img)

# img = cv2.merge([r, g, b])

# cv2.imshow('Result', img)

# cv2.waitKey(0)
# -------------------------------------------------------------------------
# ------------------------ПОБИТОВЫЕ ОПЕРАЦИИ, МАСКИ
# import cv2
# import numpy as np

# photo = cv2.imread('images/peoples.jpg')

# # пустое полотно с такими же размерами ([:2])
# img = np.zeros(photo.shape[:2], dtype="uint8") 

# circle = cv2.circle(img.copy(), (photo.shape[1] // 2, photo.shape[0] // 2), 80, 255, -1)
# square = cv2.rectangle(img.copy(), (25, 25), (250, 300), 255, -1)

# img = cv2.bitwise_and(photo, photo, mask=circle)
# # img = cv2.bitwise_or(circle, square)
# # img = cv2.bitwise_xor(circle, square)
# # img = cv2.bitwise_not(circle)

# cv2.imshow('Result', img)

# cv2.waitKey(0)
# ---------------------------------------------------------------------
# -------------------------------РАСПОЗНАВАНИЕ ЛИЦ
# import cv2
# import numpy as np

# while True:
    
#     img = cv2.imread('images/peoples.jpg')
    
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     faces = cv2.CascadeClassifier('faces.xml')

#     results = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9)

#     for (x, y, w, h) in results:
#         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), thickness=3)


#     cv2.imshow('Result', img)

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import face_recognition

def face_rec():
    tima_face_img = face_recognition.load_image_file('images/tima.jpg')
    tima_face_img_location = face_recognition.face_locations(tima_face_img)

    peoples_faces_img = face_recognition.load_image_file('images/peoples.jpg')
    peoples_faces_img_location = face_recognition.face_locations(peoples_faces_img)

    print(tima_face_img_location)
    print(peoples_faces_img_location)
    print(f'Found {len(tima_face_img_location)}face(s) in this picture')
    print(f'Found {len(peoples_faces_img_location)}face(s) in this picture')
    








def main():
    face_rec



if __name__ == '__main__':
    main()

# import cv2
# face_cascade = cv2.CascadeClassifier('faces.xml') # Загрузка каскадного классификатора для распознавания лиц
# cap = cv2.VideoCapture(0) # Захват видео с камеры по умолчанию (0)
# while True:
#     ret, frame = cap.read() # Считывание кадра с камеры

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Преобразование кадра в оттенки серого для более быстрого обнаружения

#     faces = face_cascade.detectMultiScale(gray, 2, 1) # Обнаружение лиц в кадре

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Рисуем прямоугольник вокруг каждого обнаруженного лица
        
#     cv2.imshow('Video', frame) # Отобрази кадр с обнаруженными лицами

#     if cv2.waitKey(1) == ord('q'): # Выход, если нажата клавиша "q".
#         break
