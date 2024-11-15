import cv2
import os
import imutils

personName = 'Jose Aldair Rojas Febrero'
dataPath = 'C:/Users/jrjos/OneDrive/Escritorio/Reconocimiento Facial/Data'  # Cambia a la ruta donde hayas almacenado Data
personPath = dataPath + '/' + personName

if not os.path.exists(personPath):
    print('carpeta creada :', personPath)
    os.makedirs(personPath)
    
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Clasificadores para rostro frontal y de perfil
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profileClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
count = 0

while True:
    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    # Detección de rostro frontal
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    # Detección de rostro en perfil izquierdo
    profile_left = profileClassif.detectMultiScale(gray, 1.3, 5)

    # Detección de rostro en perfil derecho (imagen reflejada)
    gray_flipped = cv2.flip(gray, 1)
    profile_right = profileClassif.detectMultiScale(gray_flipped, 1.3, 5)

    # Procesar rostros detectados en frontal
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
        count += 1

    # Procesar rostros detectados en perfil izquierdo
    for (x, y, w, h) in profile_left:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
        count += 1

    # Procesar rostros detectados en perfil derecho (ajustar coordenadas)
    for (x, y, w, h) in profile_right:
        x = frame.shape[1] - x - w  # Ajusta coordenadas del rostro reflejado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
        count += 1

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 400:
        break

cap.release()
cv2.destroyAllWindows()
