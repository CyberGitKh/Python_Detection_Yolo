import os
import sys
import cv2
import numpy as np


def Saliendo():
    print("Saliendo...")
    sys.exit()
    os.system('cls' if os.name == 'nt' else 'clear')
def load_yolo():
    # Cargar la red YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    # Cargar las clases
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    # Obtener los nombres de las capas de salida
    try:
        output_layers = net.getUnconnectedOutLayersNames()
    except AttributeError:
        # Alternativa para versiones anteriores de OpenCV 3
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # Generar colores aleatorios para cada clase (opcional)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

def detect_objects(img, net, output_layers):
    height, width, _ = img.shape
    # Crear un blob a partir de la imagen
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416),
                                 mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs, width, height

def get_box_dimensions(outputs, width, height, conf_threshold=0.5):
    boxes = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

def draw_labels(boxes, confidences, class_ids, classes, colors, img, nms_threshold=0.4):
    # Aplicar supresión de no-máximos para evitar detecciones duplicadas
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, nms_threshold)
    font = cv2.FONT_HERSHEY_PLAIN
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 2)
    return img

def main():
    
    # Cambia "video.mp4" por la ruta de tu archivo de video
    cap = cv2.VideoCapture("_DSC0369.MOV")
    
    # Abrir la webcam local (índice 0)
    #cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo acceder a la webcam")
        return

    net, classes, colors, output_layers = load_yolo()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar frame")
            break

        outputs, width, height = detect_objects(frame, net, output_layers)
        boxes, confidences, class_ids = get_box_dimensions(outputs, width, height)
        frame = draw_labels(boxes, confidences, class_ids, classes, colors, frame)

        cv2.imshow("Detección YOLO - Webcam", frame)

        # Salir al presionar la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
