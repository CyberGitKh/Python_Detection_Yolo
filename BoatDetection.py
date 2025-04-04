import cv2
import time
import os
import requests
from bs4 import BeautifulSoup
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import smtplib
from email.message import EmailMessage
import json
import datetime

# ------------------------------
# Función para enviar correo usando Gmail con mensaje en texto plano y HTML
# ------------------------------
def send_email_gmail(sender_email, sender_password, receiver_emails, subject, plain_text_body, html_body, attachment_paths=[]):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender_email
    if isinstance(receiver_emails, list):
        msg['To'] = ", ".join(receiver_emails)
    else:
        msg['To'] = receiver_emails

    msg.set_content(plain_text_body)
    msg.add_alternative(html_body, subtype='html')
    
    for attachment_path in attachment_paths:
        with open(attachment_path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(attachment_path)
        msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)
    
    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)
    print("Correo enviado exitosamente!")

# ------------------------------
# Configuración de Roboflow y modelo de barco
# ------------------------------
roboflow_api_key = 'PaRXhopJ9laN1CedAkvP'  # Reemplaza con tu API Key correcta
model_boat = 'little-boat-model/1'          # Modelo para detectar barcos (Yolov10)

# Umbral de confianza para la detección de barcos (ajusta según tus necesidades)
THRESHOLD_BOAT = 0.5
config_boat = InferenceConfiguration(confidence_threshold=THRESHOLD_BOAT)

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=roboflow_api_key
)

# ------------------------------
# Título fijo para el archivo de video
# ------------------------------
video_title = "Archivo de Video"

# ------------------------------
# Ruta del archivo de video de entrada
# ------------------------------
video_file = "ruta_del_video.mp4"  # Reemplaza con la ruta de tu archivo de video

# ------------------------------
# Función para capturar y procesar un frame a partir de una posición (en milisegundos) del video
# ------------------------------
def capturar_y_procesar(indice, start_msec):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Captura {indice}: No se pudo abrir el archivo de video.")
        return False, "Error", None, None, None
    # Posicionar el video a la marca deseada (en ms)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_msec)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Captura {indice}: No se pudo capturar un frame en {start_msec} ms.")
        return False, "Error", None, None, None

    frame = cv2.resize(frame, (640, 480))
    raw_path = f"captura_{indice}.jpg"
    cv2.imwrite(raw_path, frame)

    # Mostrar la captura brevemente (3 segundos)
    cv2.imshow(f"Captura {indice}", frame)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    # Inferencia con el modelo de barco
    with CLIENT.use_configuration(config_boat):
        result_boat = CLIENT.infer(raw_path, model_id=model_boat)

    # Función para dibujar las predicciones en la imagen
    def dibujar_predicciones(frame, predictions, color, scale=1.0):
        for pred in predictions:
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            w_scaled = w * scale
            h_scaled = h * scale
            x1 = int(x - w_scaled/2)
            y1 = int(y - h_scaled/2)
            x2 = int(x + w_scaled/2)
            y2 = int(y + h_scaled/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{pred['class']} ({pred['confidence']:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    predictions_boat = result_boat.get('predictions', [])
    dibujar_predicciones(frame, predictions_boat, (255, 0, 0))

    msg = "Barco detectado." if predictions_boat else "No se detectó barco."
    processed_path = f"captura_{indice}_procesada.jpg"
    cv2.imwrite(processed_path, frame)

    # Mostrar la imagen procesada (5 segundos)
    cv2.imshow(f"Detecciones Captura {indice}", frame)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

    match = bool(predictions_boat)
    cap_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return match, msg, processed_path, raw_path, cap_time

# ------------------------------
# Función para extraer un segmento de video (a partir de una posición en ms) durante una duración dada (en segundos)
# ------------------------------
def grabar_video_segmento(indice, start_msec, duracion=10):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("No se pudo abrir el archivo de video para extraer segmento.")
        return None

    # Posicionar el video a la marca de inicio
    cap.set(cv2.CAP_PROP_POS_MSEC, start_msec)
    fps = cap.get(cv2.CAP_PROP_FPS)
    resolucion = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    video_filename = f"video_{indice}_{timestamp}.avi"
    out = cv2.VideoWriter(video_filename, fourcc, fps if fps > 0 else 20, resolucion)

    start_time = time.time()
    # Calcular el número de frames a extraer según la duración y fps
    num_frames = int(duracion * (fps if fps > 0 else 20))
    frames_leidos = 0
    while frames_leidos < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resolucion)
        out.write(frame)
        frames_leidos += 1

    cap.release()
    out.release()
    print(f"Segmento de video grabado: {video_filename}")
    return video_filename

# ------------------------------
# Bucle principal: procesar el video cada 60 segundos (60,000 ms)
# ------------------------------
current_msec = 0
while True:
    # Se asume que se extrae un frame cada 60 segundos de video
    match, msg, proc_path, raw_path, cap_time = capturar_y_procesar(1, current_msec)
    print(f"Captura en {current_msec} ms realizada a las {cap_time}: {msg}")

    attachment_files = []
    if match:
        # Extraer un segmento de 10 segundos a partir del instante actual
        video_segment = grabar_video_segmento(1, current_msec, duracion=10)
        if video_segment:
            attachment_files.append(video_segment)

        sender_email = "mail@gmail.com"  # Reemplaza con tu dirección de Gmail
        sender_password = "app-passwd"         # Reemplaza con tu contraseña o app password
        try:
            file_path = os.path.join(os.path.dirname(__file__), "mails_users.txt")
            with open(file_path, "r") as f:
                receiver_emails = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print("Error leyendo mails_users.txt:", e)
            receiver_emails = []

        if not receiver_emails:
            print("No se encontraron destinatarios en mails_users.txt.")
        else:
            email_subject = f"Reporte de Detección de Barco - {video_title}"
            plain_text_body = (
                f"Reporte de Detección de Barco - {video_title}\n\n"
                f"Captura en {current_msec} ms realizada a las {cap_time}: {msg}\n\n"
                "Se adjunta un segmento de video del evento.\n\n"
                "Atentamente,\nEl equipo de IA_Webcam_Boat_Report"
            )
            html_body = f"""
            <html>
              <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                <h1 style="color: #2E86C1;">Reporte de Detección de Barco - {video_title}</h1>
                <p>Captura en {current_msec} ms realizada a las {cap_time}: {msg}</p>
                <p>Se adjunta un segmento de video del evento.</p>
                <p>Atentamente,<br>El equipo de IA_Webcam_Boat_Report</p>
              </body>
            </html>
            """
            send_email_gmail(sender_email, sender_password, receiver_emails, email_subject,
                             plain_text_body, html_body, attachment_files)
    else:
        print("No se detectó barco en esta captura.")

    # Borrar archivos temporales (imágenes y segmento de video)
    for path in [proc_path, raw_path]:
        if path and os.path.exists(path):
            os.remove(path)
    for video in attachment_files:
        if video and os.path.exists(video):
            os.remove(video)

    # Avanzar 60,000 ms (1 minuto) en el video
    current_msec += 60000

    # Verificar si se llegó al final del video
    cap_temp = cv2.VideoCapture(video_file)
    total_msec = cap_temp.get(cv2.CAP_PROP_FRAME_COUNT) / cap_temp.get(cv2.CAP_PROP_FPS) * 1000
    cap_temp.release()
    if current_msec > total_msec:
        print("Se alcanzó el final del video.")
        break

    # Opcional: pausa real para no saturar el sistema (puede omitirse si se desea procesar rápidamente)
    time.sleep(1)
