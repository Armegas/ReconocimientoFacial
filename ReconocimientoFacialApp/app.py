from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import numpy as np
import os
from utils import train_model

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Metodología Global
face_recognizer = None
names_dict = {}
display_ids_dict = {}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def reload_model():
    global face_recognizer, names_dict, display_ids_dict
    face_recognizer, names_dict, display_ids_dict = train_model(app.config['UPLOAD_FOLDER'])
    print(f"Modelo entrenado con {len(names_dict)} personas.")

# Carga inicial
reload_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        name = request.form.get('name')
        person_id = request.form.get('id')

        if file.filename == '':
            return redirect(request.url)

        if file and name:
            clean_name = "".join([c for c in name if c.isalnum() or c in (' ', '-')]).strip()
            clean_id = "".join([c for c in person_id if c.isalnum() or c in (' ', '-')]).strip() if person_id else "NA"
            
            filename = f"{clean_name}_{clean_id}{os.path.splitext(file.filename)[1]}"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # Reentrenar modelo
            reload_model()
            
            return redirect(url_for('index'))

    return render_template('upload.html')

def gen_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("No se pudo abrir el dispositivo de video")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Región de Interés
                roi_gray = gray[y:y+h, x:x+w]
                
                label_text = "Desconocido"
                color = (0, 0, 255) # Rojo para desconocido

                if face_recognizer is not None:
                    try:
                        # Predecir
                        # confidence es distancia, menor es mejor. 0 = coincidencia perfecta.
                        # < 100 es usualmente una coincidencia, pero reducimos a 60 para mayor precision.
                        
                        # Apply histogram equalization to match training data
                        roi_gray = cv2.equalizeHist(roi_gray)
                        
                        label_id, confidence = face_recognizer.predict(roi_gray)
                        
                        if confidence < 60:
                            name = names_dict.get(label_id, "Desconocido")
                            d_id = display_ids_dict.get(label_id, "")
                            label_text = f"{name} ({d_id}) {confidence:.1f}"
                            color = (0, 255, 0) # Verde para coincidencia
                        else:
                            label_text = f"Desconocido ({confidence:.1f})"
                    except Exception as e:
                        print(f"Error de predicción: {e}")

                # Dibujar rectángulo
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Dibujar fondo de etiqueta
                cv2.rectangle(frame, (x, y-35), (x+w, y), color, cv2.FILLED)
                
                # Dibujar texto
                cv2.putText(frame, label_text, (x+6, y-6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
