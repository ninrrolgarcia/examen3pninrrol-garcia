import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Objetos IA", page_icon="🖼️", layout="centered"
)

st.title("🖼️ Clasificador de Objetos IA")
st.caption("Desarrollado por: **Ninrrol García**")
st.markdown(
    "Sube una imagen o toma una foto para identificar el objeto usando un"
    " modelo de Machine Learning (CIFAR-10)."
)


# Cargar el modelo entrenado
@st.cache_resource
def load_model():
  # Buscar si existe .keras o .h5
  if os.path.exists("modelo_cifar10.keras"):
    return tf.keras.models.load_model("modelo_cifar10.keras")
  elif os.path.exists("modelo_cifar10.h5"):
    return tf.keras.models.load_model("modelo_cifar10.h5")
  else:
    raise FileNotFoundError("Archivo de modelo no encontrado en el servidor.")


try:
  model = load_model()
  st.success("✅ Modelo cargado correctamente.")
except Exception as e:
  st.error(f"Error al cargar el modelo: {e}")
  st.info(f"📁 Archivos detectados en la raíz: {os.listdir('.')}")
  st.stop()

# Clases de CIFAR-10 traducidas al español
CLASSES = [
    'Avión ✈️', 'Automóvil 🚗', 'Pájaro 🐦', 'Gato 🐱', 'Ciervo 🦌',
    'Perro 🐶', 'Rana 🐸', 'Caballo 🐴', 'Barco ⛵', 'Camión 🚚'
]

# Selección de modo de entrada de imagen
st.divider()
option = st.radio("Selecciona la fuente de la imagen:", ("Subir Archivo", "Usar Cámara"))

image_input = None

if option == "Subir Archivo":
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_input = Image.open(uploaded_file)
else:
    camera_file = st.camera_input("Toma una fotografía")
    if camera_file is not None:
        image_input = Image.open(camera_file)

# Si hay una imagen seleccionada, procesarla y predecir
if image_input is not None:
    st.image(image_input, caption="Imagen seleccionada", use_container_width=True)
    
    with st.spinner("Analizando la imagen..."):
        # Preprocesamiento para adaptar a las dimensiones que espera el modelo (32x32)
        img_resized = image_input.convert('RGB').resize((32, 32))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Inferencia / Predicción
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        predicted_class = CLASSES[class_idx]

    # Mostrar resultados
    st.success("¡Análisis completado!")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicción", predicted_class)
    with col2:
        st.metric("Nivel de Confianza", f"{confidence * 100:.2f}%")
