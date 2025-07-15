import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

# Lista oficial de classes do EuroSAT
CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

@st.cache_resource
def carregar_modelo():
    # https://drive.google.com/file/d/1XN5j-z4b2YQdsSqu_J5WESc3BeLliSzQ/view?usp=sharing
    url = 'https://drive.google.com/uc?id=1XN5j-z4b2YQdsSqu_J5WESc3BeLliSzQ'

    # Baixa o arquivo .tflite
    gdown.download(url, 'modelo_eurosat.tflite', quiet=False)

    # Carrega o modelo TFLite
    interpreter = tf.lite.Interpreter(model_path='modelo_eurosat.tflite')
    interpreter.allocate_tensors()

    return interpreter

def carregar_imagem():
    uploaded_file = st.file_uploader(
        'Envie uma imagem de satélite (64x64 ou maior)', 
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((64, 64))  # Garantir dimensão correta

        st.image(image, caption="Imagem enviada", width=200)
        st.success('Imagem carregada com sucesso!')

        # Pré-processar
        image = np.array(image).astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    return None

def prever_classe(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    probs = output_data[0]
    df = pd.DataFrame({
        'Classe': CLASS_NAMES,
        'Probabilidade (%)': (probs * 100).round(2)
    }).sort_values('Probabilidade (%)')

    # Mostrar gráfico
    fig = px.bar(
        df,
        y='Classe',
        x='Probabilidade (%)',
        orientation='h',
        text='Probabilidade (%)',
        title='Previsão de Classe (EuroSAT)'
    )
    st.plotly_chart(fig)

def main():
    st.set_page_config(page_title="Classificador EuroSAT")

    st.title("🌍 Classificador de Imagens Satélite - EuroSAT")
    st.write("https://colab.research.google.com/drive/1x6Ov0Tn-aEAefkPLmpmxyidWAtAmstic?usp=sharing")
    st.write("Envie uma imagem e descubra sua provável classe.")

    interpreter = carregar_modelo()
    image = carregar_imagem()

    if image is not None:
        prever_classe(interpreter, image)

if __name__ == "__main__":
    main()
