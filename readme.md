# 🌍 EuroSAT – Classificação de Imagens com TensorFlow

Este projeto treina uma **Rede Neural Convolucional (CNN)** para classificar imagens do dataset **EuroSAT** (imagens de satélite em 10 classes).\
Após o treinamento, o modelo é exportado para **TensorFlow Lite (.tflite)**, podendo ser usado em dispositivos móveis ou embarcados.\
Além disso, o código salva exemplos de imagens do conjunto de teste localmente.

---

## 📌 Funcionalidades

✅ Carregamento automático do dataset EuroSAT via `tensorflow_datasets`\
✅ Pré-processamento e normalização de imagens\
✅ Data augmentation para melhorar a generalização do modelo\
✅ Treinamento de uma CNN simples e eficiente\
✅ Callbacks para early stopping e ajuste dinâmico da learning rate\
✅ Exportação do modelo para formato `.tflite`\
✅ Salvamento de imagens de teste no disco

---

## 📦 Dependências

Instale as bibliotecas necessárias antes de executar:

```bash
pip install tensorflow tensorflow-datasets pillow matplotlib
```
```bash
pip install streamlit gdown numpy pandas plotly
```

---

## 🚀 Como Executar

1. **Clone o repositório**:

```bash
git clone https://github.com/Mazalem/Modelo-EuroSAT.git
```

2. **Execute o script principal** (em Python 3.12):

```bash
python treinando_eurosat.py
```

3. **Resultado esperado**:

- Acurácia final exibida no terminal
- Arquivo `modelo_eurosat.tflite` salvo na pasta do projeto
- Pasta `imagens_eurosat/` criada com exemplos de imagens salvas
---
## Teste o modelo EuroSAT

Você pode acessar a aplicação diretamente pelo link abaixo:

[🚀 Acessar o Modelo EuroSAT](https://modelo-eurosat.streamlit.app)
