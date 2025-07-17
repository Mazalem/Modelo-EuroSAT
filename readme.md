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

---

## 🚀 Como Executar

1. **Clone o repositório**:

```bash
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo
```

2. **Execute o script principal** (em Python 3.9+):

```bash
python treinando_eurosat.py
```

3. **Resultado esperado**:

- Acurácia final exibida no terminal
- Arquivo `modelo_eurosat.tflite` salvo na pasta do projeto
- Pasta `imagens_eurosat/` criada com exemplos de imagens salvas

---

## 🏠 Estrutura do Código

### 🔹 Carregamento do Dataset

Divide 80% para treino e 20% para teste, com rótulos supervisionados:

```python
(ds_train, ds_test), ds_info = tfds.load(
    'eurosat',
    split=['train[:80%]', 'train[80%:]'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)
```

### 🔹 Pré-processamento

Normaliza imagens (0–1) e organiza em batches:

```python
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label
```

### 🔹 Data Augmentation

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])
```

### 🔹 Modelo CNN

Convoluções + pooling + fully connected:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=IMG_SHAPE),
    data_augmentation,
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
```

---

## 📈 Resultados

Após o treinamento:

```
👌 Acurácia final no EuroSAT: 95.00%  (exemplo)
👌 Modelo salvo como .keras e .tflite com sucesso.
👌 Imagens salvas na pasta imagens_eurosat/
```

---

