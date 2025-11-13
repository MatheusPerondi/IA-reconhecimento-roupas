from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Treinar
model.train(
    data="C:\Trbalho rede neural\modelo\data.yaml",  
    epochs=80,                      
    imgsz=640,                       
    batch=16                         
)

# "from ultralytics import YOLO"
# Essa linha importa a classe YOLO da biblioteca Ultralytics. É ela que permite carregar modelos prontos ou iniciar um treinamento do zero.

# "model = YOLO('yolov8n.pt')"
# Aqui o modelo YOLO é carregado usando o arquivo yolov8n.pt. Esse é o modelo “nano”, o menor e mais leve da família YOLOv8, ideal para treinos rápidos ou máquinas com pouco poder de processamento.

# "# Treinar"
# Esse comentário apenas indica que o bloco abaixo será usado para iniciar o processo de treinamento do modelo.

# "model.train("
# Essa linha inicia o módulo de treinamento do YOLO. Tudo que for passado entre parênteses representa as configurações do treino.

# 'data="C:\Trbalho rede neural\modelo\data.yaml",'
# Define o caminho para o arquivo data.yaml, que contém a configuração do dataset (caminho das imagens, número de classes, nomes das classes, etc.). É essencial para o YOLO saber onde estão os dados de treino e validação.

# "epochs=80,"
# Define o número de épocas do treinamento. Uma época representa uma “passada completa” por todo o conjunto de treinamento. Quanto maior o número de épocas, mais o modelo aprende — mas leva mais tempo.

# "imgsz=640,"
# Define o tamanho das imagens que serão usadas no treino. O padrão é 640. Valores maiores aumentam a precisão, mas também exigem mais processamento.

# "batch=16"
# Define o tamanho do batch, ou lote, de imagens que serão processadas por vez durante o treinamento. Valores maiores aceleram o treino, mas também consomem mais memória.

# ")"
# Fecha a chamada do método train(), iniciando o processo de treinamento do modelo com todas as configurações fornecidas.