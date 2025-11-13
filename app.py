from ultralytics import YOLO
import cv2
import tkinter as tk
from PIL import Image, ImageTk

# Carregar modelo
model = YOLO(r"C:\Trbalho rede neural\runs\detect\train11\weights\best.pt")

# Iniciar webcam
cap = cv2.VideoCapture(0)

# Criar janela Tkinter
root = tk.Tk()
root.title("Detecção de Roupas - YOLO")

# Label onde a imagem será exibida
label_video = tk.Label(root)
label_video.pack()

def atualizar_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Fazer predição
    results = model(frame)
    annotated_frame = results[0].plot()

    # Converter BGR para RGB
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Converter para imagem Tkinter
    img = Image.fromarray(annotated_frame)
    imgtk = ImageTk.PhotoImage(image=img)

    # Atualizar label
    label_video.imgtk = imgtk
    label_video.config(image=imgtk)

    # Atualizar de novo após 10 ms
    label_video.after(10, atualizar_frame)

# Iniciar loop
atualizar_frame()
root.mainloop()

# Quando fechar
cap.release()
cv2.destroyAllWindows()

# "from ultralytics import YOLO"
# Essa linha importa a classe YOLO da biblioteca Ultralytics. É ela que permite carregar o modelo treinado e fazer detecções nos frames da webcam.

# "import cv2"
# Importa a biblioteca OpenCV, usada para capturar a imagem da webcam e manipular os frames.

# "import tkinter as tk"
# Importa o Tkinter, que é a biblioteca padrão do Python para criar interfaces gráficas simples.

# "from PIL import Image, ImageTk"
# Importa funções do Pillow para converter as imagens dos frames em um formato que o Tkinter consiga exibir na tela.

# "model = YOLO(r'C:\Trbalho rede neural\runs\detect\train11\weights\best.pt')"
# Carrega o modelo YOLO a partir do arquivo de pesos gerado no treinamento. Esse modelo será usado em cada frame para detectar roupas.

# "cap = cv2.VideoCapture(0)"
# Abre a webcam do computador. O número zero indica que estamos usando a primeira câmera disponível no sistema.

# "root = tk.Tk()"
# Cria a janela principal da interface gráfica onde tudo será exibido.

# "root.title('Detecção de Roupas - YOLO')"
# Define o título que aparece na barra superior da janela.

# "label_video = tk.Label(root)"
# Cria um componente visual dentro da janela, que será responsável por exibir cada imagem da webcam.

# "label_video.pack()"
# Posiciona o componente na janela e faz com que ele apareça na tela.

# "def atualizar_frame():"
# Define a função que será chamada repetidamente para capturar, processar e exibir os frames atualizados na interface.

# "ret, frame = cap.read()"
# Lê um novo frame da webcam. ret indica se a captura foi bem-sucedida e frame contém a imagem capturada.

# "if not ret: return"
# Se o frame não for capturado corretamente, a função é encerrada para evitar erros.

# "results = model(frame)"
# Envia o frame para o modelo YOLO executar a detecção de objetos em tempo real.

# "annotated_frame = results[0].plot()"
# Pega o resultado da detecção e gera uma imagem com caixas, rótulos e marcações desenhadas automaticamente.

# "annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)"
# Converte o padrão de cores do OpenCV (BGR) para o padrão usado pelo Tkinter e pelo Pillow (RGB).

# "img = Image.fromarray(annotated_frame)"
# Transforma o frame, que é um array NumPy, em uma imagem do Pillow.

# "imgtk = ImageTk.PhotoImage(image=img)"
# Converte a imagem do Pillow para um formato específico que pode ser mostrado no Tkinter.

# "label_video.imgtk = imgtk"
# Guarda a imagem dentro do componente Label. Isso evita que o Python delete a imagem automaticamente.

# "label_video.config(image=imgtk)"
# Atualiza o componente visual com a nova imagem processada, mostrando a detecção na tela.

# "label_video.after(10, atualizar_frame)"
# Pede para o Tkinter chamar a função novamente após 10 milissegundos, criando um loop contínuo para exibição ao vivo.

# "atualizar_frame()"
# Inicia o processo chamando a função pela primeira vez.

# "root.mainloop()"
# Inicia o loop principal da interface gráfica, mantendo a janela aberta e funcionando.

# "cap.release()"
# Libera a webcam depois que a janela é fechada.

# "cv2.destroyAllWindows()"
# Fecha qualquer janela criada pelo OpenCV, mesmo que não tenhamos aberto nenhuma explicitamente.