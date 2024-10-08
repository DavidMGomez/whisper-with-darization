import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# Cargar el modelo preentrenado de SpeechBrain
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp")

# Cargar el archivo de audio (debe estar en formato .wav)
signal, sr = torchaudio.load("audio_convertido.wav")  # sr: frecuencia de muestreo (16000 Hz es estándar para voz)

# Revisar las dimensiones de la señal de audio
# signal tiene la forma [num_channels, num_samples], lo convertimos a [batch_size, num_samples]
signal = signal.squeeze(0)  # Eliminar la dimensión de los canales si es mono (debe ser [num_samples])

# Lista de segmentos (inicio y fin en segundos)
segmentos = [(1.0, 3.0), (5.0, 7.0)]  # Ejemplo de segmentos (1-3s y 5-7s)

# Función para extraer el segmento de audio correspondiente
def obtener_segmento(signal, sr, inicio, fin):
    inicio_muestra = int(inicio * sr)  # Convertir segundos a muestras
    fin_muestra = int(fin * sr)        # Convertir segundos a muestras
    return signal[inicio_muestra:fin_muestra]

# Lista para almacenar los embeddings de cada segmento
embeddings_list = []

# Procesar cada segmento
for inicio, fin in segmentos:
    # Extraer el segmento de audio
    segmento = obtener_segmento(signal, sr, inicio, fin)
    
    # Convertir el segmento a formato batch (batch_size=1, num_samples)
    segmento_tensor = torch.unsqueeze(segmento, 0)  # Añadir dimensión de batch (1, num_samples)
    
    # Calcular la longitud relativa del segmento (1.0 ya que es un segmento completo)
    wav_lens = torch.tensor([1.0])

    # Extraer embeddings del segmento
    embeddings = classifier.encode_batch(segmento_tensor, wav_lens)
    
    # Guardar embeddings
    embeddings_np = embeddings.squeeze().detach().cpu().numpy()
    embeddings_list.append(embeddings_np)

# Mostrar los embeddings extraídos de los segmentos
for idx, emb in enumerate(embeddings_list):
    print(f"Embeddings del segmento {idx + 1}: {len(emb)}")
    print(f"Embeddings del segmento {idx + 1}: {emb}")