# config.py

import numpy as np

# Video ve model konfigürasyonları
VIDEO_INPUT_PATH = r"C:\Users\mbaki\Desktop\TeknoFest\deneme-1\data\videos\istockphoto-1278725068-640_adpp_is.mp4"
VIDEO_OUTPUT_PATH = r"C:\Users\mbaki\Desktop\TeknoFest\deneme-1\data\outputs\stokholm.mp4"

MODEL_PATH = r"C:\Users\mbaki\Desktop\TeknoFest\deneme-1\models\yolov8n.pt"
MODEL_TRAINED_PATH = r"C:\Users\mbaki\Desktop\TeknoFest\deneme-1\src\runs\detect\train2\weights\best.pt"
MODEL_TRAIN_PATH = r"C:\Users\mbaki\Desktop\TeknoFest\deneme-1\data\datasets\v2\data.yaml"

# Eğer video bilgisinden fps alınamazsa kullanılacak varsayılan değer
FPS = 24

# Perspektif dönüşüm için kaynak noktalar (resize sonrası)
SOURCE = np.array([
    [0, 0],       # Sol üst köşe
    [768, 0],     # Sağ üst köşe
    [768, 432],   # Sağ alt köşe
    [0, 432]      # Sol alt köşe
])

# Hedef (dönüşüm) boyutları
TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1]
])

# Video çıktı boyutu (width, height) – orijinal kodda hesaplanan boyut (1152, 648)
VIDEO_SIZE = (768, 432)

# Drone'un yüksekliği ve kamera parametrelerine göre kalibrasyon katsayısı
# Örneğin: Drone 50 m yükseklikteyse ve ölçümlerinizde 1 piksel yaklaşık 0.1 metreye denk geliyorsa
PIXEL_TO_METER = 0.1
