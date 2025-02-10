# train.py
import torch
import config
from detectors.yolov8_trainer import YOLOTrainer

def main():
    # Donanım kontrolü: GPU varsa kullan, yoksa CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # GPU kullanıyorsanız CUDNN benchmark ile performansı artırabilirsiniz
        torch.backends.cudnn.benchmark = True

    # Eğitim parametreleri
    data_path = config.MODEL_TRAIN_PATH  # Eğitim verileri ve sınıf bilgilerini içeren YAML dosyası
    model_path = config.MODEL_PATH         # Başlangıç için kullanılacak model dosyası
    epochs = 25                          # Eğitim epoch sayısı
    batch = 32                          # Batch size
    imgsz = 640                          # Görüntü boyutu

    print(f"Eğitim başlatılıyor: data={data_path}, epochs={epochs}, batch={batch}, imgsz={imgsz}, device={device}")

    # YOLOTrainer nesnesini oluşturuyoruz (sadece model_path parametresi ile)
    trainer = YOLOTrainer(model_path=model_path)

    # Eğitim işlemini başlatıyoruz
    trainer.train(data=data_path, epochs=epochs, batch=batch, imgsz=imgsz)
    print("Eğitim tamamlandı.")

if __name__ == "__main__":
    main()
