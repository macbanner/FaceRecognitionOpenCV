import cv2
import numpy as np
from PIL import Image
import os


dataset_yolu = 'dataset'
trainer_yolu = 'trainer'

# Eğitici oluştur.
# LBPH (Local Binary Patterns Histograms) yüz tanıyıcı kullanıyoruz
recognizer = cv2.face.LBPHFaceRecognizer.create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# Görüntüleri ve etiket verilerini almak için metot.
def getImagesAndLabels(path):
    imagePaths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                imagePaths.append(os.path.join(root, file))

    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        try:
            PIL_img = Image.open(imagePath).convert('L')  # Gri tonlamaya çevir
            img_numpy = np.array(PIL_img, 'uint8')

            # Dosya adından ID'yi ve ismi çıkar
            # Örnek: dataset/User.1.AhmetSalih/1.jpg -> id = 1
            filename = os.path.basename(imagePath)  # 1.jpg
            user_info_part = os.path.basename(os.path.dirname(imagePath))  # User.1.AhmetSalih

            id_str = user_info_part.split('.')[1]
            id = int(id_str)

            # Görüntüden yüzleri tespit et
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        except Exception as e:
            print(f"Hata: {imagePath} işlenirken sorun oluştu - {e}")
            continue

    return faceSamples, ids


print("\n [BİLGİ] Yüzler eğitiliyor. Lütfen bekleyin ...")

faces, ids = getImagesAndLabels(dataset_yolu)

if not faces:
    print("[HATA] Eğitilecek yüz bulunamadı.")
    exit()

recognizer.train(faces, np.array(ids))

# Modeli trainer/trainer.yml dosyasına kaydet
if not os.path.exists(trainer_yolu):
    os.makedirs(trainer_yolu)
recognizer.write(os.path.join(trainer_yolu, 'trainer.yml'))

print(f"\n [BİLGİ] {len(np.unique(ids))} farklı yüz eğitildi. Programdan çıkılıyor.")