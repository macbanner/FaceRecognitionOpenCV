import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
# Eğitilmiş modeli yükle
try:
    recognizer.read('trainer/trainer.yml')
except cv2.error as e:
    print(f"[HATA] trainer.yml dosyası yüklenemedi: {e}")
    print("Lütfen önce modeli eğitin.")
    exit()

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# ID'leri isimlere eşleştirmek için bir sözlük/liste oluştur
# Bu bilgiyi 'dataset' klasöründeki klasör adlarından alabiliriz
names = ['Bilinmeyen']  # ID 0 için varsayılan
dataset_yolu = 'dataset'
try:
    user_folders = [d for d in os.listdir(dataset_yolu) if os.path.isdir(os.path.join(dataset_yolu, d))]
    max_id = 0
    temp_names = {}

    for folder_name in user_folders:
        try:
            # Klasör adı formatı: User.ID.İsim
            parts = folder_name.split('.')
            if len(parts) >= 3 and parts[0] == 'User':
                id_val = int(parts[1])
                name_val = " ".join(parts[2:])  # İsimde boşluk olabilir
                temp_names[id_val] = name_val
                if id_val > max_id:
                    max_id = id_val
        except ValueError:
            print(f"[UYARI] Klasör adı '{folder_name}' geçerli formatta değil (User.ID.İsim). Atlanıyor.")
            continue

    # names listesini doldur
    if temp_names:  # en az bir kullanıcı varsa
        names = ['Bilinmeyen'] * (max_id + 1)  # max_id'ye kadar Bilinmeyen ile doldur
        for id_val, name_val in temp_names.items():
            if 0 <= id_val < len(names):  # Geçerli bir index mi
                names[id_val] = name_val
            else:  # Kullanıcı id names içinde yoksa.
                print(f"[UYARI] ID {id_val} için geçersiz indeks. Bu kullanıcı atlanıyor.")

except FileNotFoundError:
    print(f"[HATA] 'dataset' klasörü bulunamadı. Lütfen önce veri toplayın ve eğitin.")
    exit()
except Exception as e:
    print(f"[HATA] İsimler okunurken bir sorun oluştu: {e}")
    pass


kamera = cv2.VideoCapture(0)
kamera.set(3, 640)  # Genişlik
kamera.set(4, 480)  # Yükseklik

# Minimum pencere boyutu (tespit edilecek yüzün minimum boyutu)
minW = 0.1 * kamera.get(3)
minH = 0.1 * kamera.get(4)

while True:
    ret, cerceve = kamera.read()
    if not ret:
        print("Kamera okunamadı.")
        break

    gri = cv2.cvtColor(cerceve, cv2.COLOR_BGR2GRAY)

    yuzler = faceCascade.detectMultiScale(
        gri,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in yuzler:
        cv2.rectangle(cerceve, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gri[y:y + h, x:x + w])

        # Güvenilirlik değerini kontrol et (0 mükemmel eşleşme)
        # LBPH algoritması için 100'ün altındaki değerleri baz alıyorum.
        # Daha düşük değerler => daha iyi eşleşme.
        if confidence < 70:  # Eşik değeri 70 olsun.
            if 0 < id < len(names):  # ID'nin geçerli aralıkta mı değil mi kontrol
                kisi_ismi = names[id]
            else:
                kisi_ismi = "Bilinmeyen (ID Dışı)"
            # Yüzde olarak güvenilirlik (100 - confidence)
            guvenilirlik_yuzdesi = f"  {round(100 - confidence)}%"
        else:
            kisi_ismi = "Bilinmeyen"
            guvenilirlik_yuzdesi = f"  {round(100 - confidence)}%"

        cv2.putText(cerceve, str(kisi_ismi), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(cerceve, str(guvenilirlik_yuzdesi), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('Yüz Tanıma', cerceve)

    k = cv2.waitKey(10) & 0xff  # Çıkış için ESC
    if k == 27:
        break

print("\n [BİLGİ] Programdan çıkılıyor.")
kamera.release()
cv2.destroyAllWindows()