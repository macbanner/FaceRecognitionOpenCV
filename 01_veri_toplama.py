import cv2
import os

#Kamera açılır.
kamera = cv2.VideoCapture(0)
kamera.set(3, 640)  # Genişlik
kamera.set(4, 480)  # Yükseklik

# Yüz tespiti için Haar Cascade yükle
yuz_tespit_modeli = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Her kişi için bir sayısal ID girin
kisi_id = input('\n Bir kullanıcı ID girin (sayı) ve Enter tuşuna basın ==>  ')
kisi_isim = input('\n Bir kullanıcı ismi girin ve Enter tuşuna basın ==>  ')

print("\n [BİLGİ] Yüz yakalama başlıyor. Kameraya bakın ve bekleyin ...")
# Yakalanan yüz sayısı için sayaç.
sayac = 0


dataset_yolu = 'dataset'
if not os.path.exists(dataset_yolu):
    os.makedirs(dataset_yolu)

kisi_klasoru = os.path.join(dataset_yolu, f"User.{str(kisi_id)}.{kisi_isim}")
if not os.path.exists(kisi_klasoru):
    os.makedirs(kisi_klasoru)
else:
    print(f"[UYARI] {kisi_klasoru} zaten mevcut. İçine ekleme yapılacak.")

while (True):
    ret, cerceve = kamera.read()
    if not ret:
        print("Kamera okunamadı!")
        break

    gri = cv2.cvtColor(cerceve, cv2.COLOR_BGR2GRAY)
    yuzler = yuz_tespit_modeli.detectMultiScale(gri, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in yuzler:
        cv2.rectangle(cerceve, (x, y), (x + w, y + h), (255, 0, 0), 2)
        sayac += 1
        # Yakalanan görüntüyü dataset klasörüne kaydet
        dosya_yolu = os.path.join(kisi_klasoru, str(sayac) + ".jpg")
        cv2.imwrite(dosya_yolu, gri[y:y + h, x:x + w])
        cv2.imshow('Yüz Yakalama', cerceve)

    k = cv2.waitKey(100) & 0xff  # Çıkış için ESC veya 100ms bekle
    if k == 27:  # ESC tuşu
        break
    elif sayac >= 50:  # 50 yüz örneği al ve dur
        break

print(f"\n [BİLGİ] {sayac} adet yüz örneği {kisi_klasoru} klasörüne kaydedildi.")
print(" [BİLGİ] Programdan çıkılıyor.")
kamera.release()
cv2.destroyAllWindows()