# Gıda Görüntüsü Sınıflandırması / Food Image Classification with MATLAB

Bu proje, farklı gıda görsellerini tanımak için **MATLAB tabanlı görüntü işleme ve makine öğrenmesi** algoritmalarını kullanan bir sınıflandırma sistemidir.

This project is a MATLAB-based food classification system using **image processing and machine learning techniques** to identify food types from images.

---

## Proje Hakkında / About the Project

Bu projede, farklı yemek kategorilerine ait görseller sınıflandırılmak üzere analiz edilmiştir. Görüntü işleme adımlarında Gabor filtreleme, Histogram of Oriented Gradients (HOG) ve kenar tespiti gibi yöntemler; sınıflandırma aşamasında ise Support Vector Machines (SVM) kullanılmıştır.

The project classifies food images into categories using steps such as Gabor filtering, Histogram of Oriented Gradients (HOG), edge detection, and SVM-based classification.

---

## Kullanılan Teknolojiler / Technologies

- MATLAB R2023a
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox
- Pretrained classifier `.mat` file

---

## Kullanılan Algoritmalar / Algorithms Used

-  **Gabor Filtering** – Texture-based feature extraction
-  **HOG (Histogram of Oriented Gradients)** – Shape and edge-based descriptor
-  **SVM (Support Vector Machine)** – Multi-class classifier
-  **Preprocessing** – Resizing, grayscale conversion, edge detection

---

## Veri Seti / Dataset

-  `Train_Images/` – Eğitim seti
-  `Test_Images/` – Test seti
-  Sınıflar: Pizza, Burger, Salad, Sandwich, Pasta (Örnek)
- Görseller `.jpg` formatındadır ve farklı açılardan çekilmiş gıda görüntülerini içerir.

---

## Çalışma Akışı / Workflow

1. Görsel giriş olarak alınır (`uigetfile`)
2. Görsel ön işleme: boyutlandırma, gri tonlama
3. Gabor ve HOG ile öznitelik çıkarımı yapılır
4. Daha önce eğitilmiş SVM modeli (`trainedClassifier.mat`) ile tahmin yapılır
5. Tahmin sonucu GUI veya konsolda görüntülenir

---

## Başarım / Performance

- Ortalama doğruluk: **%87–92** (5 sınıf üzerinde test edilmiştir)
- Karışıklık matrisi kullanılarak sınıflandırma başarımı değerlendirilmiştir


---
