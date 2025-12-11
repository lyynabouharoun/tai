# Traitement d'images classiques : Amélioration, filtrage et segmentation

## Description du projet
Ce projet explore et implémente des techniques classiques de traitement d'image numérique sur le dataset Kodak. L'objectif est d'améliorer le contraste, de réduire le bruit, d'appliquer des opérations morphologiques et de segmenter les objets d'intérêt. Le projet est entièrement réalisé en Python et utilise des bibliothèques standards pour le traitement d'image et la visualisation.

Le pipeline inclut :
- Conversion en niveaux de gris
- Amélioration du contraste (Histogram Equalization et CLAHE)
- Ajout contrôlé de bruit (gaussien et sel & poivre)
- Filtrage linéaire et non-linéaire (moyenneur, gaussien, médian, bilatéral)
- Opérations morphologiques (érosion, dilatation, ouverture, fermeture, top-hat)
- Segmentation automatique par seuillage d'Otsu
- Calcul des métriques PSNR et SSIM

## Auteurs
- Lyna Bouharoun  
- Massil Mzir  

## Encadrant
- M. Boudaoud

## Dataset
Le projet utilise le **Kodak Lossless True Color Image Suite** : 24 images haute qualité disponibles sur [http://r0k.us/graphics/kodak/](http://r0k.us/graphics/kodak/).

## Structure du projet
```
image-processing-project/
├─ data/kodak/              # Images originales
├─ results/                 # Images et métriques générées
│  ├─ equalized/
│  ├─ noisy/
│  ├─ filtered/
│  ├─ morpho/
│  ├─ segmentation/
│  ├─ histograms/
│  └─ metrics.csv
└─ scripts/
   └─ process_images_extended.py
```

## Bibliothèques utilisées
- **OpenCV (cv2)** : lecture/écriture d'images, égalisation, filtrage, morphologie et seuillage
- **NumPy** : manipulation de données numériques et génération de bruit
- **scikit-image** : métriques PSNR et SSIM
- **Matplotlib** : visualisation et histogrammes
- **pandas** : export des métriques

## Installation
1. Cloner le dépôt :
```bash
git clone https://github.com/lyynabouharoun/tai.git
cd tai
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation
Exécuter le script principal pour traiter toutes les images :
```bash
python scripts/process_images_extended.py
```

Les résultats seront sauvegardés dans le dossier `results/`.

## Résultats
Le projet génère :
- Images améliorées et filtrées
- Histogrammes comparatifs
- Segmentation par seuillage d'Otsu
- Tableau des métriques PSNR et SSIM pour toutes les images

## Lien vers le projet
[GitHub Repository](https://github.com/lyynabouharoun/tai)
