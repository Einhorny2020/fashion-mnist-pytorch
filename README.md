Projet IA : Classification de vêtements avec PyTorch (Fashion MNIST)

Ce projet vous permet d'entraîner un réseau de neurones simple à reconnaître des vêtements à partir d’images en noir et blanc grâce au dataset **Fashion MNIST**.

Objectifs

- Comprendre la structure d’un projet PyTorch de classification image
- Charger, normaliser et traiter des données visuelles
- Construire un modèle `nn.Sequential` simple
- Entraîner et tester le modèle
- Visualiser les prédictions

Exécuter le projet en ligne

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Einhorny2020/fashion-mnist-pytorch/blob/main/fashion_classifier.ipynb)

Exécution locale

Pour macOS :
```bash
chmod +x install_env_pytorch_mac.sh
./install_env_pytorch_mac.sh
source venv/bin/activate
python fashion_classifier.py
```

Pour Windows :
```cmd
install_env_pytorch_windows.bat
venv\Scripts\activate
python fashion_classifier.py
```

Dépendances

- Python 3.10
- torch
- torchvision
- matplotlib

Exemple de résultat

Le script affiche 5 images de vêtements avec leur **classe réelle** et la **prédiction du modèle**.
