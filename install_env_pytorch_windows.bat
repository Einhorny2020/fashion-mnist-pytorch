@echo off
echo 🚀 Création de l'environnement PyTorch pour classification Fashion MNIST
mkdir mon-projet-pytorch
cd mon-projet-pytorch
python -m venv venv
call venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision matplotlib
echo ✅ Environnement prêt ! Tu peux lancer :
echo venv\Scripts\activate
echo python fashion_classifier.py
pause