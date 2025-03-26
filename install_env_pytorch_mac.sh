#!/bin/bash
echo "🚀 Création de l'environnement PyTorch pour classification Fashion MNIST"
mkdir -p mon-projet-pytorch
cd mon-projet-pytorch
if ! command -v /opt/homebrew/bin/python3.10 &> /dev/null; then
    echo "❗ Python 3.10 introuvable. Installation via Homebrew..."
    brew install python@3.10
fi
/opt/homebrew/bin/python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision matplotlib
echo "✅ Installation terminée. Lance :"
echo "source venv/bin/activate && python fashion_classifier.py"