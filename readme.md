# 🧠 Deep Learning Framework from Scratch (NumPy)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.8%20%7C%203.11-blue)
![Framework](https://img.shields.io/badge/framework-pure__numpy-orange)

Ce dépôt contient un **moteur de Deep Learning complet codé intégralement en pur NumPy**, sans l'aide de bibliothèques comme PyTorch ou TensorFlow. L'objectif de ce projet est de démontrer une bonne compréhension des mathématiques sous-jacentes aux réseaux de neurones : passes avant/arrière (backpropagation), gestion dynamique des graphes de couches, mécanismes de régularisation et optimisation stochastique.

Le framework est validé et évalué sur deux cas d'école : une tâche de **classification multi-classe (MNIST)** et une tâche de **régression non-linéaire (California Housing)**.

---

## 🚀 Fonctionnalités du Framework

L'architecture modulaire du framework reproduit fidèlement la philosophie de l'API `torch.nn` :

* **Gestion des Couches :** Structure `SequentialNetwork` orchestrant les passes `forward` et `backward`.
* **Couches Linéaires :** `LinearLayer` avec initialisation des poids de type **He (Kaiming) Normal** et support natif de la régularisation **L2 (Weight Decay)**.
* **Fonctions d'Activation :** Implémentation vectorisée de `ReLU` et `Softmax`.
* **Régularisation :** Couche de `InvertedDropout` dynamique (comportement différencié automatique entre `training=True` et `training=False`).
* **Optimiseur :** Descente de Gradient Stochastique avec **Momentum (0.9)**.
* **Fonctions de Perte :** `MSELoss` pour la régression et `CrossEntropyLoss` (sécurisée numériquement) pour la classification.

---

## 📊 Résultats & Performances Évaluées

Le modèle a été évalué selon une méthodologie scientifique stricte sur un **Jeu de Test (Test Set)** totalement isolé, garantissant une mesure parfaite de sa capacité de généralisation.

### 1. Classification Multi-Classe (MNIST)
* **Architecture :** Perceptron Multicouche (MLP) dense (~109 000 paramètres).
* **Optimisation :** Taux d'apprentissage $\alpha = 0.02$, Batch Size = 256.

**Rapport de Performance (Test Set) :**
```text
=======================================================
        RAPPORT DE PERFORMANCE DE CLASSIFICATION        
=======================================================
Classe     | Précision  | Rappel     | F1-Score  
-------------------------------------------------------
Chiffre 0  | 0.9701     | 0.9873     | 0.9786    
Chiffre 1  | 0.9731     | 0.9895     | 0.9813    
...
Chiffre 8  | 0.9740     | 0.9280     | 0.9504    
Chiffre 9  | 0.9483     | 0.9625     | 0.9553    
-------------------------------------------------------
MOYENNE (Macro) | 0.9675     | 0.9667     | 0.9670    
=======================================================


=============================================
        RAPPORT DE PERFORMANCE RÉGRESSION        
=============================================
Erreur Absolue Moyenne (MAE)  : 0.3455
Racine de l'Erreur au Carré (RMSE) : 0.4927
Coefficient de Détermination (R²) : 0.7553
=============================================