#Importer les librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

#importer dataset
Data= pd.read_csv('bill_authentification.csv')
x=Data.iloc[:,:-1].values
y=Data.iloc[:,-1].values

#Apercu des données
#Data.head()
Data.shape

from os import X_OK
#Diviser les donées en jeu d'entrainement et jeu de test
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20)

#Créer le modèle SVM SVC par defult le kernel est RBM ici on a mis lineaire
classifier=SVC(kernel='linear', random_state=0)
classifier.fit(X_train,y_train)

#Prédiction sur le Test set
y_pred=classifier.predict(X_test)

#Matrice de confusion
cm=confusion_matrix(y_test,y_pred)


#Rapport de classification
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

# Remplacer par ton nom de fichier si différent
df = pd.read_csv("bill_authentification.csv")
# Afficher un aperçu des données
print(df.head())

# 4. Tracer un nuage de points avec deux premières colonnes (ex: amount vs merchant_category)
plt.figure(figsize=(10,6))
colors = ['red' if label == 0 else 'green' for label in df['auth']]  # colorier selon la classe

plt.scatter(df['amount'], df['merchant_category'], c=colors, alpha=0.6)
plt.title("Visualisation des données (amount vs merchant_category)")
plt.xlabel("Amount")
plt.ylabel("Merchant Category")
plt.grid(True)
plt.show()

#Codage Remplacer les fonctions de sklearn.model_selection.train_test_split, sklearn.svm.SVC, et sklearn.metrics.confusion_matrix / classification_report par des formules mathématiques codées à la main est un défi classique pour mieux comprendre l’apprentissage automatique. Voici comment le faire en pur Python et NumPy, sans utiliser Scikit-learn.
#Codage Remplacer les fonctions de sklearn
import pandas as pd

#Visualisation de données
import matplotlib.pyplot as plt

# 1. Charger les données
df = pd.read_csv("bill_authentification.csv")

# 2. Convertir en tableaux numpy
#value converti Dataframe en tableau Numpy
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#value converti Dataframe en tableau Numpy

# 3. Normaliser les étiquettes en -1 et +1 (requis pour SVM)
# important pour la fonction de hinge
y = np.where(y == 0, -1, 1)

# 4. Diviser les données manuellement (80% entraînement, 20% test)
class SimpleSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate         # Taux d’apprentissage (learning rate)
        self.lambda_param = lambda_param  # Paramètre de régularisation
        self.n_iters = n_iters          # Nombre d’itérations d’entraînement
        self.w = None                  # Poids (vecteur)
        self.b = None                  # Biais (scalaire)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)  # Initialisation des poids à zéro
        self.b = 0                    # Initialisation du biais à zéro

        # Boucle d’optimisation (descente de gradient)
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Calcul de la condition SVM (marge ≥ 1)
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    # Si marge correcte, seulement régularisation des poids
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Sinon, mise à jour avec contribution du terme hinge loss
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b  # Calcul du score linéaire
        return np.sign(approx)                # Retourne -1 ou +1 selon le signe (prédiction)


# 6. Entraîner le modèle
model = SimpleSVM()
model.fit(X_train, y_train)

# 7. Prédictions
y_pred = model.predict(X_test)

# 8. Reconvertir -1/+1 en 0/1 pour comparaison
y_test_binary = np.where(y_test == -1, 0, 1)
y_pred_binary = np.where(y_pred == -1, 0, 1)

# 9. Matrice de confusion manuelle
def compute_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1)) #(Vrai +) : nombre d’ex correctement prédits + (1)
    TN = np.sum((y_true == 0) & (y_pred == 0)) #(Vrai -) : nombre d’ex correctement prédits - (0).
    FP = np.sum((y_true == 0) & (y_pred == 1)) #(Faux +) : ex - prédit comme + (faux +).
    FN = np.sum((y_true == 1) & (y_pred == 0)) #(Faux -) : ex + prédit comme - (faux -).
    return np.array([[TN, FP], [FN, TP]])

# 10. Rapport de classification manuel
def classification_metrics(cm): #entrée cm, une matrice de confusion 2x2
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    accuracy = (TP + TN) / (TP + TN + FP + FN) # Exactitude proportion de bonnes prédictions (+ et -) sur l’ensemble des prédictions.
    precision = TP / (TP + FP) if (TP + FP) else 0 #parmi tous les éléments prédits positifs, quelle proportion est réellement positive.
    recall = TP / (TP + FN) if (TP + FN) else 0 #sensibilité : parmi tous les vrais + réels, quelle proportion a été correctement identifiée.
    #Score F1 : moyenne harmonique entre précision et rappel, donne un équilibre entre les deux.
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return accuracy, precision, recall, f1

#On calcule la matrice de confusion entre les vraies étiquettes et les prédictions.
cm = compute_confusion_matrix(y_test_binary, y_pred_binary)

#On calcule les métriques (accuracy, précision, rappel, F1) à partir de cette matrice.
acc, prec, rec, f1 = classification_metrics(cm)

# 11. Affichage
print("Matrice de confusion :\n", cm)
print(f"Exactitude (Accuracy): {acc:.2f}")
print(f"Précision: {prec:.2f}")
print(f"Rappel (Recall): {rec:.2f}")
print(f"F1-score: {f1:.2f}")