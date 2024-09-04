import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
import streamlit.components.v1 as components

# =============================================================================================================================================

df = pd.read_csv('maladie_observations.csv')

# Calcul de la moyenne de chaque colonne
mean_values = df.mean()

# Remplissage des valeurs NaN par la moyenne de chaque colonne
df.fillna(mean_values, inplace=True)

encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

y = df['label']
x = df.drop('label', axis=1)

# =============================================================================================================================================

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=50)

# Création d'un modèle avec SVM

# gamma='auto'  kernel='linear'
svm_model = SVC(C=1.0, kernel='linear', gamma='scale', degree=3, coef0=0.0, shrinking=True,
                probability=False, tol=1e-3, cache_size=200, class_weight=None, verbose=False,
                max_iter=1000, decision_function_shape='ovr', break_ties=False, random_state=None)
svm_model.fit(x_train, y_train)
prediction = svm_model.predict(x_test)

# Créer une SVM avec un noyau gaussien de paramètre gamma=0.01
svm_rbf = SVC(kernel='rbf', gamma=0.01)
svm_rbf.fit(x_train, y_train)

# Prédire sur le jeu de test
y_test_pred = svm_rbf.decision_function(x_test)

# =============================================================================================================================================

# Construire la courbe ROC
fpr, tpr, _ = metrics.roc_curve(y_test, prediction)
auc = metrics.auc(fpr, tpr)

# =============================================================================================================================================

# Générer des données synthétiques
x_syn, y_syn = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Diviser les données en ensembles d'entraînement et de test
x_train_syn, x_test_syn, y_train_syn, y_test_syn = train_test_split(x_syn, y_syn, test_size=0.2, random_state=42)

# Créer un classificateur d'arbre de décision
base_clf = DecisionTreeClassifier()

# Créer un classificateur de bagging avec des arbres de décision
bagging_clf = BaggingClassifier(base_estimator=base_clf, n_estimators=50, random_state=42)
bagging_clf.fit(x_train_syn, y_train_syn)

# Prédire les étiquettes pour l'ensemble de test
y_pred_syn = bagging_clf.predict(x_test_syn)
accuracy_syn = accuracy_score(y_test_syn, y_pred_syn)

# Visualiser les résultats
# Assure-toi que x_syn est bien un tableau 2D (n_samples, n_features)
if x_syn.ndim == 2:
    x_min, x_max = x_syn[:, 0].min() - 1, x_syn[:, 0].max() + 1
    y_min, y_max = x_syn[:, 1].min() - 1, x_syn[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = bagging_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(x_train_syn[:, 0], x_train_syn[:, 1], c=y_train_syn, marker='o', edgecolors='k', label='Train')
    plt.scatter(x_test_syn[:, 0], x_test_syn[:, 1], c=y_test_syn, marker='s', edgecolors='k', label='Test')
    plt.xlabel('x_train')
    plt.ylabel('y_train')
    plt.title('Bagging Decision Boundary')
    plt.legend()
    plt.show()
else:
    st.write("Les données synthétiques générées ne sont pas au bon format pour la visualisation 2D.")

# =============================================================================================================================================

# Titre du tableau de bord
st.header(":green[Présentation des données interactives Covid-19 avec prédiction]")
st.markdown("""
* Interaction des symptômes
* Représentation interactive des symptômes
* Prédiction
""")

with st.sidebar:
    menu = st.selectbox('Menu', ["Tableau de bord", "Temperature", "Pouls", "Glycemie", "Oxygene", "Tension"])

# Affichage en fonction de la sélection
if menu == "Tableau de bord":
    st.write(":green[Bienvenue sur l'application d'analyse descriptive du dataset.]")
    st.write(":blue[Ce tableau de bord montre un exemple de visualisation de données.]")

# Sélection interactive des colonnes
st.header(":red[Sélection Interactive]")
option = st.selectbox(
    'Choisissez une colonne à afficher',
    ('temperature', 'pouls', 'oxygene', 'glycemie', 'tension', 'label')
)
st.write('Vous avez sélectionné:', option)

# Filtrer les données en fonction de la sélection
filtered_df = df[[option]]
st.line_chart(filtered_df)

# Charger le dataset à nouveau
@st.cache_data
def load_data():
    df = pd.read_csv('maladie_observations.csv')
    return df

df = load_data()

# Menu de navigation
menu = ["Accueil", "Voir les données", "Statistiques descriptives"]
choix = st.sidebar.selectbox("Navigation", menu)

# Affichage en fonction de la sélection
if choix == "Accueil":
    st.write(":green[Bienvenue sur l'application d'analyse descriptive du dataset.]")
elif choix == "Voir les données":
    st.header("Voir les données")
    st.write("Voici un aperçu du dataset :")
    st.write(df.head())
elif choix == "Statistiques descriptives":
    st.header("Statistiques descriptives")

    # Sélection des colonnes numériques
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    col = st.selectbox("Sélectionnez une colonne pour voir les statistiques descriptives", numeric_cols)

    # Afficher les statistiques descriptives
    st.write(df[col].describe())

    # Afficher la distribution de la colonne sélectionnée
    st.subheader(f"Distribution de {col}")

    # Créer deux colonnes dans l'interface
    col1, col2 = st.columns(2)

    # Affichage pour les colonnes float64
    with col1:
        st.subheader("Statistiques descriptives (Line Chart)")
        col_float = st.selectbox("Sélectionnez une colonne float64", numeric_cols)

        # Afficher la distribution de la colonne sélectionnée
        st.line_chart(df[col_float])

    # Affichage pour les colonnes int64
    with col2:
        st.subheader("Statistiques descriptives (Bar Chart)")
        col_int = st.selectbox("Sélectionnez une colonne int64", numeric_cols)

        # Afficher la distribution de la colonne sélectionnée
        st.bar_chart(df[col_int].value_counts().sort_index())

# Prédiction

st.title(":blue[Prédiction cas de Covid]")

# Entrer les valeurs par l'utilisateur
st.sidebar.subheader("Entrez les valeurs")
tpe = st.sidebar.slider("Température", min_value=1.0, max_value=10.0, value=6.0)
oxg = st.sidebar.slider("Oxygène", min_value=1.0, max_value=10.0, value=6.0)
pls = st.sidebar.slider("Pouls", min_value=1.0, max_value=10.0, value=6.0)
glm = st.sidebar.slider("Glycémie", min_value=1.0, max_value=10.0, value=6.0)
tsn = st.sidebar.slider("Tension", min_value=1.0, max_value=10.0, value=6.0)

# Faire la prédiction
input_data = np.array([[tpe, oxg, pls, glm, tsn]])
prediction = svm_model.predict(input_data)[0]
st.write(f"La prédiction pour les valeurs données est: {prediction}")

# Footer
st.sidebar.text("© 2024 Mamadou Mbow * Machine Learning")