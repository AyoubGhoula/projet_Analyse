import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
    

from scipy.cluster.hierarchy import linkage, dendrogram


df = pd.read_csv("cancer_des_poumons.csv")


##################################################################
########## 3.1. La phase de préparation de données ###############
##################################################################





# 1 : Vérifier s’il existe des observations qui sont manquantes ou NaN, si c’est le cas alors remplacer les valeurs manquantes dans chaque colonne par la moyenne de la variable.

# le nombre des observations dans la base
nombre_observations = df.shape[0]
# le nombre des caractéristiques.
nombre_caract = df.shape[1]
print("nombre_observations : ", nombre_observations)
print("nombre_caractere : ", nombre_caract)




# 2 : 
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    if df[col].isnull().sum() > 0:
        meanValue = df[col].mean()
        df[col].fillna(meanValue, inplace=True)

# print(df)







# Phase II : Transformations :

# Question 1 : transformer chaine en entier
colonnes_number = df.select_dtypes(include=['float64', 'int64']).columns
colonnes_string = df.select_dtypes(exclude=['float64', 'int64']).columns
mappingCols = {}
for col in colonnes_string :
    unique_values = df[col].unique()
    
    for index, value in enumerate(unique_values) :
        print(value)
        df[col] = df[col].replace(value, index)

# print(df)





# normaliser les valeurs
# Question 2
mean = df.mean()
std = df.std()
print("Mean of DF : ", mean)
print("Standard Diviation of DF : ", std)
for col in df.columns :
    df[col] = (df[col] - mean[col]) / std[col]

mean = df.mean()
std = df.std()
print("New Mean of DF : ", round(mean, 2))
print("New Standard Diviation of DF : ", std)






# afficher la matrice de corrélation 
# Question 3
corr_matrix = df.corr()
# print(corr)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de Corrélation")
plt.show()






### ANALYSE ::

"""
Les variables "GENDER" et "CHEST PAIN" ont une corrélation de -0.36, ce qui indique une forte corrélation négative. Cela suggère une relation où le genre et la douleur thoracique ont tendance à varier de manière opposée.
Les variables "GENDER" et "ALCOHOL CONSUMING" ont une corrélation de -0.45, ce qui indique une forte corrélation négative. Cela suggère une relation où le genre et la consommation d'alcool ont tendance à varier de manière opposée.
Les variables "YELLOW FINGERS" et "ANXIETY" ont une corrélation de 0.57, ce qui indique une forte corrélation positive. Cela suggère une relation où la présence de doigts jaunes et l'anxiété ont tendance à varier de manière similaire.
Les variables "YELLOW FINGERS" et "SWALLOWING DIFFICULTY" ont une corrélation de 0.35, ce qui indique une corrélation positive mais moins forte. Cela suggère une relation où la présence de doigts jaunes et les difficultés à avaler peuvent varier ensemble mais avec moins de force.
Les variables "ANXIETY" et "SWALLOWING DIFFICULTY" ont une corrélation de 0.49, ce qui indique une forte corrélation positive. Cela suggère une relation où l'anxiété et les difficultés à avaler ont tendance à varier de manière similaire.
Les variables "PEER PRESSURE" et "SWALLOWING DIFFICULTY" ont une corrélation de 0.37, ce qui indique une corrélation positive mais moins forte. Cela suggère une relation où la pression des pairs et les difficultés à avaler peuvent varier ensemble mais avec moins de force.
Les variables "FATIGUE" et "SHORTNESS OF BREATH" ont une corrélation de 0.44, ce qui indique une forte corrélation positive. Cela suggère une relation où la fatigue et le souffle court ont tendance à varier de manière similaire.
Les variables "ALLERGY" et "ALCOHOL CONSUMING" ont une corrélation de 0.34, ce qui indique une corrélation positive mais moins forte. Cela suggère une relation où les allergies et la consommation d'alcool peuvent varier ensemble mais avec moins de force.
Les variables "WHEEZING" et "COUGHING" ont une corrélation de 0.37, ce qui indique une corrélation positive mais moins forte. Cela suggère une relation où les sifflements et la toux peuvent varier ensemble mais avec moins de force.
Les variables "ALCOHOL CONSUMING" et "CHEST PAIN" ont une corrélation de 0.33, ce qui indique une corrélation positive mais moins forte. Cela suggère une relation où la consommation d'alcool et la douleur thoracique peuvent varier ensemble mais avec moins de force.

"""

# Exemple de données pour la courbe
corr_threshold = 0.5  # Définir un seuil de corrélation (par exemple, 0.5)

highly_correlated_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > corr_threshold:
            pair = (corr_matrix.columns[i], corr_matrix.columns[j])
            highly_correlated_pairs.append(pair)

print("Couples de variables les plus corrélées :")
for pair in highly_correlated_pairs:
    print(pair)


###########################################################################
############ 3.2. La phase d’extraction des caractéristiques ##############
###########################################################################



"""
REMARQUE ::
Variance : 
En ACP, la variance est utilisée pour évaluer la quantité d'information contenue dans chaque variable.
Les variables avec une variance élevée contribuent davantage à la création des composantes principales
car elles ont plus d'impact sur la variation des données.
==> La variance mesure la dispersion des données dans chaque dimension (variable) .
Valeur Propre :
Les valeurs propres sont utilisées pour déterminer l'importance de chaque composante principale dans la représentation
des données.
Les composantes principales avec des valeurs propres élevées expliquent une grande partie
de la variation des données et sont donc considérées comme importantes.
==> Les valeurs propres mesurent la quantité de variance expliquée par chaque composante principale dans l'ACP.
"""

# Question 1 :: Appliquer sur la base une ACP normée. Interpréter les valeurs propres.




# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Appliquer l'ACP
acp = PCA()
X_acp = acp.fit_transform(X_scaled)

# # Afficher tous les composants principaux
# num_components = len(acp.components_)

# for i in range(num_components):
#     plt.figure(figsize=(8, 6))
#     plt.bar(range(len(df.columns)), acp.components_[i])
#     plt.xlabel('Variables')
#     plt.ylabel(f'PC{i+1}')
#     plt.title(f'Composant Principal {i+1}')
#     plt.xticks(range(len(df.columns)), df.columns, rotation=90)
#     plt.grid(True)
#     plt.show()






# Create a DataFrame to store the results
components_df = pd.DataFrame(columns=['Composante Principale', 'Valeur Propre', 'Pourcentage', 'Pourcentage Cumulé'])

# Calculate the percentage of variance explained and cumulative percentage
explained_variance_ratio = acp.explained_variance_ratio_
cumulative_percentage = np.cumsum(explained_variance_ratio) * 100

# Fill in the DataFrame with the results
for i, (ev, cp) in enumerate(zip(acp.explained_variance_, cumulative_percentage)):
    components_df.loc[i] = [f'CP{i+1}', ev, explained_variance_ratio[i] * 100, cp]

# Display the table
print(components_df)





## Afficher les valeurs propres
valeursPropres = acp.explained_variance_
print("Valeurs propres :", valeursPropres)

# Créer la figure
plt.figure(figsize=(10, 6))

# Tracer la courbe des valeurs propres avec plot
plt.plot(range(1, len(valeursPropres) + 1), valeursPropres, marker='o', linestyle='--', label='Valeurs propres')

# Tracer les barres des valeurs propres avec bar
plt.bar(range(1, len(valeursPropres) + 1), valeursPropres, alpha=0.5, label='Valeurs propres (Barres)')

# Étiqueter les axes et le titre
plt.xlabel('Composantes principales')
plt.ylabel('Valeurs propres')
plt.title('Graphique des valeurs propres')
plt.grid(True)
plt.legend()  # Ajouter une légende pour distinguer les deux représentations

# Afficher la figure
plt.show()





# Question 2 :: Déterminer le pourcentage d’inertie à partir de l’éboulis des valeurs propres. Quelles sont les composantes principales à tirées ?

# Calculer le pourcentage d'inertie expliqué par chaque composante principale
pourcentage_inertie = explained_variance_ratio * 100

# Calculer le pourcentage d'inertie cumulé
pourcentage_inertie_cumule = np.cumsum(pourcentage_inertie)

# Tracer le graphique du pourcentage d'inertie cumulé
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pourcentage_inertie_cumule) + 1), pourcentage_inertie_cumule, marker='o')
plt.xlabel('Composantes principales')
plt.ylabel('Pourcentage d\'inertie cumulé')
plt.title('Pourcentage d\'inertie cumulé')
plt.grid(True)
plt.show()





# Quelles sont les composantes principales à tirées ?
# Les trois premières composantes principales sont à retenir, jusqu'au coude (~ 3 ou 4).

# Calculer les valeurs propres de l'ACP
valeurs_propres = acp.explained_variance_

# Calculer le critère de Kaiser (valeurs propres supérieures à 1)
kaiser_criterion = valeurs_propres > 1

# Trouver le nombre de composantes principales à retenir en utilisant le critère de Kaiser
nombre_composantes_principales = np.sum(kaiser_criterion)

# Imprimer le nombre de composantes principales à retenir selon le critère de Kaiser
print("Nombre de composantes principales à retenir selon le critère de Kaiser :", nombre_composantes_principales)






# Maintenant, vous pouvez utiliser ce nombre de composantes principales pour ajuster votre PCA
acp_kaiser = PCA(n_components=nombre_composantes_principales)
composantes_principales_kaiser = acp_kaiser.fit_transform(df)

# Afficher les valeurs propres associées aux composantes principales retenues selon le critère de Kaiser
valeurs_propres_kaiser = acp_kaiser.explained_variance_
print("Valeurs propres des composantes principales retenues selon le critère de Kaiser :", valeurs_propres_kaiser)

# Créer un DataFrame pour afficher les composantes principales et leurs valeurs propres selon le critère de Kaiser
data_kaiser = {'Composante Principale': [f'CP{i+1}' for i in range(len(valeurs_propres_kaiser))],
               'Valeur Propre': valeurs_propres_kaiser}
df_valeurs_propres_kaiser = pd.DataFrame(data_kaiser)

# Afficher le DataFrame des valeurs propres selon le critère de Kaiser
print("Valeurs propres des composantes principales retenues selon le critère de Kaiser :")
print(df_valeurs_propres_kaiser)



# Comparaison entre les axes :

# Indices des axes choisis
x_axis_index = 2  # Indice de l'axe X
y_axis_index = 4  # Indice de l'axe Y

# Création du graphique pour comparer les axes choisis
plt.figure(figsize=(8, 6))
plt.scatter(composantes_principales_kaiser[:, x_axis_index], composantes_principales_kaiser[:, y_axis_index], marker='o')
plt.xlabel(f'Composante Principale {x_axis_index + 1}')
plt.ylabel(f'Composante Principale {y_axis_index + 1}')
plt.title(f'Comparaison des Composantes Principales {x_axis_index + 1} et {y_axis_index + 1}')
plt.grid(True)
plt.show()





# Calculer la saturation des variables
saturation = np.abs(acp.components_ * np.sqrt(acp.explained_variance_).reshape(-1, 1))

# Afficher les vecteurs de corrélation

CP1 = 1
CP2 = 2

# Créer une figure et un axe
fig, ax = plt.subplots(figsize=(8, 8))

# Afficher les vecteurs de corrélation
for i, (comp_x, comp_y) in enumerate(zip(acp.components_[CP1 - 1], acp.components_[CP2 - 1])):
    ax.arrow(0, 0, comp_x, comp_y, head_width=0.05, head_length=0.1, fc='b', ec='b')
    ax.text(comp_x + 0.05, comp_y + 0.05, df.columns[i], color='b')

# Afficher le cercle de corrélation
plt.grid()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel(f"Composante Principale {CP1}")
plt.ylabel(f"Composante Principale {CP2}")
plt.title("Cercle de Corrélation (Saturation des Variables)")

# Ajuster les limites des axes pour une meilleure lisibilité
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

# Afficher la figure
plt.show()




#####################################################
########### 3.3. La phase de data Mining ############
#####################################################


# Question 1
# Appliquer l'algorithme des K-means pour diviser les données en deux classes.


# Générer des données aléatoires avec 2 clusters
X, y = make_blobs(n_samples=300, centers=2, cluster_std=0.60, random_state=0)

# Appliquer l'algorithme K-means avec 2 clusters
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_acp)



#### Question 2
# Afficher dans un graphe les centroïdes et les données appartenant à chaque classe.

# Obtenir les centroïdes des clusters
centroids = kmeans.cluster_centers_

# Obtenir les labels des clusters
labels = kmeans.labels_

# Afficher les données avec les centroïdes et les clusters colorés
plt.figure(figsize=(8, 6))
plt.scatter(X_acp[:, 0], X_acp[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.legend()
plt.grid(True)
plt.show()


# Question 3
# Appliquer l'algorithme Classification Ascendante Hiérarchique (CAH) pour diviser les données en deux classes.

# Génération de données aléatoires pour l'exemple
np.random.seed(0)
X = np.random.rand(10, 2)  # 10 échantillons avec 2 caractéristiques

# Appliquer l'algorithme de CAH pour diviser les données en deux clusters
Z = linkage(X, method='ward')  # Utilisation de la méthode de liaison ward pour la CAH

# Afficher le dendrogramme
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.xlabel('Échantillons')
plt.ylabel('Distance euclidienne')
plt.title('Dendrogramme CAH')
plt.show()




# Question 4
# Comparer les résultats des deux algorithmes.


# Générer des données aléatoires pour l'exemple
X, _ = make_blobs(n_samples=100, centers=3, random_state=42, cluster_std=1.0)

# Appliquer l'algorithme K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Appliquer l'algorithme CAH
linkage_matrix = linkage(X, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix)
plt.title('Dendrogramme de la Classification Ascendante Hiérarchique (CAH)')
plt.xlabel('Index des échantillons')
plt.ylabel('Distance euclidienne')
plt.show()

# Comparer les résultats des deux algorithmes visuellement
plt.figure(figsize=(14, 5))

# Afficher les clusters K-means
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='red', label='Centroides')
plt.title('Clusters avec K-means')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Afficher les clusters CAH
plt.subplot(1, 2, 2)
dendrogram(linkage_matrix)
plt.title('Clusters avec CAH')
plt.xlabel('Index des échantillons')
plt.ylabel('Distance euclidienne')

plt.tight_layout()
plt.show()


# Afficher les inerties totales


# Calculer l'inertie totale de K-means
inertie_kmeans = kmeans.inertia_


# Calculer l'inertie totale de la CAH
inertie_cah = linkage_matrix[-1, 2]


print(f"Inertie totale de K-means : {inertie_kmeans}")
print(f"Inertie totale de la CAH : {inertie_cah}")