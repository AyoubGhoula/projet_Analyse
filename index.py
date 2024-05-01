""""""


import pandas as pn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

"""_____________________________________________________________________________"""


"""____La phase de préparation de données_____"""
# ---lire la base de données---
df=pn.read_csv("cancer_des_poumons.csv")
print(df)
# Interpréter le jeu de données
print("Informations sur le jeu de données:")
nb_observations = df.shape[0] #Nombre d'observations dans la base de données
nb_caracteristiques = df.shape[1]#Nombre de caractéristiques dans la base de données
print("Nombre d'observations dans la base de données :", nb_observations)
print("Nombre de caractéristiques dans la base de données :", nb_caracteristiques)
#----remplacer les valeurs manquantes par la moyenne de chaque colonne----
    #--Vérifier s’il existe des observations qui sont manquantes ou NaN--
if df.isnull().values.any():
    print("il y a des valeurs manquantes :")
    print(df.isnull().sum().sum())
    #--remplacer les valeurs manquantes par la moyenne de chaque colonne--
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_columns:
    if df[col].isnull().any():
        col_mean = df[col].mean()
        df[col] = df[col].fillna(col_mean)
        print("La colonne", col, "a été remplie avec la valeur moyenne:", col_mean)
#----transformer les caractéristiques dont les valeurs sont de type chaine de caractères en entier----

no_numeric_columns = df.select_dtypes(exclude=['float64', 'int64']).columns
for column in no_numeric_columns:
        mapping_column = {columns: index for index, columns in enumerate(df[column].unique())}
        df[column] = df[column].replace(mapping_column)
        print("La colonne", column, "a été transformer avec les valeurs :", mapping_column)

#----Vérifier si la base est normalisée ou non (centrée-réduite)
# verifier si la base de données est déjà normalisée
means = df.mean()
stds = df.std()
print("Moyennes des colonnes :\n", means)
print("Écarts-types des colonnes :\n", stds)
if means.all() != 0 or stds.all() != 1:
    print("La base de données n'est pas encore normalisée")
    scaler = StandardScaler()
    df_scaled= scaler.fit_transform(df)
    print("La base de données a été normalisée (centrée-réduite)")
    
    """for col in df.columns:  
        df[col] = (df[col] - means[col]) / stds[col]"""
        
else :
    print("La base de données est déjà normalisée")
    

#----Afficher la matrice de corrélation puis analyser les dépendances des variables----
print(df_scaled)
correlation_matrix = df.corr()
print(correlation_matrix)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix,vmin=-1,vmax=1, annot=True, cmap='coolwarm', fmt="0.2f")
#plt.savefig('heatmap.png')
plt.show()

# Parcourir la moitié supérieure de la matrice de corrélation pour éviter la redondance
correlated_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.5: # SPESIFICATIO EN USAN LA MATRICE DE CORRELATION
            correlated_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
print(correlated_pairs)


"""____Analyse en composantes principales (ACP)____"""
#---Appliquer l'ACP sur la base de données normal---
acp = PCA()
df_acp=acp.fit_transform(df_scaled)
print(pn.DataFrame(df_acp))
# Interpréter les valeurs propres.
valeurs_propres = acp.explained_variance_ratio_
plt.title("Histogramme des valeurs propres")
plt.plot(np.arange(1,nb_caracteristiques+1),acp.explained_variance_, color='red')# explained_variance_ pour obtenir les valeurs propres associées à chaque composante principale.
plt.bar(np.arange(1,nb_caracteristiques+1),acp.explained_variance_,color='blue')
plt.ylabel("valeur propre")
plt.xlabel("ordre de valeur propre")
plt.show()
# Calculer les coordonnées des variables originales dans l'espace des composantes principales
variable_saturation = acp.components_.T * np.sqrt(acp.explained_variance_)

# Afficher les coordonnées des variables originales
print("Coordonnées des variables originales dans l'espace des composantes principales:")
print(pn.DataFrame(variable_saturation, columns=[f"PC{i + 1}" for i in range(len(df.columns))], index=df.columns))
explained_variance_ratio = acp.explained_variance_ratio_#L'attribut explained_variance_ratio_ dans scikit-learn est utilisé pour obtenir le pourcentage de variance expliquée par chaque composante principale.

cumulative_variance_ratio = np.cumsum(explained_variance_ratio)


plt.figure(figsize=(10, 6))
plt.bar(np.arange(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100, label='Ratio de variance expliqué', alpha=0.7)
plt.plot(np.arange(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio * 100, marker='o', color='red', label='Ratio de variance expliquée cumulée', linestyle='-', linewidth=2)
plt.xlabel("Numéro de composant")
plt.ylabel("Pourcentage (%)")
plt.title("Pourcentage de variance expliquée et de variance expliquée cumulée")
plt.legend()
plt.grid(True)
plt.show()

# ----Afficher la saturation des variables et tracer le cercle de corrélation. Interpréter.----
# Get the loadings (variable saturations)
loadings = acp.components_.T * np.sqrt(acp.explained_variance_)

# Plot the correlation circle
plt.figure(figsize=(8, 8))
plt.xlim(-1, 1)
plt.ylim(-1, 1)
# plt.xlabel("PC1 ({}%)".format(round(explained_variance_ratio[0] * 100, 2)))
# plt.ylabel("PC2 ({}%)".format(round(explained_variance_ratio[1] * 100, 2)))
plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
plt.scatter(loadings[:, 0], loadings[:, 1], alpha=0.7)

circle = plt.Circle((0, 0), 1, color='b', fill=False)
plt.gca().add_artist(circle)
# Annotate variable names
for i, var in enumerate(df.columns):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1])
    plt.annotate(var, (loadings[i, 0], loadings[i, 1]))

plt.title("Correlation Circle")
plt.grid(True)
plt.show()

"""La phase de data Mining"""
# ---Appliquer l'algorithme des K-means pour diviser les données en deux classes---
kmeans = KMeans(n_clusters=2)
kmeans.fit(df_acp)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.figure(figsize=(8, 6))
plt.scatter(df_acp[:, 0], df_acp[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Première composante principale')
plt.ylabel('Deuxième composante principale')
plt.title('ACP avec KMeans Clustering 2 classes')
plt.legend()
plt.grid(True)
plt.show()


# Appliquer l'algorithme de Classification Ascendante Hiérarchique (CAH)
Z = linkage(df_acp, method='ward')

# Afficher le dendrogramme pour CAH
plt.figure(figsize=(10, 6))
dendrogram(Z,color_threshold=0, above_threshold_color=2)
plt.title('Dendrogramme pour CAH')
plt.xlabel('Indice de l\'échantillon')
plt.ylabel('Distance euclidienne')
plt.show()



