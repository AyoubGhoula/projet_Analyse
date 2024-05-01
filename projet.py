import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Pretraitement
dF = pd.read_csv("cancer_des_poumons.csv")

# Afficher les données et les informations de forme
print(dF)
print(dF.shape)

# Vérifier les valeurs manquantes
print(dF.isnull().sum())

# Remplacer les valeurs manquantes par 0
dF = dF.fillna(0)

# Remplacer les valeurs 0 par la moyenne de chaque colonne
for col in dF.columns:
    if (dF[col] == 0).any():  # Vérifier s'il y a des zéros dans la colonne
        col_mean = dF[col][dF[col] != 0].mean()  # Calculer la moyenne sans tenir compte des zéros
        dF[col] = round(dF[col].replace(0, col_mean), 2)  # Remplacer les zéros par la moyenne

# Transformation
no_numeric_columns = dF.select_dtypes(exclude=['float64', 'int64']).columns
for column in no_numeric_columns:
        mapping_column = {columns: index for index, columns in enumerate(dF[column].unique())}
        dF[column] = dF[column].replace(mapping_column )
        print("La colonne", column, "a été transformer avec les valeurs :", mapping_column)
print(dF)
# Vérifier si les données sont déjà normalisées
numeric_cols = dF.select_dtypes(include='number')
mean_values = numeric_cols.mean()
std_values = numeric_cols.std()
is_normalized = (mean_values.abs() < 1e-10).all() and (std_values.abs() - 1 < 1e-10).all()

if is_normalized:
    print("La base de données est déjà normalisée.")
else:
    # Normaliser les données
    dF[numeric_cols.columns] = (numeric_cols - mean_values) / std_values
    print("Statistiques descriptives après la normalisation :")
    print(dF.describe())
    print("La base de données est normalisée.")

# Calculer la matrice de corrélation
correlation_matrix = numeric_cols.corr()

# Afficher la matrice de corrélation sous forme de heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matrice de Corrélation')
plt.show()

# Identifier les couples de variables les plus corrélées
correlation_threshold = 0.5  # Définir un seuil de corrélation
correlated_pairs = []

# Parcourir la moitié supérieure de la matrice de corrélation pour éviter la redondance
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold: # SPESIFICATIO EN USAN LA MATRICE DE CORRELATION
            correlated_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

# Afficher les couples de variables les plus corrélées
print("Couples de variables les plus corrélées :")
for pair in correlated_pairs:
    print(pair)

# Sélectionner uniquement les colonnes numériques
numeric_cols = dF.select_dtypes(include='number')

# Standardiser les données
scaler = StandardScaler()
dF_standardized = scaler.fit_transform(numeric_cols)

# Appliquer une ACP normée sur les données
pca = PCA(n_components=len(numeric_cols.columns))
pca.fit(dF_standardized)

# Interpréter les valeurs propres
valeurs_propres = pca.explained_variance_ratio_
print("Valeurs propres:", valeurs_propres)

# Calculer les coordonnées des variables originales dans l'espace des composantes principales
variable_saturation = pca.components_.T * np.sqrt(pca.explained_variance_)

# Afficher les coordonnées des variables originales
print("Coordonnées des variables originales dans l'espace des composantes principales:")
print(pd.DataFrame(variable_saturation, columns=[f"PC{i + 1}" for i in range(len(df.columns))], index=df.columns))

# # Déterminer le pourcentage d'inertie à partir de l'éboulis des valeurs propres
# inertie_cumulative = np.cumsum(valeurs_propres)
# plt.plot(range(1, len(inertie_cumulative) + 1), inertie_cumulative, marker='o', linestyle='-')
# plt.xlabel('Nombre de composantes principales')
# plt.ylabel('Pourcentage d\'inertie cumulée')
# plt.title('Éboulis des valeurs propres')
# plt.show()


# # Tracer le cercle de corrélation
# plt.figure(figsize=(8, 8))
# plt.axhline(0, linestyle='--', linewidth=0.5)
# plt.axvline(0, linestyle='--', linewidth=0.5)
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("Cercle de Corrélation des Variables")

# # Tracer le cercle
# circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--')
# plt.gca().add_artist(circle)

# for i in range(len(pca.components_[0])):
#     plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], color='r', alpha=0.5, head_width=0.05)
#     plt.text(pca.components_[0, i]*1.1, pca.components_[1, i]*1.1, numeric_cols.columns[i], color='g', ha='center', va='center')

# plt.grid()
# plt.show()



# # # Configuration de l'algorithme K-means avec 2 clusters
# # kmeans = KMeans(n_clusters=2, random_state=0)
# # clusters = kmeans.fit_predict(dF_standardized)  # Utiliser dF_standardized ici

# # # Ajouter les étiquettes de clusters au DataFrame pour une analyse ultérieure
# # dF['Cluster'] = clusters  # Utiliser dF au lieu de data

# # # Transform data onto the first two principal components
# # pc = pca.transform(dF_standardized)
# # pc1 = pc[:, 0]
# # pc2 = pc[:, 1]

# # # Retrieve the centroids
# # centroids = kmeans.cluster_centers_

# # # Project centroids onto the first two principal components
# # centroid_pc1 = centroids.dot(pca.components_[:2].T)[:, 0]
# # centroid_pc2 = centroids.dot(pca.components_[:2].T)[:, 1]

# # # Re-create the plot
# # plt.figure(figsize=(10, 8))
# # plt.scatter(pc1[clusters == 0], pc2[clusters == 0], s=50, c='lightblue', label='Cluster 0', edgecolors='black')
# # plt.scatter(pc1[clusters == 1], pc2[clusters == 1], s=50, c='orange', label='Cluster 1', edgecolors='black')
# # plt.scatter(centroid_pc1, centroid_pc2, s=200, c='red', label='Centroids', marker='X')
# # plt.title('Visualisation des Clusters et des Centroïdes avec ACP')
# # plt.xlabel('Première Composante Principale')
# # plt.ylabel('Deuxième Composante Principale')
# # plt.legend()
# # plt.grid(True)
# # plt.show()


# # Appliquer l'algorithme de Classification Ascendante Hiérarchique (CAH)
# Z = linkage(dF_standardized, method='ward')

# # Afficher le dendrogramme pour CAH
# plt.figure(figsize=(10, 6))
# dendrogram(Z)
# plt.title('Dendrogramme pour CAH')
# plt.xlabel('Indice de l\'échantillon')
# plt.ylabel('Distance euclidienne')
# plt.show()
# # #Application deKMeans
# # kmeans = KMeans(n_clusters=2)
# # kmeans.fit(dF)

# # #Visualisation
# # colormap=np.array(["red", "green"])
# # plt.scatter(dF.iloc[:,0],dF.iloc[:,1],c=colormap[kmeans.labels_],s=50)
# # plt.show()