import pandas as pn
import numpy as np
"""La phase de préparation de données"""
df=pn.read_csv("cancer_des_poumons.csv")
print(df)
# Interpréter le jeu de données
nb_observations = df.shape[0]
nb_caracteristiques = df.shape[1]
print("Nombre d'observations dans la base de données :", nb_observations)
print("Nombre de caractéristiques dans la base de données :", nb_caracteristiques)
#remplacer les valeurs manquantes dans chaque colonne par la moyenne de la  variable.
print(df.isnull().sum())
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
for column in numeric_columns:
    if df[column].isnull().values.any():
        mean_value=df[column].mean().round(2)
        df[column]=df[column].fillna(mean_value)
        print("La colonne", column, "a été remplie avec la valeur moyenne:", mean_value)

#transformer les caractéristiques dont les valeurs sont de type chaine de caractères en entier.
no_numeric_columns = df.select_dtypes(exclude=['float64', 'int64']).columns
for column in no_numeric_columns:
        mapping_column = {columns: index for index, columns in enumerate(df[column].unique())}
        df[column] = df[column].replace(mapping_column )
        print("La colonne", column, "a été transformer avec les valeurs :", mapping_column)
print(df)
means = df.mean()
stds = df.std()
print("Moyennes des colonnes :\n", means)
print("Écarts-types des colonnes :\n", stds)



