import sys
from PyQt5.QtWidgets import QApplication, QFileDialog , QMainWindow,QHeaderView
from PyQt5.uic import loadUi
from PyQt5.QtGui import QStandardItemModel, QStandardItem

"""_______________________________________________________________________________________"""
import pandas as pn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
"""_______________________________________________________________________________________"""




class select_file(QMainWindow):
    df=None
    columns=None
    df_acp=None
    dic={}
    acp = PCA()
    nb_caracteristiques=0
    model=None
    def __init__(self):
        super().__init__()
        df=self.df
        loadUi("sel_file.ui", self)
        self.select_button.clicked.connect(self.select_file)
        self.ok_button.clicked.connect(self.selcted)
        self.setMaximumSize(515, 309)
        self.setMinimumSize(515, 309)
        self.widget_2.hide()
        self.widget_3.hide()
        self.widget_4.hide()
        self.widget_5.hide()
        self.vleur_manq_but.clicked.connect(self.handle_missing_values)
        self.codage_but.clicked.connect(self.handle_encoding)
        self.normalize_but.clicked.connect(self.handle_normalization)
        self.mat_corr.clicked.connect(self.handle_correlation_matrix)
        self.acp_but.clicked.connect(self.handle_principal_component_analysis)
        self.titre_acp.setText("Analyse en composantes principales")
        self.return1.clicked.connect(self.retour1)
        self.return2.clicked.connect(self.retour2)
        self.return3.clicked.connect(self.retour2)
        self.acp_interface_but.clicked.connect(self.handle_pca_interface)
        self.valeur_propre_but.clicked.connect(self.handle_eigenvalues_preservation)
        self.histograme_vp_but.clicked.connect(self.handle_eigenvalues_histogram)
        self.pourcentage_but.clicked.connect(self.handle_percentage_variance_preserved)
        self.saturation_but.clicked.connect(self.handle_eigenvalues_saturation)
        self.combo_box_acp_cp.currentIndexChanged.connect(self.handle_combobox_selection)
        self.cercledecorr_but.clicked.connect(self.selct_acp)
        self.data_but.clicked.connect(self.data_Mining_interface)
        self.k_means_but.clicked.connect(self.k_means)
        self.cah_but.clicked.connect(self.cah)
    def data_Mining_interface(self):
        self.selct_widg.hide()
        self.widget_2.hide()
        self.widget_4.hide()
        self.widget_5.show()
        self.cluster_results_table.setModel(self.model)


    def handle_pca_interface(self):
        self.widget_2.hide()
        self.selct_widg.hide()
        self.widget_4.show()
        model = QStandardItemModel()
        model.clear()
        self.tableView_acp.setModel(model)
        self.tableView_3.setModel(model)
        self.dic["acp"]=False
    def retour2(self):
        self.widget_2.show()
        self.widget_4.hide()
        self.widget_5.hide()
    def retour1(self):
        self.setMaximumSize(515, 309)
        self.setMinimumSize(515, 309)
        self.widget_2.hide()
        self.widget_4.hide()
        self.selct_widg.show()
    def select_file(self):
        file_path= QFileDialog.getOpenFileName(self, "Select File", "", "*.csv")
        if file_path:
            self.file_path_txt.setText(file_path[0])
            
    def selcted(self):
        ch=""
        self.df=self.file_path_txt.text()
        if self.df :
            self.setMaximumSize(911, 621)
            self.setMinimumSize(911, 621)
            self.selct_widg.hide()
            self.widget_2.show()
            self.widget.show()
            self.widget_3.hide()
            """____La phase de préparation de données_____"""
            # ---lire la base de données---
            self.df=pn.read_csv(self.df)
            self.columns=self.df.columns
            # print the donner dans tableau
            self.table = QStandardItemModel()
            self.populate_table(self.df)
            self.tableView.setModel(self.table)
            nb_observations = self.df.shape[0]
            self.nb_caracteristiques = self.df.shape[1]
            self.dic={"codage":False,"valeur_manquante":False,"acp":False}
            self.shape_txt.setText("Nombre d'observations : "+str(nb_observations)+"\nNombre de caractéristiques : "+str(self.nb_caracteristiques))
            if self.df.isnull().values.any():
                ch+="il y a des valeurs manquantes : "+str(self.df.isnull().sum().sum())+ "\n"
            no_numeric_columns = self.df.select_dtypes(exclude=['float64', 'int64']).columns
            if len(no_numeric_columns)>0:
                ch+="les colonnes non numériques sont : "+str(list(no_numeric_columns))+"\n"
            self.notification.setText(ch)
            
    def populate_table(self,df):
        self.table.setHorizontalHeaderLabels(self.columns)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                item = QStandardItem(str(df.iloc[i, j]))
                self.table.setItem(i, j, item)
    def populate_table_acp(self,df):
        self.table.setHorizontalHeaderLabels(self.columns)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                value = df.iloc[i, j]
                formatted_value = "{:.4f}".format(value)
                item = QStandardItem(formatted_value)
                self.table.setItem(i, j, item)
    
    def handle_missing_values(self):
        if self.dic["valeur_manquante"]==False:
            df=self.df
            ch=""
            numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                if df[col].isnull().any():
                    col_mean = df[col].mean()
                    df[col] = df[col].fillna(round(col_mean,4))
                    ch+="La colonne "+ str(col)+ " a été remplie avec la valeur moyenne: "+str(round(col_mean,4))+"\n"
            self.notification.setText(ch)
            self.populate_table(self.df)
            self.tableView.setModel(self.table)
            self.dic["valeur_manquante"]=True
    def handle_encoding(self):
        if self.dic["codage"]==False:
            df=self.df
            ch=""
            no_numeric_columns = df.select_dtypes(exclude=['float64', 'int64']).columns
            for column in no_numeric_columns:
                mapping_column = {columns: index for index, columns in enumerate(df[column].unique())}
                df[column] = df[column].replace(mapping_column)
                ch+="La colonne "+ str(column)+ f" a été transformer avec les valeurs : {mapping_column} \n"
            means = self.df.mean()
            stds = self.df.std()
            if means.all() != 0 or stds.all() != 1:
                ch+="La base de données n'est pas encore normalisée \n"
                self.dic["normalisation"]=False
            self.notification.setText(ch)
            self.populate_table(self.df)
            self.tableView.setModel(self.table)
            self.dic["codage"]=True
    def handle_normalization(self):
        if self.dic["codage"]==True and self.dic["valeur_manquante"]==True and self.dic["normalisation"]==False:
            df=self.df
            means=df.mean()
            stds=df.std()
            for col in df.columns:  
                df[col] = round((df[col] - means[col]) / stds[col],4)
            self.populate_table(self.df)
            self.tableView.setModel(self.table)
            self.dic["normalisation"]=True
            self.notification.setText(f"La base de données a été normalisée avec succès")
            self.widget_3.show()
    def handle_correlation_matrix(self):
        correlation_matrix = self.df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix,vmin=-1,vmax=1, annot=True, cmap='coolwarm', fmt="0.2f")
        plt.show()
        correlated_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.5: # SPESIFICATIO EN USAN LA MATRICE DE CORRELATION
                    correlated_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
        self.notification.setStyleSheet("color: white;cursor: pointer;display: inline-block;font-family: Inter,-apple-system,system-ui,\"Segoe UI\",Helvetica,Arial,sans-serif;font-size: 14px;font-weight: 600;letter-spacing: normal;line-height: 1.5")
        self.notification.setText(f"Les paires de variables corrélées sont: {correlated_pairs}")


    def handle_principal_component_analysis(self):
        acp = self.acp
        self.df_acp=acp.fit_transform(self.df)
        self.df_acp = pn.DataFrame(data=self.df_acp )
        self.populate_table_acp(self.df_acp)
        self.model=self.table
        self.tableView.setModel(self.table)
        self.tableView_acp.setModel(self.table)
        self.dic["acp"]=True
    def handle_eigenvalues_preservation(self):
        if self.dic["acp"]==True:   
            pca=self.acp
        # Créer un modèle de données
            model = QStandardItemModel()
            model.setHorizontalHeaderLabels(["Composante Principale", "Valeur Propre", "Pourcentage", "Pourcentage Cumulé"])
            
            # Remplir le modèle avec les résultats de l'ACP
            for i in range(len(pca.explained_variance_)):
                cp_item = QStandardItem(f"CP{i+1}")
                eigenvalue_item = QStandardItem(f"{pca.explained_variance_[i]:.4f}")
                percentage_item = QStandardItem(f"{pca.explained_variance_ratio_[i]*100:.2f}%")
                cumulative_percentage_item = QStandardItem(f"{pca.explained_variance_ratio_[:i+1].sum()*100:.2f}%")
                
                model.appendRow([cp_item, eigenvalue_item, percentage_item, cumulative_percentage_item])
            self.tableView_acp.setModel(model)
            self.handle_combobox_selection()
    
        # Ajuster la taille des colonnes pour qu'elles s'adaptent au contenu
            self.tableView_acp.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.titre_acp.setText("ACP - Valeurs propres")
            for i in range(len(self.columns)):
                self.cp1.addItem(f"{i + 1}")
                self.cp2.addItem(f"{i + 1}")
    def handle_eigenvalues_histogram(self):
        if self.dic["acp"]==True:
            valeurs_propres = self.acp.explained_variance_
            plt.title("Histogramme des valeurs propres")
            plt.plot(np.arange(1,self.nb_caracteristiques+1),valeurs_propres, color='red')# explained_variance_ pour obtenir les valeurs propres associées à chaque composante principale.
            plt.bar(np.arange(1,self.nb_caracteristiques+1),valeurs_propres,color='blue')
            plt.ylabel("valeur propre")
            plt.xlabel("ordre de valeur propre")
            plt.show()
    def handle_percentage_variance_preserved(self):
        if self.dic["acp"]==True:    
            explained_variance_ratio = self.acp.explained_variance_ratio_
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
    def handle_eigenvalues_saturation(self):
        if self.dic["acp"]==True:
            variable_saturation = self.acp.components_.T * np.sqrt(self.acp.explained_variance_)
            model = QStandardItemModel()

            # Ajouter des en-têtes de colonne
            model.setVerticalHeaderLabels(self.columns)
            model.setHorizontalHeaderLabels([f"CP{i + 1}" for i in range(len(self.acp.components_))])

            # Ajouter les données au modèle
            for i, row in enumerate(variable_saturation):
                for j, value in enumerate(row):
                    item = QStandardItem(f"{value:.4f}")  # Format avec 4 décimales
                    model.setItem(i, j, item)
            self.tableView_acp.setModel(model)
        
            # Ajuster la taille des colonnes pour qu'elles s'adaptent au contenu
            self.tableView_acp.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.titre_acp.setText(f"ACP-Saturations des variables sur les CP")
    def handle_combobox_selection(self):
        if self.dic["acp"]==True:
            selected_method = self.combo_box_acp_cp.currentText()
            model = QStandardItemModel()
            model.setHorizontalHeaderLabels(["Composante Principale", "Valeur Propre", "Pourcentage", "Pourcentage Cumulé"])
            pca = self.acp 
            if selected_method == "kaizer":
                for i in range(len(pca.explained_variance_)):
                    if pca.explained_variance_[i]>=1:
                        cp_item = QStandardItem(f"CP{i+1}")
                        eigenvalue_item = QStandardItem(f"{pca.explained_variance_[i]:.4f}")
                        percentage_item = QStandardItem(f"{pca.explained_variance_ratio_[i]*100:.2f}%")
                        cumulative_percentage_item = QStandardItem(f"{pca.explained_variance_ratio_[:i+1].sum()*100:.2f}%")
                    else:
                        break
                    model.appendRow([cp_item, eigenvalue_item, percentage_item, cumulative_percentage_item])
                self.tableView_3.setModel(model)
                pass
            elif selected_method == "Critère du pourcentage ( 70%)":
                for i in range(len(pca.explained_variance_)):
                    if pca.explained_variance_ratio_[:i+1].sum()*100<=70:
                        cp_item = QStandardItem(f"CP{i+1}")
                        eigenvalue_item = QStandardItem(f"{pca.explained_variance_[i]:.4f}")
                        percentage_item = QStandardItem(f"{pca.explained_variance_ratio_[i]*100:.2f}%")
                        cumulative_percentage_item = QStandardItem(f"{pca.explained_variance_ratio_[:i+1].sum()*100:.2f}%")
                    else:
                        break
                    model.appendRow([cp_item, eigenvalue_item, percentage_item, cumulative_percentage_item])
                self.tableView_3.setModel(model)
                pass
            elif selected_method == "Critère du pourcentage ( 80%)":
                for i in range(len(pca.explained_variance_)):
                    if pca.explained_variance_ratio_[:i+1].sum()*100<=80:
                        cp_item = QStandardItem(f"CP{i+1}")
                        eigenvalue_item = QStandardItem(f"{pca.explained_variance_[i]:.4f}")
                        percentage_item = QStandardItem(f"{pca.explained_variance_ratio_[i]*100:.2f}%")
                        cumulative_percentage_item = QStandardItem(f"{pca.explained_variance_ratio_[:i+1].sum()*100:.2f}%")
                    else:
                        break
                    model.appendRow([cp_item, eigenvalue_item, percentage_item, cumulative_percentage_item])
                self.tableView_3.setModel(model)
                pass
            elif selected_method == "Critère du pourcentage ( 90%)":
                for i in range(len(pca.explained_variance_)):
                    if pca.explained_variance_ratio_[:i+1].sum()*100<=90:
                        cp_item = QStandardItem(f"CP{i+1}")
                        eigenvalue_item = QStandardItem(f"{pca.explained_variance_[i]:.4f}")
                        percentage_item = QStandardItem(f"{pca.explained_variance_ratio_[i]*100:.2f}%")
                        cumulative_percentage_item = QStandardItem(f"{pca.explained_variance_ratio_[:i+1].sum()*100:.2f}%")
                    else:
                        break
                    model.appendRow([cp_item, eigenvalue_item, percentage_item, cumulative_percentage_item])
                self.tableView_3.setModel(model)
                pass
    def update_plot(self,pc1, pc2):
        pca=self.acp
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        plt.figure(figsize=(8, 8))
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.scatter(loadings[:, pc1 - 1], loadings[:, pc2 - 1], alpha=0.7)

        circle = plt.Circle((0, 0), 1, color='b', fill=False)
        plt.gca().add_artist(circle)

        # Annotate variable names
        for i, var in enumerate(self.columns):
            plt.arrow(0, 0, loadings[i, pc1 - 1], loadings[i, pc2 - 1])
            plt.annotate(var, (loadings[i, pc1 - 1], loadings[i, pc2 - 1]))

        plt.title("Correlation Circle")
        plt.xlabel(f"PC{pc1}")
        plt.ylabel(f"PC{pc2}")
        plt.grid(True)
        plt.show()
    def selct_acp(self):
        selected_pc1=int(self.cp1.currentText())
        selected_pc2=int(self.cp2.currentText())
        self.update_plot(selected_pc1, selected_pc2)
    def k_means(self):
        acp = PCA()
        df_acp=acp.fit_transform(self.df)
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
        self.k_means_txt.setText(f"K-means Inertie totale:\n{kmeans.inertia_}")
    def cah(self):
        acp = PCA()
        df_acp=acp.fit_transform(self.df)
        Z = linkage(df_acp, method='ward')
        # Afficher le dendrogramme pour CAH
        plt.figure(figsize=(10, 6))
        dendrogram(Z)
        plt.title('Dendrogramme pour CAH')
        plt.xlabel('Indice de l\'échantillon')
        plt.ylabel('Distance euclidienne')
        plt.show()
        c, coph_dists = cophenet(Z, pdist(self.df))
        # Calculate the total inertia
        total_inertia = np.sum(coph_dists)
        self.cah_txt.setText(f"CAH Inertie totale:\n{total_inertia}")
            
        
        


            
            




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = select_file()
    window.show()
    sys.exit(app.exec_())
