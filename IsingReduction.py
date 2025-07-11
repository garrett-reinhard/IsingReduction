import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
import optuna

def PCA_analysis(directory, scaler, n_components):

    full_paths = glob.glob(os.path.join(directory, 'spins_iter-*.csv'))

    flist = []
    for filename in full_paths:
        df = pd.read_csv(filename, header=None)
        flist.append(df)

    arraylist = []
    for file in flist:
        array = file.to_numpy()
        flattened_array = array.flatten()
        arraylist.append(flattened_array)

    df = pd.DataFrame(arraylist)

    scaled = scaler.fit_transform(df)

    pca = sklearn.decomposition.PCA(n_components)
    pca.fit(scaled)

    principal_components = pca.transform(scaled)
    return principal_components

def cluster_and_plot(pca, fig_name, model=sklearn.cluster.KMeans(n_clusters=2, random_state=42)):
    cluster_labels = model.fit_predict(principal_components)

    pdf = pd.DataFrame(data=principal_components)
    pdf['cluster'] = cluster_labels

    sns.pairplot(pdf, hue='cluster', diag_kind='kde')
    my_suptitle = plt.suptitle('Pairplot of Principal Components by Cluster', y=1.05)
    plt.savefig(f'{fig_name}', bbox_inches='tight',bbox_extra_artists=[my_suptitle])
    plt.show()

def clusters(pca, model=sklearn.cluster.KMeans(n_clusters=2, random_state=42)):
    cluster_labels = model.fit_predict(pca)

    return cluster_labels

## betaJ 0010

def objective(trial):

    directory_path = './Data/betaJ-0010_vf-050_nrows-100_ncols-100'

    n_components = trial.suggest_int("n_components", 2, 6)
    scaler_type = trial.suggest_categorical('scaler_type', ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler', 'PowerTransformer'])

    if scaler_type ==  'StandardScaler':
        scaler_type = sklearn.preprocessing.StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler_type = sklearn.preprocessing.MinMaxScaler()
    elif scaler_type == 'RobustScaler':
        scaler_type = sklearn.preprocessing.RobustScaler()
    elif scaler_type == 'MaxAbsScaler':
        scaler_type = sklearn.preprocessing.MaxAbsScaler()
    elif scaler_type == 'PowerTransformer':
        scaler_type = sklearn.preprocessing.PowerTransformer()

    pca = PCA_analysis(directory=directory_path, n_components=n_components, scaler=scaler_type)
    cluster_labels = clusters(pca=pca)

    score = sklearn.metrics.silhouette_score(pca, cluster_labels)

    return score

study = optuna.create_study(direction='maximize', study_name='pca_kmeans_optimization')

with open('betaJ-0010_vf-050.out', 'w') as f:
    print("Starting PCA and KMeans hyperparameter optimization")
    study.optimize(objective, n_trials=75)
    print("Optimization finished.")
    print("\nPCA and KMeans Optimization Results:", file=f)
    print(f"Number of finished trials: {len(study.trials)}", file=f)
    print(f"Best trial:", file=f)
    print(f"  Value (Silhouette Score): {study.best_value:.4f}", file=f)
    print(f"  Params: {study.best_params}", file=f)

best_n_components = study.best_params['n_components']
scaler_type = study.best_params['scaler_type']

if scaler_type ==  'StandardScaler':
    scaler_type = sklearn.preprocessing.StandardScaler()
elif scaler_type == 'MinMaxScaler':
    scaler_type = sklearn.preprocessing.MinMaxScaler()
elif scaler_type == 'RobustScaler':
    scaler_type = sklearn.preprocessing.RobustScaler()
elif scaler_type == 'MaxAbsScaler':
    scaler_type = sklearn.preprocessing.MaxAbsScaler()
elif scaler_type == 'PowerTransformer':
    scaler_type = sklearn.preprocessing.PowerTransformer()

directory_path = './Data/betaJ-0010_vf-050_nrows-100_ncols-100'

principal_components = PCA_analysis(directory=directory_path, n_components=best_n_components, scaler=scaler_type)
cluster_and_plot(pca=principal_components, fig_name='betaJ-0010_vf-050.png')

## betaJ 0100

def objective(trial):

    directory_path = './Data/betaJ-0100_vf-050_nrows-100_ncols-100'

    n_components = trial.suggest_int("n_components", 2, 6)
    scaler_type = trial.suggest_categorical('scaler_type', ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler', 'PowerTransformer'])

    if scaler_type ==  'StandardScaler':
        scaler_type = sklearn.preprocessing.StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler_type = sklearn.preprocessing.MinMaxScaler()
    elif scaler_type == 'RobustScaler':
        scaler_type = sklearn.preprocessing.RobustScaler()
    elif scaler_type == 'MaxAbsScaler':
        scaler_type = sklearn.preprocessing.MaxAbsScaler()
    elif scaler_type == 'PowerTransformer':
        scaler_type = sklearn.preprocessing.PowerTransformer()

    pca = PCA_analysis(directory=directory_path, n_components=n_components, scaler=scaler_type)
    cluster_labels = clusters(pca=pca)

    score = sklearn.metrics.silhouette_score(pca, cluster_labels)

    return score

study = optuna.create_study(direction='maximize', study_name='pca_kmeans_optimization')

with open('betaJ-0100_vf-050.out', 'w') as f:
    print("Starting PCA and KMeans hyperparameter optimization")
    study.optimize(objective, n_trials=75)
    print("Optimization finished.")
    print("\nPCA and KMeans Optimization Results:", file=f)
    print(f"Number of finished trials: {len(study.trials)}", file=f)
    print(f"Best trial:", file=f)
    print(f"  Value (Silhouette Score): {study.best_value:.4f}", file=f)
    print(f"  Params: {study.best_params}", file=f)

best_n_components = study.best_params['n_components']
scaler_type = study.best_params['scaler_type']

if scaler_type ==  'StandardScaler':
    scaler_type = sklearn.preprocessing.StandardScaler()
elif scaler_type == 'MinMaxScaler':
    scaler_type = sklearn.preprocessing.MinMaxScaler()
elif scaler_type == 'RobustScaler':
    scaler_type = sklearn.preprocessing.RobustScaler()
elif scaler_type == 'MaxAbsScaler':
    scaler_type = sklearn.preprocessing.MaxAbsScaler()
elif scaler_type == 'PowerTransformer':
    scaler_type = sklearn.preprocessing.PowerTransformer()

directory_path = './Data/betaJ-0100_vf-050_nrows-100_ncols-100'

principal_components = PCA_analysis(directory=directory_path, n_components=best_n_components, scaler=scaler_type)
cluster_and_plot(pca=principal_components, fig_name='betaJ-0100_vf-050.png')

## betaJ 1000

def objective(trial):

    directory_path = './Data/betaJ-1000_vf-050_nrows-100_ncols-100'

    n_components = trial.suggest_int("n_components", 2, 6)
    scaler_type = trial.suggest_categorical('scaler_type', ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler', 'PowerTransformer'])

    if scaler_type ==  'StandardScaler':
        scaler_type = sklearn.preprocessing.StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler_type = sklearn.preprocessing.MinMaxScaler()
    elif scaler_type == 'RobustScaler':
        scaler_type = sklearn.preprocessing.RobustScaler()
    elif scaler_type == 'MaxAbsScaler':
        scaler_type = sklearn.preprocessing.MaxAbsScaler()
    elif scaler_type == 'PowerTransformer':
        scaler_type = sklearn.preprocessing.PowerTransformer()

    pca = PCA_analysis(directory=directory_path, n_components=n_components, scaler=scaler_type)
    cluster_labels = clusters(pca=pca)

    score = sklearn.metrics.silhouette_score(pca, cluster_labels)

    return score

study = optuna.create_study(direction='maximize', study_name='pca_kmeans_optimization')

with open('betaJ-1000_vf-050.out', 'w') as f:
    print("Starting PCA and KMeans hyperparameter optimization")
    study.optimize(objective, n_trials=75)
    print("Optimization finished.")
    print("\nPCA and KMeans Optimization Results:", file=f)
    print(f"Number of finished trials: {len(study.trials)}", file=f)
    print(f"Best trial:", file=f)
    print(f"  Value (Silhouette Score): {study.best_value:.4f}", file=f)
    print(f"  Params: {study.best_params}", file=f)

best_n_components = study.best_params['n_components']
scaler_type = study.best_params['scaler_type']

if scaler_type ==  'StandardScaler':
    scaler_type = sklearn.preprocessing.StandardScaler()
elif scaler_type == 'MinMaxScaler':
    scaler_type = sklearn.preprocessing.MinMaxScaler()
elif scaler_type == 'RobustScaler':
    scaler_type = sklearn.preprocessing.RobustScaler()
elif scaler_type == 'MaxAbsScaler':
    scaler_type = sklearn.preprocessing.MaxAbsScaler()
elif scaler_type == 'PowerTransformer':
    scaler_type = sklearn.preprocessing.PowerTransformer()

directory_path = './Data/betaJ-1000_vf-050_nrows-100_ncols-100'

principal_components = PCA_analysis(directory=directory_path, n_components=best_n_components, scaler=scaler_type)
cluster_and_plot(pca=principal_components, fig_name='betaJ-1000_vf-050.png')

## betaJ 0200 w 025 vf

def objective(trial):

    directory_path = './Data/betaJ-0200_vf-025_nrows-100_ncols-100'

    n_components = trial.suggest_int("n_components", 2, 6)
    scaler_type = trial.suggest_categorical('scaler_type', ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler', 'PowerTransformer'])

    if scaler_type ==  'StandardScaler':
        scaler_type = sklearn.preprocessing.StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler_type = sklearn.preprocessing.MinMaxScaler()
    elif scaler_type == 'RobustScaler':
        scaler_type = sklearn.preprocessing.RobustScaler()
    elif scaler_type == 'MaxAbsScaler':
        scaler_type = sklearn.preprocessing.MaxAbsScaler()
    elif scaler_type == 'PowerTransformer':
        scaler_type = sklearn.preprocessing.PowerTransformer()

    pca = PCA_analysis(directory=directory_path, n_components=n_components, scaler=scaler_type)
    cluster_labels = clusters(pca=pca)

    score = sklearn.metrics.silhouette_score(pca, cluster_labels)

    return score

study = optuna.create_study(direction='maximize', study_name='pca_kmeans_optimization')

with open('betaJ-0200_vf-025.out', 'w') as f:
    print("Starting PCA and KMeans hyperparameter optimization")
    study.optimize(objective, n_trials=75)
    print("Optimization finished.")
    print("\nPCA and KMeans Optimization Results:", file=f)
    print(f"Number of finished trials: {len(study.trials)}", file=f)
    print(f"Best trial:", file=f)
    print(f"  Value (Silhouette Score): {study.best_value:.4f}", file=f)
    print(f"  Params: {study.best_params}", file=f)

best_n_components = study.best_params['n_components']
scaler_type = study.best_params['scaler_type']

if scaler_type ==  'StandardScaler':
    scaler_type = sklearn.preprocessing.StandardScaler()
elif scaler_type == 'MinMaxScaler':
    scaler_type = sklearn.preprocessing.MinMaxScaler()
elif scaler_type == 'RobustScaler':
    scaler_type = sklearn.preprocessing.RobustScaler()
elif scaler_type == 'MaxAbsScaler':
    scaler_type = sklearn.preprocessing.MaxAbsScaler()
elif scaler_type == 'PowerTransformer':
    scaler_type = sklearn.preprocessing.PowerTransformer()

directory_path = './Data/betaJ-0200_vf-025_nrows-100_ncols-100'

principal_components = PCA_analysis(directory=directory_path, n_components=best_n_components, scaler=scaler_type)
cluster_and_plot(pca=principal_components, fig_name='betaJ-0200_vf-025.png')

## betaJ 0200 vf 050

def objective(trial):

    directory_path = './Data/betaJ-0200_vf-050_nrows-100_ncols-100'

    n_components = trial.suggest_int("n_components", 2, 6)
    scaler_type = trial.suggest_categorical('scaler_type', ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler', 'PowerTransformer'])

    if scaler_type ==  'StandardScaler':
        scaler_type = sklearn.preprocessing.StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler_type = sklearn.preprocessing.MinMaxScaler()
    elif scaler_type == 'RobustScaler':
        scaler_type = sklearn.preprocessing.RobustScaler()
    elif scaler_type == 'MaxAbsScaler':
        scaler_type = sklearn.preprocessing.MaxAbsScaler()
    elif scaler_type == 'PowerTransformer':
        scaler_type = sklearn.preprocessing.PowerTransformer()

    pca = PCA_analysis(directory=directory_path, n_components=n_components, scaler=scaler_type)
    cluster_labels = clusters(pca=pca)

    score = sklearn.metrics.silhouette_score(pca, cluster_labels)

    return score

study = optuna.create_study(direction='maximize', study_name='pca_kmeans_optimization')

with open('betaJ-0200_vf-050.out', 'w') as f:
    print("Starting PCA and KMeans hyperparameter optimization")
    study.optimize(objective, n_trials=75)
    print("Optimization finished.")
    print("\nPCA and KMeans Optimization Results:", file=f)
    print(f"Number of finished trials: {len(study.trials)}", file=f)
    print(f"Best trial:", file=f)
    print(f"  Value (Silhouette Score): {study.best_value:.4f}", file=f)
    print(f"  Params: {study.best_params}", file=f)

best_n_components = study.best_params['n_components']
scaler_type = study.best_params['scaler_type']

if scaler_type ==  'StandardScaler':
    scaler_type = sklearn.preprocessing.StandardScaler()
elif scaler_type == 'MinMaxScaler':
    scaler_type = sklearn.preprocessing.MinMaxScaler()
elif scaler_type == 'RobustScaler':
    scaler_type = sklearn.preprocessing.RobustScaler()
elif scaler_type == 'MaxAbsScaler':
    scaler_type = sklearn.preprocessing.MaxAbsScaler()
elif scaler_type == 'PowerTransformer':
    scaler_type = sklearn.preprocessing.PowerTransformer()

directory_path = './Data/betaJ-0200_vf-050_nrows-100_ncols-100'

principal_components = PCA_analysis(directory=directory_path, n_components=best_n_components, scaler=scaler_type)
cluster_and_plot(pca=principal_components, fig_name='betaJ-0200_vf-050.png')

## betaJ 0200 vf 080

def objective(trial):

    directory_path = './Data/betaJ-0200_vf-080_nrows-100_ncols-100'

    n_components = trial.suggest_int("n_components", 2, 6)
    scaler_type = trial.suggest_categorical('scaler_type', ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler', 'PowerTransformer'])

    if scaler_type ==  'StandardScaler':
        scaler_type = sklearn.preprocessing.StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler_type = sklearn.preprocessing.MinMaxScaler()
    elif scaler_type == 'RobustScaler':
        scaler_type = sklearn.preprocessing.RobustScaler()
    elif scaler_type == 'MaxAbsScaler':
        scaler_type = sklearn.preprocessing.MaxAbsScaler()
    elif scaler_type == 'PowerTransformer':
        scaler_type = sklearn.preprocessing.PowerTransformer()

    pca = PCA_analysis(directory=directory_path, n_components=n_components, scaler=scaler_type)
    cluster_labels = clusters(pca=pca)

    score = sklearn.metrics.silhouette_score(pca, cluster_labels)

    return score

study = optuna.create_study(direction='maximize', study_name='pca_kmeans_optimization')

with open('betaJ-0200_vf-080.out', 'w') as f:
    print("Starting PCA and KMeans hyperparameter optimization")
    study.optimize(objective, n_trials=75)
    print("Optimization finished.")
    print("\nPCA and KMeans Optimization Results:", file=f)
    print(f"Number of finished trials: {len(study.trials)}", file=f)
    print(f"Best trial:", file=f)
    print(f"  Value (Silhouette Score): {study.best_value:.4f}", file=f)
    print(f"  Params: {study.best_params}", file=f)

best_n_components = study.best_params['n_components']
scaler_type = study.best_params['scaler_type']

if scaler_type ==  'StandardScaler':
    scaler_type = sklearn.preprocessing.StandardScaler()
elif scaler_type == 'MinMaxScaler':
    scaler_type = sklearn.preprocessing.MinMaxScaler()
elif scaler_type == 'RobustScaler':
    scaler_type = sklearn.preprocessing.RobustScaler()
elif scaler_type == 'MaxAbsScaler':
    scaler_type = sklearn.preprocessing.MaxAbsScaler()
elif scaler_type == 'PowerTransformer':
    scaler_type = sklearn.preprocessing.PowerTransformer()

directory_path = './Data/betaJ-0200_vf-050_nrows-100_ncols-100'

principal_components = PCA_analysis(directory=directory_path, n_components=best_n_components, scaler=scaler_type)
cluster_and_plot(pca=principal_components, fig_name='betaJ-0200_vf-080.png')
