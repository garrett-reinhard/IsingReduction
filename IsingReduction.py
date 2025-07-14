import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
import optuna
import umap


def PCA_analysis_all_data(directories, data_set_label, scaler, n_components):
    
    df_list = []
    all_labels = []

    for i, directory in enumerate(directories):

        label = data_set_label[i]
            
        full_paths = glob.glob(os.path.join(directory, 'spins_iter-*.csv'))

        flist = []
        for filename in full_paths:
            df = pd.read_csv(filename, header=None) 
            flist.append(df)
            all_labels.append(label)

        arraylist = []
        for file in flist:
            array = file.to_numpy()
            flattened_array = array.flatten()
            arraylist.append(flattened_array)
        
        df = pd.DataFrame(arraylist)
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True) 
    labels = pd.Series(all_labels, name='dataset')
 
    scaled = scaler.fit_transform(merged_df)

    pca = sklearn.decomposition.PCA(n_components)
    pca.fit(scaled)

    principal_components = pca.transform(scaled)

    return principal_components, labels

def clusters(pca, model=sklearn.cluster.KMeans(n_clusters=2, random_state=42)):
    
    cluster_labels = model.fit_predict(pca)

    return cluster_labels

def PCA_plots(principal_components, n_components, fig_name, labels):

    pc_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
    pc_df['dataset'] = labels
    sns.pairplot(pc_df, hue='dataset')
    plt.savefig(fig_name)
    plt.show()
    
    return

def tSNE_analysis_all_data(directories, data_set_label, scaler, n_components, perplexity=30, n_iter=250):
    
    df_list = []
    all_labels = []

    for i, directory in enumerate(directories):

        label = data_set_label[i]
            
        full_paths = glob.glob(os.path.join(directory, 'spins_iter-*.csv'))

        flist = []
        for filename in full_paths:
            df = pd.read_csv(filename, header=None) 
            flist.append(df)
            all_labels.append(label)

        arraylist = []
        for file in flist:
            array = file.to_numpy()
            flattened_array = array.flatten()
            arraylist.append(flattened_array)
        
        df = pd.DataFrame(arraylist)
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True) 
    labels = pd.Series(all_labels, name='dataset')
 
    scaled = scaler.fit_transform(merged_df)

    tsne = sklearn.manifold.TSNE(n_components, random_state=42, perplexity=perplexity, n_iter=n_iter)
    tsne_x = tsne.fit_transform(scaled)

    return tsne_x, labels

def tsne_plots(tsne, n_components, fig_name, labels):

    pc_df = pd.DataFrame(data=tsne, columns=[f'tSNE{i+1}' for i in range(n_components)])
    pc_df['dataset'] = labels
    sns.pairplot(pc_df, hue='dataset')
    plt.savefig(fig_name)
    plt.show()
    
    return

def umap_analysis_all_data(directories, data_set_label, scaler, n_neighbors, min_dist, n_components, metric):
    
    df_list = []
    all_labels = []

    for i, directory in enumerate(directories):

        label = data_set_label[i]
            
        full_paths = glob.glob(os.path.join(directory, 'spins_iter-*.csv'))

        flist = []
        for filename in full_paths:
            df = pd.read_csv(filename, header=None) 
            flist.append(df)
            all_labels.append(label)

        arraylist = []
        for file in flist:
            array = file.to_numpy()
            flattened_array = array.flatten()
            arraylist.append(flattened_array)
        
        df = pd.DataFrame(arraylist)
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True) 
    labels = pd.Series(all_labels, name='dataset')
 
    scaled = scaler.fit_transform(merged_df)

    umap_reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    umap_embedding = umap_reducer.fit_transform(scaled)

    return umap_embedding, labels

def umap_plots(umap, n_components, fig_name, labels):

    pc_df = pd.DataFrame(data=umap, columns=[f'UMAP{i+1}' for i in range(n_components)])
    pc_df['dataset'] = labels
    sns.pairplot(pc_df, hue='dataset')
    plt.savefig(fig_name)
    plt.show()
    
    return


n_trials = 1

 
### Changing betaJ

 
### PCA


directory_list = ['./Data/betaJ-0010_vf-050_nrows-100_ncols-100', './Data/betaJ-0100_vf-050_nrows-100_ncols-100', './Data/betaJ-1000_vf-050_nrows-100_ncols-100']

dataset_label = ['0010vf-050', '0100vf-050', '1000vf-050']

def objective(trial):

    n_components = trial.suggest_int("n_components", 2, 4)
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

    pca, x = PCA_analysis_all_data(directories=directory_list, data_set_label=dataset_label, n_components=n_components, scaler=scaler_type)
    cluster_labels = clusters(pca=pca)

    score = sklearn.metrics.silhouette_score(pca, cluster_labels)

    return score

study = optuna.create_study(direction='maximize', study_name='pca_optimization')
    
with open('betaJ-0010_vf-050.out', 'w') as f:
    print("Starting PCA and KMeans hyperparameter optimization")
    study.optimize(objective, n_trials=n_trials)
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

principal_components, labels = PCA_analysis_all_data(directories=directory_list, data_set_label=dataset_label, n_components=best_n_components, scaler=scaler_type)
PCA_plots(principal_components=principal_components, n_components=best_n_components, fig_name='betaJ_PCA.png', labels=labels)

 
### TSNE


directory_list = ['./Data/betaJ-0010_vf-050_nrows-100_ncols-100', './Data/betaJ-0100_vf-050_nrows-100_ncols-100', './Data/betaJ-1000_vf-050_nrows-100_ncols-100']

dataset_label = ['0010vf-050', '0100vf-050', '1000vf-050']

def objective(trial):

    n_components = trial.suggest_int('n_components', 2, 4)
    scaler_type = trial.suggest_categorical('scaler_type', ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler', 'PowerTransformer'])
    perplexity = trial.suggest_float('perplexity', 5, 50, step=0.5)
    n_iter = trial.suggest_int('n_iter', 250, 2000, step=50)

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

    tsne, x = tSNE_analysis_all_data(directories=directory_list, data_set_label=dataset_label, n_components=n_components, scaler=scaler_type, perplexity=perplexity, n_iter=n_iter)
    cluster_labels = clusters(pca=tsne)

    score = sklearn.metrics.silhouette_score(tsne, cluster_labels)

    return score

study = optuna.create_study(direction='maximize', study_name='tsne_optimization')
    
with open('betaJ-0010_vf-050.out', 'w') as f:
    print("Starting TSNE hyperparameter optimization")
    study.optimize(objective, n_trials=n_trials)
    print("Optimization finished.")
    print("\nTSNE Optimization Results:", file=f)
    print(f"Number of finished trials: {len(study.trials)}", file=f)
    print(f"Best trial:", file=f)
    print(f"  Value (Silhouette Score): {study.best_value:.4f}", file=f)
    print(f"  Params: {study.best_params}", file=f)

best_n_components = study.best_params['n_components']
scaler_type = study.best_params['scaler_type']
best_perplexity = study.best_params['perplexity']
best_iter = study.best_params['n_iters']

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

tsne, labels = tSNE_analysis_all_data(directories=directory_list, data_set_label=dataset_label, n_components=best_n_components, scaler=scaler_type, perplexity=best_perplexity, n_iters=best_iter)
tsne_plots(tsne=tsne, n_components=best_n_components, fig_name='betaJ_tsne.png', labels=labels)


directory_list = ['./Data/betaJ-0010_vf-050_nrows-100_ncols-100', './Data/betaJ-0100_vf-050_nrows-100_ncols-100', './Data/betaJ-1000_vf-050_nrows-100_ncols-100']

dataset_label = ['0010vf-050', '0100vf-050', '1000vf-050']

def objective(trial):

    n_components = trial.suggest_int("n_components", 2, 4)
    scaler_type = trial.suggest_categorical('scaler_type', ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler', 'PowerTransformer'])
    min_dist = trial.suggest_float('min_dist', 0.0, 1.0, step=0.5)
    n_neighbors = trial.suggest_int('n_neighbors', 10, 250, step=10)
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'])

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

    umap, labels = umap_analysis_all_data(directories=directory_list, data_set_label=dataset_label, scaler=scaler_type, n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    cluster_labels = clusters(pca=umap)

    score = sklearn.metrics.silhouette_score(umap, cluster_labels)

    return score

study = optuna.create_study(direction='maximize', study_name='umap_optimization')
    
with open('betaJ-0010_vf-050.out', 'w') as f:
    print("Starting UMAP hyperparameter optimization")
    study.optimize(objective, n_trials=n_trials)
    print("Optimization finished.")
    print("\nUMAP Optimization Results:", file=f)
    print(f"Number of finished trials: {len(study.trials)}", file=f)
    print(f"Best trial:", file=f)
    print(f"  Value (Silhouette Score): {study.best_value:.4f}", file=f)
    print(f"  Params: {study.best_params}", file=f)

best_n_components = study.best_params['n_components']
scaler_type = study.best_params['scaler_type']
best_min_dist = study.best_params['min_dist']
best_n_neighbors = study.best_params['n_neighbors']
best_metric = study.best_params['metric']

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

umap, labels = umap_analysis_all_data(directories=directory_list, data_set_label=dataset_label, scaler=scaler_type, n_neighbors=best_n_neighbors, min_dist=best_min_dist, n_components=best_n_components, metric=best_metric)
umap_plots(umap=umap, n_components=best_n_components, fig_name='umap_betaJ.png', labels=labels)

 
### Changing Volume Fraction

 
### PCA


directory_list = ['./Data/betaJ-0200_vf-025_nrows-100_ncols-100', './Data/betaJ-0200_vf-050_nrows-100_ncols-100', './Data/betaJ-0200_vf-080_nrows-100_ncols-100']

dataset_label = ['0200vf-025', '0200vf-050', '0200vf-080']

def objective(trial):

    n_components = trial.suggest_int("n_components", 2, 4)
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

    pca, x = PCA_analysis_all_data(directories=directory_list, data_set_label=dataset_label, n_components=n_components, scaler=scaler_type)
    cluster_labels = clusters(pca=pca)

    score = sklearn.metrics.silhouette_score(pca, cluster_labels)

    return score

study = optuna.create_study(direction='maximize', study_name='pca_optimization')
    
with open('betaJ-0010_vf-050.out', 'w') as f:
    print("Starting PCA and KMeans hyperparameter optimization")
    study.optimize(objective, n_trials=n_trials)
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

principal_components, labels = PCA_analysis_all_data(directories=directory_list, data_set_label=dataset_label, n_components=best_n_components, scaler=scaler_type)
PCA_plots(principal_components=principal_components, n_components=best_n_components, fig_name='volume_PCA.png', labels=labels)

 
### TSNE


directory_list = ['./Data/betaJ-0200_vf-025_nrows-100_ncols-100', './Data/betaJ-0200_vf-050_nrows-100_ncols-100', './Data/betaJ-0200_vf-080_nrows-100_ncols-100']

dataset_label = ['0200vf-025', '0200vf-050', '0200vf-080']

def objective(trial):

    n_components = trial.suggest_int('n_components', 2, 4)
    scaler_type = trial.suggest_categorical('scaler_type', ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler', 'PowerTransformer'])
    perplexity = trial.suggest_float('perplexity', 5, 50, step=0.5)
    n_iter = trial.suggest_int('n_iter', 250, 2000, step=50)

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

    tsne, x = tSNE_analysis_all_data(directories=directory_list, data_set_label=dataset_label, n_components=n_components, scaler=scaler_type, perplexity=perplexity, n_iter=n_iter)
    cluster_labels = clusters(pca=tsne)

    score = sklearn.metrics.silhouette_score(tsne, cluster_labels)

    return score

study = optuna.create_study(direction='maximize', study_name='tsne_optimization')
    
with open('betaJ-0010_vf-050.out', 'w') as f:
    print("Starting TSNE hyperparameter optimization")
    study.optimize(objective, n_trials=n_trials)
    print("Optimization finished.")
    print("\nTSNE Optimization Results:", file=f)
    print(f"Number of finished trials: {len(study.trials)}", file=f)
    print(f"Best trial:", file=f)
    print(f"  Value (Silhouette Score): {study.best_value:.4f}", file=f)
    print(f"  Params: {study.best_params}", file=f)

best_n_components = study.best_params['n_components']
scaler_type = study.best_params['scaler_type']
best_perplexity = study.best_params['perplexity']
best_iter = study.best_params['n_iters']

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

tsne, labels = tSNE_analysis_all_data(directories=directory_list, data_set_label=dataset_label, n_components=best_n_components, scaler=scaler_type, perplexity=best_perplexity, n_iters=best_iter)
tsne_plots(tsne=tsne, n_components=best_n_components, fig_name='volume_tsne.png', labels=labels)

 
### UMAP


directory_list = ['./Data/betaJ-0200_vf-025_nrows-100_ncols-100', './Data/betaJ-0200_vf-050_nrows-100_ncols-100', './Data/betaJ-0200_vf-080_nrows-100_ncols-100']

dataset_label = ['0200vf-025', '0200vf-050', '0200vf-080']

def objective(trial):

    n_components = trial.suggest_int("n_components", 2, 4)
    scaler_type = trial.suggest_categorical('scaler_type', ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler', 'PowerTransformer'])
    min_dist = trial.suggest_float('min_dist', 0.0, 1.0, step=0.5)
    n_neighbors = trial.suggest_int('n_neighbors', 10, 250, step=10)
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'])

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

    umap, labels = umap_analysis_all_data(directories=directory_list, data_set_label=dataset_label, scaler=scaler_type, n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    cluster_labels = clusters(pca=umap)

    score = sklearn.metrics.silhouette_score(umap, cluster_labels)

    return score

study = optuna.create_study(direction='maximize', study_name='umap_optimization')
    
with open('betaJ-0010_vf-050.out', 'w') as f:
    print("Starting UMAP hyperparameter optimization")
    study.optimize(objective, n_trials=n_trials)
    print("Optimization finished.")
    print("\nUMAP Optimization Results:", file=f)
    print(f"Number of finished trials: {len(study.trials)}", file=f)
    print(f"Best trial:", file=f)
    print(f"  Value (Silhouette Score): {study.best_value:.4f}", file=f)
    print(f"  Params: {study.best_params}", file=f)

best_n_components = study.best_params['n_components']
scaler_type = study.best_params['scaler_type']
best_min_dist = study.best_params['min_dist']
best_n_neighbors = study.best_params['n_neighbors']
best_metric = study.best_params['metric']

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

umap, labels = umap_analysis_all_data(directories=directory_list, data_set_label=dataset_label, scaler=scaler_type, n_neighbors=best_n_neighbors, min_dist=best_min_dist, n_components=best_n_components, metric=best_metric)
umap_plots(umap=umap, n_components=best_n_components, fig_name='umap_volume.png', labels=labels)
