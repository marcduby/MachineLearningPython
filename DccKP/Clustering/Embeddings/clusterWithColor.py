

def create_umap_comparison_file_for_node_type(csv_file, node_type, color_strategy='cluster'):
    '''
    Creates a UMAP graphic file for the dataframe provided with color-coded information
    
    Parameters:
    - csv_file: path to CSV file
    - node_type: type of node for analysis
    - color_strategy: 'cluster', 'feature', 'density', or 'column'
        - 'cluster': Color by KMeans clusters
        - 'feature': Color by first principal component
        - 'density': Color by local density
        - 'column': Color by a specific column (requires column_name parameter)
    '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import umap
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    
    # Read data
    file_out = "umap_parameter_comparison_{}.png".format(node_type)
    df = pd.read_csv(csv_file)

    # log
    print("\nExperimenting with different UMAP parameters for node type: {}".format(node_type))
    print("Using color strategy: {}".format(color_strategy))

    # get the features
    feature_columns = [col for col in df.columns if col.startswith('val_')]
    features = df[feature_columns].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Prepare color information based on strategy
    if color_strategy == 'cluster':
        # Color by KMeans clusters
        kmeans = KMeans(n_clusters=5, random_state=42)
        color_data = kmeans.fit_predict(features_scaled)
        color_label = 'K-Means Clusters (k=5)'
        cmap = 'tab10'
        
    elif color_strategy == 'feature':
        # Color by first principal component
        pca = PCA(n_components=1)
        color_data = pca.fit_transform(features_scaled).flatten()
        color_label = 'First Principal Component'
        cmap = 'viridis'
        
    elif color_strategy == 'density':
        # Color by local density (distance to 5th nearest neighbor)
        nbrs = NearestNeighbors(n_neighbors=6).fit(features_scaled)  # 6 because it includes the point itself
        distances, indices = nbrs.kneighbors(features_scaled)
        color_data = 1.0 / (distances[:, 5] + 1e-10)  # Inverse distance to 5th neighbor
        color_label = 'Local Density'
        cmap = 'plasma'
        
    elif color_strategy == 'magnitude':
        # Color by feature magnitude (L2 norm)
        color_data = np.linalg.norm(features_scaled, axis=1)
        color_label = 'Feature Magnitude'
        cmap = 'coolwarm'
        
    else:  # Default to cluster
        kmeans = KMeans(n_clusters=5, random_state=42)
        color_data = kmeans.fit_predict(features_scaled)
        color_label = 'K-Means Clusters (k=5)'
        cmap = 'tab10'
    
    # Parameter combinations
    param_combinations = [
        {'n_neighbors': 3,  'min_dist': 0.001,  'title': 'n_neighbors=3,  min_dist=0.001'},
        {'n_neighbors': 5,  'min_dist': 0.005,  'title': 'n_neighbors=5,  min_dist=0.005'},
        {'n_neighbors': 5, 'min_dist': 0.01, 'title': 'n_neighbors=5, min_dist=0.01'},
        {'n_neighbors': 10, 'min_dist': 0.002,  'title': 'n_neighbors=10, min_dist=0.002'},
        {'n_neighbors': 15, 'min_dist': 0.001,  'title': 'n_neighbors=15, min_dist=0.001'},
        {'n_neighbors': 10, 'min_dist': 0.01,  'spread': 0.3,
         'title': 'n_neighbors=10, min_dist=0.01, spread=0.3'},
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Add a colorbar to the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]

    for i, params in enumerate(param_combinations):
        # Calculate row and column indices
        row = i // 3
        col = i % 3
        
        # Create UMAP reducer with parameters
        umap_params = {
            'n_neighbors': params['n_neighbors'],
            'min_dist': params['min_dist'],
            'n_components': 2,
            'random_state': 42
        }
        if 'spread' in params:
            umap_params['spread'] = params['spread']
            
        umap_reducer = umap.UMAP(**umap_params)
        embedding = umap_reducer.fit_transform(features_scaled)
        
        # Create scatter plot with colors
        scatter = axes[row, col].scatter(
            embedding[:, 0], embedding[:, 1], 
            c=color_data, 
            cmap=cmap,
            alpha=0.7, 
            s=50,
            edgecolors='black',
            linewidth=0.1
        )
        
        axes[row, col].set_title(params['title'], fontsize=10)
        axes[row, col].set_xlabel('UMAP Dimension 1')
        axes[row, col].set_ylabel('UMAP Dimension 2')
        axes[row, col].grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label(color_label, rotation=270, labelpad=20)

    plt.tight_layout()
    plt.subplots_adjust(right=0.9)  # Make room for colorbar
    plt.savefig(file_out, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Parameter comparison plot saved as '{}'".format(file_out))


def create_umap_comparison_with_specific_column(csv_file, node_type, column_name):
    '''
    Creates UMAP comparison colored by a specific column in the dataframe
    '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import umap
    from sklearn.preprocessing import StandardScaler
    
    # Read data
    file_out = "umap_parameter_comparison_{}_{}.png".format(node_type, column_name)
    df = pd.read_csv(csv_file)

    print("\nExperimenting with different UMAP parameters for node type: {}".format(node_type))
    print("Coloring by column: {}".format(column_name))
    
    if column_name not in df.columns:
        print("Warning: Column '{}' not found in dataframe. Available columns: {}".format(
            column_name, list(df.columns)))
        return

    # get the features
    feature_columns = [col for col in df.columns if col.startswith('val_')]
    features = df[feature_columns].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Get color data from specified column
    color_data = df[column_name].values
    
    # Choose colormap based on data type
    if df[column_name].dtype in ['object', 'category']:
        # Categorical data
        unique_vals = df[column_name].unique()
        color_map = {val: i for i, val in enumerate(unique_vals)}
        color_data = [color_map[val] for val in color_data]
        cmap = 'tab10'
    else:
        # Numerical data
        cmap = 'viridis'
    
    # Parameter combinations
    param_combinations = [
        {'n_neighbors': 3,  'min_dist': 0.001,  'title': 'n_neighbors=3,  min_dist=0.001'},
        {'n_neighbors': 5,  'min_dist': 0.005,  'title': 'n_neighbors=5,  min_dist=0.005'},
        {'n_neighbors': 5, 'min_dist': 0.01, 'title': 'n_neighbors=5, min_dist=0.01'},
        {'n_neighbors': 10, 'min_dist': 0.002,  'title': 'n_neighbors=10, min_dist=0.002'},
        {'n_neighbors': 15, 'min_dist': 0.001,  'title': 'n_neighbors=15, min_dist=0.001'},
        {'n_neighbors': 10, 'min_dist': 0.01,  'spread': 0.3,
         'title': 'n_neighbors=10, min_dist=0.01, spread=0.3'},
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    for i, params in enumerate(param_combinations):
        row = i // 3
        col = i % 3
        
        umap_params = {
            'n_neighbors': params['n_neighbors'],
            'min_dist': params['min_dist'],
            'n_components': 2,
            'random_state': 42
        }
        if 'spread' in params:
            umap_params['spread'] = params['spread']
            
        umap_reducer = umap.UMAP(**umap_params)
        embedding = umap_reducer.fit_transform(features_scaled)
        
        scatter = axes[row, col].scatter(
            embedding[:, 0], embedding[:, 1], 
            c=color_data, 
            cmap=cmap,
            alpha=0.7, 
            s=50,
            edgecolors='black',
            linewidth=0.1
        )
        
        axes[row, col].set_title(params['title'], fontsize=10)
        axes[row, col].set_xlabel('UMAP Dimension 1')
        axes[row, col].set_ylabel('UMAP Dimension 2')
        axes[row, col].grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label(column_name, rotation=270, labelpad=20)

    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.savefig(file_out, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Parameter comparison plot saved as '{}'".format(file_out))


# Example usage:
# create_umap_comparison_file_for_node_type('data.csv', 'node_type1', color_strategy='cluster')
# create_umap_comparison_file_for_node_type('data.csv', 'node_type1', color_strategy='feature')
# create_umap_comparison_file_for_node_type('data.csv', 'node_type1', color_strategy='density')
# create_umap_comparison_with_specific_column('data.csv', 'node_type1', 'your_column_name')


