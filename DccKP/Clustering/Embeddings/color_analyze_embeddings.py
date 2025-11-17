

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')
import dcc_utils as dutils


# constants
logger = dutils.get_logger(__name__)
DEBUG = True


def create_umap_with_clustering(csv_file_path, output_file=None):
    """
    Create UMAP visualization with multiple clustering approaches.
    """
    # Read and process data
    df = pd.read_csv(csv_file_path)
    keys = df['key'].values
    feature_columns = [col for col in df.columns if col.startswith('val_')]
    features = df[feature_columns].values
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply UMAP
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = umap_reducer.fit_transform(features_scaled)
    
    # Apply different clustering algorithms
    clustering_methods = {}
    
    # 1. K-Means clustering (try different k values)
    for k in [2, 3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        clustering_methods[f'K-Means (k={k})'] = clusters
    
    # 2. DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(features_scaled)
    clustering_methods['DBSCAN'] = clusters
    
    # 3. Gaussian Mixture Model
    gmm = GaussianMixture(n_components=3, random_state=42)
    clusters = gmm.fit_predict(features_scaled)
    clustering_methods['Gaussian Mixture'] = clusters
    
    # 4. Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=3)
    clusters = hierarchical.fit_predict(features_scaled)
    clustering_methods['Hierarchical'] = clusters
    
    # 5. Clustering on UMAP embedding (often works better)
    kmeans_umap = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters_umap = kmeans_umap.fit_predict(embedding)
    clustering_methods['K-Means on UMAP'] = clusters_umap
    
    # Create visualization
    n_methods = len(clustering_methods)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    colors = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'tab10']
    
    for i, (method_name, clusters) in enumerate(clustering_methods.items()):
        if i < len(axes):
            # Handle noise points in DBSCAN (labeled as -1)
            unique_clusters = np.unique(clusters)
            n_clusters = len(unique_clusters)
            
            if -1 in unique_clusters:  # DBSCAN noise points
                # Plot noise points in gray
                noise_mask = clusters == -1
                axes[i].scatter(embedding[noise_mask, 0], embedding[noise_mask, 1], 
                              c='lightgray', alpha=0.5, s=30, label='Noise')
                
                # Plot clustered points
                clustered_mask = clusters != -1
                if np.any(clustered_mask):
                    scatter = axes[i].scatter(embedding[clustered_mask, 0], embedding[clustered_mask, 1], 
                                            c=clusters[clustered_mask], cmap='tab10', alpha=0.7, s=50)
            else:
                scatter = axes[i].scatter(embedding[:, 0], embedding[:, 1], 
                                        c=clusters, cmap='tab10', alpha=0.7, s=50)
            
            axes[i].set_title(f'{method_name}\n(n_clusters: {n_clusters})')
            axes[i].set_xlabel('UMAP Dimension 1')
            axes[i].set_ylabel('UMAP Dimension 2')
            axes[i].grid(True, alpha=0.3)
            
            # Add silhouette score if applicable
            if len(unique_clusters) > 1 and -1 not in unique_clusters:
                sil_score = silhouette_score(features_scaled, clusters)
                axes[i].text(0.02, 0.98, f'Silhouette: {sil_score:.3f}', 
                           transform=axes[i].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for i in range(len(clustering_methods), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Clustering plot saved to: {output_file}")
    else:
        plt.show()
    
    return embedding, clustering_methods

def create_detailed_cluster_analysis(csv_file_path, n_clusters=3):
    """
    Create a detailed analysis with the best clustering approach.
    """
    # Read and process data
    df = pd.read_csv(csv_file_path)
    keys = df['key'].values
    feature_columns = [col for col in df.columns if col.startswith('val_')]
    features = df[feature_columns].values
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply UMAP
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = umap_reducer.fit_transform(features_scaled)
    
    # Apply clustering on UMAP embedding (often gives better visual results)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embedding)
    
    # Create detailed visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Basic clustering
    scatter1 = axes[0, 0].scatter(embedding[:, 0], embedding[:, 1], 
                                  c=clusters, cmap='tab10', alpha=0.7, s=60)
    axes[0, 0].set_title(f'UMAP with K-Means Clustering (k={n_clusters})')
    axes[0, 0].set_xlabel('UMAP Dimension 1')
    axes[0, 0].set_ylabel('UMAP Dimension 2')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add cluster centers
    centers = kmeans.cluster_centers_
    axes[0, 0].scatter(centers[:, 0], centers[:, 1], 
                      c='red', marker='x', s=200, linewidths=3, label='Centroids')
    axes[0, 0].legend()
    
    # Plot 2: With sample labels
    colors = plt.cm.tab10(clusters)
    axes[0, 1].scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.7, s=60)
    for i, (x, y) in enumerate(embedding):
        axes[0, 1].annotate(f'{i}', (x, y), xytext=(2, 2), 
                           textcoords='offset points', fontsize=8)
    axes[0, 1].set_title('UMAP with Sample Labels')
    axes[0, 1].set_xlabel('UMAP Dimension 1')
    axes[0, 1].set_ylabel('UMAP Dimension 2')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cluster size analysis
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    axes[1, 0].bar(unique_clusters, counts, color=plt.cm.tab10(unique_clusters))
    axes[1, 0].set_title('Cluster Sizes')
    axes[1, 0].set_xlabel('Cluster ID')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        axes[1, 0].text(unique_clusters[i], count + 0.1, str(count), 
                       ha='center', va='bottom')
    
    # Plot 4: Distance from cluster centers
    distances = []
    for i, cluster_id in enumerate(clusters):
        center = centers[cluster_id]
        dist = np.linalg.norm(embedding[i] - center)
        distances.append(dist)
    
    scatter4 = axes[1, 1].scatter(embedding[:, 0], embedding[:, 1], 
                                  c=distances, cmap='coolwarm', alpha=0.7, s=60)
    axes[1, 1].set_title('Distance from Cluster Centers')
    axes[1, 1].set_xlabel('UMAP Dimension 1')
    axes[1, 1].set_ylabel('UMAP Dimension 2')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter4, ax=axes[1, 1], label='Distance from Center')
    
    plt.tight_layout()
    plt.show()
    
    # Print cluster analysis
    print("\nCluster Analysis:")
    print(f"Number of clusters: {n_clusters}")
    print(f"Silhouette score: {silhouette_score(embedding, clusters):.3f}")
    
    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {np.sum(mask)} samples")
        print(f"  Sample indices: {np.where(mask)[0].tolist()}")
        if len(keys) > 0:
            cluster_keys = [keys[i][:30] + '...' if len(keys[i]) > 30 else keys[i] 
                           for i in np.where(mask)[0]]
            print(f"  Sample keys: {cluster_keys}")
    
    return embedding, clusters, kmeans

def find_optimal_clusters(csv_file_path, max_clusters=10):
    """
    Find optimal number of clusters using elbow method and silhouette analysis.
    """
    # Read and process data
    df = pd.read_csv(csv_file_path)
    feature_columns = [col for col in df.columns if col.startswith('val_')]
    features = df[feature_columns].values
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply UMAP
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = umap_reducer.fit_transform(features_scaled)
    
    # Test different numbers of clusters
    k_range = range(2, min(max_clusters + 1, len(features)))
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embedding)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(embedding, clusters))
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow plot
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette plot
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_title('Silhouette Score vs Number of Clusters')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.grid(True, alpha=0.3)
    
    # Mark best silhouette score
    best_k = k_range[np.argmax(silhouette_scores)]
    ax2.axvline(x=best_k, color='red', linestyle='--', alpha=0.7)
    ax2.text(best_k, max(silhouette_scores), f'Best k={best_k}', 
             ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Recommended number of clusters: {best_k}")
    print(f"Best silhouette score: {max(silhouette_scores):.3f}")
    
    return best_k, silhouette_scores

# Example usage
if __name__ == "__main__":
    # load configuration
    config = dutils.load_config()

    # get the node embedding file names
    file_name = config.get(dutils.KEY_INFERENCE).get(dutils.KEY_NODE_EMBEDDINGS_BY_TYPE)

    # for node_type in ['gene', 'trait', 'gene_set', 'factor']:
    for node_type in ['gene']:
        csv_file = file_name.format(node_type)
    # csv_file = 'your_data.csv'
    
        print("1. Finding optimal number of clusters...")
        optimal_k, scores = find_optimal_clusters(csv_file)

        print(f"\n2. Creating clustering comparison...")
        embedding, clustering_results = create_umap_with_clustering(csv_file, 'umap_clustering.png')

        print(f"\n3. Detailed analysis with k={optimal_k}...")
        embedding, clusters, model = create_detailed_cluster_analysis(csv_file, n_clusters=optimal_k)

