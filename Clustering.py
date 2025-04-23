import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples

# import seaborn as sns
from collections import Counter


# 1. Data loading and preprocessing function
def load_and_prepare_data(file_path):
    # Load data
    print(f"Loading data: {file_path}")
    data = pd.read_csv(file_path)

    # Basic data overview
    print(f"Data shape: {data.shape}")
    print("Data columns:", data.columns.tolist())

    # Basic data cleaning
    # Remove missing values
    if data.isnull().sum().sum() > 0:
        print("Processing missing values...")
        data = data.dropna()
        print(f"Data shape after processing: {data.shape}")

    # Group by video ID, calculate clustering features
    print("Grouping by video ID, calculating features...")

    # Create sentiment count columns
    data["positive"] = (data["sentiment"] == "Positive").astype(int)
    data["negative"] = (data["sentiment"] == "Negative").astype(int)
    data["neutral"] = (data["sentiment"] == "Neutral").astype(int)

    # EDA
    total_rows = len(data)
    print(f"数据集总共有 {total_rows} 条数据")
    # 查看每个值的出现次数
    sentiment_counts = data["sentiment"].value_counts()
    print("\nsentiment列中可能的值和其出现次数:")
    print(sentiment_counts)

    # positive_count = data["sentiment"].value_counts().get("POSITIVE", 0)
    # print(f"POSITIVE的数据有 {positive_count} 个")
    # negative_count = data["sentiment"].value_counts().get("NEGATIVE", 0)
    # print(f"NEGATIVE的数据有 {negative_count} 个")
    # neutral_count = data["sentiment"].value_counts().get("NEUTRAL", 0)
    # print(f"NEUTRAL的数据有 {neutral_count} 个")

    # Aggregate features by video ID
    video_features = (
        data.groupby("video_id")
        .agg(
            {
                "positive": "mean",  # Positive comment ratio
                "negative": "mean",  # Negative comment ratio
                "neutral": "mean",  # Neutral comment ratio
                "score": "mean",  # Average sentiment score
                "likeCount": "sum",  # Total comment likes
                "comment_text": "count",  # Comment count
            }
        )
        .reset_index()
    )

    # Rename columns
    video_features = video_features.rename(columns={"comment_text": "comment_count"})

    # Add video metadata
    video_metadata = data.groupby("video_id").first()[["category"]].reset_index()

    # Merge features and metadata
    video_features = pd.merge(video_features, video_metadata, on="video_id")

    # Convert categorical variables to dummy variables
    video_features = pd.get_dummies(
        video_features, columns=["category"], prefix="category"
    )

    # Save feature names and remove ID column
    feature_names = video_features.columns.tolist()
    feature_names.remove("video_id")

    # Feature standardization
    print("Feature standardization...")
    features_for_scaling = video_features[feature_names].copy()
    scaler = StandardScaler()
    video_features_scaled = scaler.fit_transform(features_for_scaling)

    print(f"Prepared feature dimensions for clustering: {video_features_scaled.shape}")
    print(f"Feature names: {feature_names}")

    return data, video_features, video_features_scaled, feature_names


# 2. Determining optimal number of clusters
def find_optimal_k(features, k_range=range(2, 11)):

    print("Finding optimal number of clusters...")
    silhouette_scores = []

    for k in k_range:
        print(f"Trying clustering with k={k}...")
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(features)

        # Calculate silhouette coefficient
        silhouette_avg = silhouette_score(features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"Silhouette coefficient for k={k}: {silhouette_avg:.4f}")

    # Find optimal number of clusters
    best_k_idx = np.argmax(silhouette_scores)
    best_k = k_range[best_k_idx]

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, "o-", color="blue")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette coefficient")
    plt.title("Silhouette coefficients for different numbers of clusters")
    plt.axvline(x=best_k, color="red", linestyle="--")
    plt.grid(True)
    plt.savefig("silhouette_scores.png")
    plt.close()

    print(
        f"Optimal number of clusters: {best_k}, Silhouette score: {silhouette_scores[best_k_idx]:.4f}"
    )

    return best_k, silhouette_scores


# 3. K-means++ clustering implementation
def perform_kmeans(features, n_clusters):
    print(f"Performing K-means++ clustering (k={n_clusters})...")

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=42,
    )

    cluster_labels = kmeans.fit_predict(features)
    cluster_centers = kmeans.cluster_centers_

    # Output the size of each cluster
    cluster_sizes = Counter(cluster_labels)
    for cluster_id, size in sorted(cluster_sizes.items()):
        print(f"Cluster {cluster_id}: {size} samples")

    return cluster_labels, cluster_centers


# 4. Outlier detection and handling
def detect_and_handle_outliers(features, cluster_labels, cluster_centers):
    print("Detecting and handling outliers...")

    # Calculate distance from each point to its cluster center
    distances = np.zeros(len(features))
    for i in range(len(features)):
        cluster_id = cluster_labels[i]
        distances[i] = np.linalg.norm(features[i] - cluster_centers[cluster_id])

    # Set outlier threshold (using 95th percentile)
    threshold = np.percentile(distances, 95)
    outliers = distances > threshold
    outlier_indices = np.where(outliers)[0]

    print(f"Detected {len(outlier_indices)} outliers (threshold: {threshold:.4f})")

    # If no outliers, return original labels
    if len(outlier_indices) == 0:
        return cluster_labels

    # Copy original labels
    updated_labels = cluster_labels.copy()

    # Apply hierarchical clustering to outliers
    if len(outlier_indices) > 1:  # Need at least 2 points for clustering
        outlier_features = features[outlier_indices]

        # Determine number of outlier clusters (max 3 or number of outliers, whichever is smaller)
        n_outlier_clusters = min(3, len(outlier_indices))

        # Apply hierarchical clustering
        hc = AgglomerativeClustering(n_clusters=n_outlier_clusters, linkage="ward")

        outlier_cluster_labels = hc.fit_predict(outlier_features)

        # Assign new cluster labels to outliers
        max_original_cluster = np.max(cluster_labels)
        for i, idx in enumerate(outlier_indices):
            updated_labels[idx] = max_original_cluster + 1 + outlier_cluster_labels[i]

        # Output outlier cluster information
        outlier_sizes = Counter(outlier_cluster_labels)
        for cluster_id, size in sorted(outlier_sizes.items()):
            new_id = max_original_cluster + 1 + cluster_id
            print(f"Outlier cluster {new_id}: {size} samples")

    return updated_labels


# 5. Metadata enhanced cluster analysis
def analyze_clusters_with_metadata(data, video_features, cluster_labels):
    print("Analyzing relationship between clusters and metadata...")

    try:
        # Add cluster labels to video_features
        video_features_with_clusters = video_features.copy()
        video_features_with_clusters["cluster"] = cluster_labels

        # Check data types, ensure all numeric columns are of correct type
        print("Checking data types...")
        for col in video_features_with_clusters.columns:
            # print(f"Column '{col}': {video_features_with_clusters[col].dtype}")

            # Try to convert columns that should be numeric
            if (
                col not in ["video_id", "cluster"]
                and video_features_with_clusters[col].dtype == "object"
            ):
                try:
                    # Try to convert to numeric type
                    video_features_with_clusters[col] = pd.to_numeric(
                        video_features_with_clusters[col], errors="coerce"
                    )
                    print(f"  Column '{col}' converted to numeric type")
                except Exception as e:
                    print(f"  Warning: Error converting column '{col}': {e}")

                # Check if there are NaN values after conversion
                if video_features_with_clusters[col].isna().any():
                    na_count = video_features_with_clusters[col].isna().sum()
                    print(f"  Warning: Column '{col}' has {na_count} NaN values")

                    # Option to fill NaN values
                    if (
                        na_count < len(video_features_with_clusters) * 0.5
                    ):  # If NaN values don't exceed 50%
                        mean_val = video_features_with_clusters[col].mean()
                        video_features_with_clusters[col] = (
                            video_features_with_clusters[col].fillna(mean_val)
                        )
                        print(f"  Filled NaN values with mean {mean_val}")

        # Select only numeric columns for analysis
        numeric_columns = video_features_with_clusters.select_dtypes(
            include=["number"]
        ).columns.tolist()
        if "cluster" in numeric_columns:
            numeric_columns.remove("cluster")  # Keep cluster for grouping
        if "video_id" in numeric_columns:
            numeric_columns.remove("video_id")  # Exclude ID column

        print(f"Numeric columns for statistics calculation: {numeric_columns}")

        # 1. Analyze feature distribution for each cluster
        print("Calculating cluster feature distributions...")
        cluster_profiles = video_features_with_clusters.groupby("cluster")[
            numeric_columns
        ].mean()

        # 2. Analyze relationship between clusters and categories
        print("Analyzing relationship between clusters and categories...")
        # Find category columns
        category_columns = [
            col for col in video_features.columns if col.startswith("category_")
        ]

        # Calculate proportion of each category in each cluster
        category_distribution = {}
        for category in category_columns:
            category_name = category.replace("category_", "")
            # Ensure boolean values are converted to integers (0 and 1)
            series = video_features_with_clusters[category].astype(int)
            category_by_cluster = video_features_with_clusters.groupby("cluster")[
                category
            ].apply(
                lambda x: x.astype(
                    int
                ).mean()  # Explicitly convert to int before calculating
            )
            category_distribution[category_name] = category_by_cluster

        # 3. Calculate cluster sizes
        cluster_sizes = (
            video_features_with_clusters["cluster"].value_counts().sort_index()
        )

        # 4. Calculate average sentiment distribution for each cluster
        sentiment_columns = [
            col for col in ["positive", "negative", "neutral"] if col in numeric_columns
        ]

        if sentiment_columns:
            sentiment_distribution = video_features_with_clusters.groupby("cluster")[
                sentiment_columns
            ].mean()
        else:
            print("Warning: No sentiment columns found for analysis")
            sentiment_distribution = pd.DataFrame()

        # Create analysis results dictionary
        cluster_analysis = {
            "cluster_profiles": cluster_profiles,
            "category_distribution": category_distribution,
            "cluster_sizes": cluster_sizes,
            "sentiment_distribution": sentiment_distribution,
            "video_features_with_clusters": video_features_with_clusters,
        }

        # Print some basic analysis results
        print("\nCluster sizes:")
        print(cluster_sizes)

        if not sentiment_distribution.empty:
            print("\nCluster sentiment distribution:")
            print(sentiment_distribution)

        return cluster_analysis

    except Exception as e:
        import traceback

        print(f"Error analyzing clusters: {e}")
        print(traceback.format_exc())

        # Return an empty analysis results dictionary
        return {
            "cluster_profiles": pd.DataFrame(),
            "category_distribution": {},
            "cluster_sizes": pd.Series(),
            "sentiment_distribution": pd.DataFrame(),
            "video_features_with_clusters": video_features.copy(),
        }


# 6. Visualization functions
def visualize_clusters(features, labels, centers=None, method="tsne"):
    print(f"Visualizing clustering results using {method.upper()}...")

    # Reduce to 2D
    if method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(features)
        title = "PCA Visualization of Clustering Results"
    else:  # Default to t-SNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        reduced_features = reducer.fit_transform(features)
        title = "t-SNE Visualization of Clustering Results"

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use scatter plot to display each point
    scatter = ax.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.7,
        s=50,
    )

    # If cluster centers provided, display them (works for PCA, not for t-SNE)
    if centers is not None and method.lower() == "pca":
        centers_reduced = reducer.transform(centers)
        ax.scatter(
            centers_reduced[:, 0],
            centers_reduced[:, 1],
            marker="X",
            s=200,
            c="red",
            alpha=1,
            edgecolors="black",
        )

    # Add legend and title
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)

    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    # Save figure
    plt.tight_layout()
    plt.savefig(f"cluster_visualization_{method}.png")
    plt.close()

    return fig


def plot_silhouette(features, labels):
    print("Plotting silhouette coefficient diagram...")

    # Calculate silhouette coefficient for each sample
    silhouette_vals = silhouette_samples(features, labels)

    # Calculate average silhouette coefficient
    avg_silhouette = silhouette_score(features, labels)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    y_lower = 10
    n_clusters = len(np.unique(labels))

    # Plot silhouette for each cluster
    for i in range(n_clusters):
        # Get silhouette coefficient values for current cluster
        ith_cluster_silhouette_values = silhouette_vals[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.viridis(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Add cluster label next to silhouette plot
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f"Cluster {i}")

        # Calculate y_lower for next cluster
        y_lower = y_upper + 10

    # Add vertical line for average silhouette coefficient
    ax.axvline(x=avg_silhouette, color="red", linestyle="--")
    ax.set_title(
        f"Silhouette Plot for Each Cluster (Average Silhouette Score: {avg_silhouette:.4f})"
    )
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")

    # Set x-axis range
    ax.set_xlim([-0.1, 1])

    # Set y-axis range
    ax.set_ylim([0, y_lower])

    plt.tight_layout()
    plt.savefig("silhouette_plot.png")
    plt.close()

    return fig


def plot_cluster_profiles(cluster_analysis):
    print("Plotting cluster feature distributions...")

    # 1. Plot sentiment distribution
    sentiment_distribution = cluster_analysis["sentiment_distribution"]

    fig, ax = plt.subplots(figsize=(12, 8))
    sentiment_distribution.plot(kind="bar", ax=ax)
    ax.set_title("Sentiment Distribution by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion")
    ax.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig("cluster_sentiment_distribution.png")
    plt.close()

    # 2. Plot category distribution
    category_distribution = pd.DataFrame(cluster_analysis["category_distribution"])

    # Use tab20 colormap for distinct colors (up to 20 colors)
    n_clusters = len(category_distribution.columns)
    if n_clusters <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

    fig, ax = plt.subplots(figsize=(14, 8))
    category_distribution.T.plot(kind="bar", ax=ax, color=colors)
    ax.set_title("Category Distribution by Cluster")
    ax.set_xlabel("Category")
    ax.set_ylabel("Proportion")
    ax.legend(title="Cluster")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("cluster_category_distribution.png")
    plt.close()

    # 3. Plot cluster sizes
    cluster_sizes = cluster_analysis["cluster_sizes"]

    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_sizes.plot(kind="bar", ax=ax)
    ax.set_title("Cluster Sizes")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig("cluster_sizes.png")
    plt.close()

    # 4. Plot feature heatmap - using pure matplotlib instead of seaborn
    cluster_profiles = cluster_analysis["cluster_profiles"]

    # Standardize feature values for better comparison
    normalized_profiles = (
        cluster_profiles - cluster_profiles.mean()
    ) / cluster_profiles.std()

    fig, ax = plt.subplots(figsize=(16, 10))

    # Create heatmap
    im = ax.imshow(normalized_profiles.values, cmap="coolwarm")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Standardized Value")

    # Set tick labels
    ax.set_xticks(np.arange(len(normalized_profiles.columns)))
    ax.set_yticks(np.arange(len(normalized_profiles.index)))
    ax.set_xticklabels(normalized_profiles.columns)
    ax.set_yticklabels(normalized_profiles.index)

    # Rotate x-axis labels for better display
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations in each cell
    for i in range(len(normalized_profiles.index)):
        for j in range(len(normalized_profiles.columns)):
            value = normalized_profiles.iloc[i, j]
            text_color = "white" if abs(value) > 1.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color)

    ax.set_title("Cluster Feature Heatmap (Standardized Values)")
    ax.set_ylabel("Cluster")

    plt.tight_layout()
    plt.savefig("cluster_feature_heatmap.png")
    plt.close()

    return fig


# 7. Cluster evaluation
def evaluate_clustering(features, labels):
    print("Evaluating clustering quality...")

    # 1. Calculate silhouette coefficient (overall and for each sample)
    silhouette_avg = silhouette_score(features, labels)
    silhouette_samples_values = silhouette_samples(features, labels)

    # 2. Calculate average silhouette coefficient for each cluster
    cluster_silhouette = {}
    for i in np.unique(labels):
        cluster_silhouette[i] = np.mean(silhouette_samples_values[labels == i])

    # 3. Calculate number of samples in each cluster
    cluster_sizes = Counter(labels)

    # 4. Calculate average distance within each cluster (intra-cluster cohesion)
    intra_cluster_distances = {}
    for i in np.unique(labels):
        cluster_points = features[labels == i]
        if len(cluster_points) <= 1:
            intra_cluster_distances[i] = 0
        else:
            # Calculate average distance between all pairs of points in the cluster
            distances = []
            for j in range(len(cluster_points)):
                for k in range(j + 1, len(cluster_points)):
                    distances.append(
                        np.linalg.norm(cluster_points[j] - cluster_points[k])
                    )
            intra_cluster_distances[i] = np.mean(distances) if distances else 0

    # Summarize evaluation metrics
    evaluation_metrics = {
        "silhouette_avg": silhouette_avg,
        "cluster_silhouette": cluster_silhouette,
        "cluster_sizes": cluster_sizes,
        "intra_cluster_distances": intra_cluster_distances,
    }

    # Print evaluation results
    print(f"Average silhouette score: {silhouette_avg:.4f}")
    print("\nSilhouette scores by cluster:")
    for cluster_id, silhouette in sorted(cluster_silhouette.items()):
        print(f"Cluster {cluster_id}: {silhouette:.4f}")

    return evaluation_metrics


# 8. Main function
def main():
    # 1. Load data
    data, video_features, features_scaled, feature_names = load_and_prepare_data(
        "youtube_comments_with_3sentiment.csv"
    )

    # 2. Find optimal number of clusters
    best_k, silhouette_scores = find_optimal_k(features_scaled)
    print(f"Optimal number of clusters: {best_k}")

    # 3. Perform K-means++ clustering
    labels, centers = perform_kmeans(features_scaled, best_k)

    # 4. Handle outliers
    labels = detect_and_handle_outliers(features_scaled, labels, centers)

    # 5. Analyze clustering results
    cluster_analysis = analyze_clusters_with_metadata(data, video_features, labels)

    # 6. Visualization
    visualize_clusters(features_scaled, labels, centers, method="pca")
    visualize_clusters(features_scaled, labels, centers, method="tsne")
    plot_silhouette(features_scaled, labels)
    plot_cluster_profiles(cluster_analysis)

    # 7. Evaluate clustering quality
    metrics = evaluate_clustering(features_scaled, labels)

    # 8. Save results
    video_features["cluster"] = labels
    video_features.to_csv("youtube_videos_clustered.csv", index=False)

    print(
        "\nClustering analysis complete! Results saved to 'youtube_videos_clustered.csv'"
    )

    return data, cluster_analysis, metrics


if __name__ == "__main__":
    main()
