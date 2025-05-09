import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Load and preprocess the image
image_path = "../output_videos/cropped_image.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

# Use top half of the image
top_half_image = image[0:int(image.shape[0]/2), :]
plt.imshow(top_half_image)
plt.show()

# Convert to PyTorch tensor and reshape
image_2d = torch.tensor(top_half_image.reshape(-1, 3), dtype=torch.float32)

# K-Means implementation in PyTorch
def kmeans_pytorch(X, num_clusters=2, num_iters=10):
    # Randomly choose initial cluster centers
    indices = torch.randperm(X.shape[0])[:num_clusters]
    centroids = X[indices]

    for _ in range(num_iters):
        # Compute distances and assign clusters
        distances = torch.cdist(X, centroids)
        labels = distances.argmin(dim=1)

        # Update centroids
        new_centroids = torch.stack([X[labels == i].mean(dim=0) for i in range(num_clusters)])
        
        # Break if centroids do not change
        if torch.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return labels, centroids

# Run KMeans
labels, centroids = kmeans_pytorch(image_2d, num_clusters=2)

# Reshape labels to match image shape
clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

# Display clustered image
plt.imshow(clustered_image)
plt.show()

# Determine the background cluster
corner_clusters = [clustered_image[0, 0].item(), clustered_image[0, -1].item(),
                   clustered_image[-1, 0].item(), clustered_image[-1, -1].item()]
non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
print("Non-player cluster:", non_player_cluster)

player_cluster = 1 - non_player_cluster
print("Player cluster:", player_cluster)

# Get the color of the player cluster
print("Player cluster color (RGB):", centroids[player_cluster])
