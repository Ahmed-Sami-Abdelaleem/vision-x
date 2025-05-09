import torch
import numpy as np
import cv2

class TeamAssigner:
    def __init__(self, device="cpu"):
        self.device = device
        self.team_colors = {}
        self.player_team_dict = {}

    def kmeans_pytorch(self, X, num_clusters=2, num_iters=10):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        indices = torch.randperm(X.shape[0])[:num_clusters]
        centroids = X[indices]

        for _ in range(num_iters):
            distances = torch.cdist(X, centroids)
            labels = distances.argmin(dim=1)

            new_centroids = torch.stack([
                X[labels == i].mean(dim=0) if (labels == i).any() else centroids[i]
                for i in range(num_clusters)
            ])

            if torch.allclose(new_centroids, centroids, atol=1e-4):
                break

            centroids = new_centroids

        return labels.cpu().numpy(), centroids.cpu().numpy()

    def get_clustering_model(self, image):
        # Reshape image to 2D
        image_2d = image.reshape(-1, 3)
        labels, centers = self.kmeans_pytorch(image_2d, num_clusters=2)
        return labels, centers

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        image = frame[y1:y2, x1:x2]
        top_half_image = image[:image.shape[0] // 2, :]

        labels, centers = self.get_clustering_model(top_half_image)
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = centers[player_cluster]
      

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []

        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        labels, centers = self.kmeans_pytorch(np.array(player_colors), num_clusters=2)

        self.kmeans_centers = centers
        self.team_colors[1] = centers[0]
        self.team_colors[2] = centers[1]
        print(f"Team colors detected: {self.team_colors}")
        print(f"Team 1 color: {self.team_colors[1]}")
        print(f"Team 2 color: {self.team_colors[2]}")

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        # Compare to kmeans_centers
        player_tensor = torch.tensor(player_color).unsqueeze(0)
        centers_tensor = torch.tensor(self.kmeans_centers)
        distances = torch.cdist(player_tensor, centers_tensor)
        team_id = distances.argmin().item() + 1

        if player_id == 91:
            team_id = 1

        self.player_team_dict[player_id] = team_id
        return team_id
    
    