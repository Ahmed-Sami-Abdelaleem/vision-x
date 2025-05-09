import sys 
import numpy as np
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self, players, ball_bbox):
        # First check if ball_bbox is valid (not NaN or empty)
        if (not ball_bbox or  # Empty bbox
            len(ball_bbox) != 4 or  # Not 4 coordinates
            any(np.isnan(coord) for coord in ball_bbox)):  # Contains NaN values
            return -1  # Return -1 if ball position is invalid

        try:
            ball_position = get_center_of_bbox(ball_bbox)
        except:
            return -1  # Return -1 if center calculation fails

        minimum_distance = float('inf')
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']
            
            # Skip players with invalid bboxes
            if len(player_bbox) != 4 or any(np.isnan(coord) for coord in player_bbox):
                continue

            try:
                distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
                distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
                distance = min(distance_left, distance_right)
            except:
                continue  # Skip if distance calculation fails

            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player