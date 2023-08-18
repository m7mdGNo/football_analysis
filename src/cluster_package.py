import cv2
import numpy as np


def predict_team(player, frame, kmeans):
    x1, y1, x2, y2 = player
    player_img = frame[y1:y2, x1:x2]
    player_img = cv2.resize(player_img, (32, 32))
    player_img = player_img.reshape(1, -1)
    label = kmeans.predict(player_img)

    if label == 0:
        return "team1"
    return "team2"


def cluster_players_team(frame, players, kmeans, conf=0.25):
    colors = {"team1": (255, 0, 0), "team2": (0, 255, 0)}
    if len(players) > 10:
        players_imgs = []
        for cords in players:
            x1, y1, x2, y2 = cords
            player_img = cv2.resize(frame[y1:y2, x1:x2], (32, 32))
            players_imgs.append(player_img)
        players_imgs = np.array(players_imgs).reshape(len(players), -2)
        kmeans.fit(players_imgs)

        for p in players_imgs:
            label = kmeans.predict(p.reshape(1, -1))
            p = p.reshape(32, 32, 3)

            if label == 0:
                colors["team1"] = [int(c) for c in p[16, 15, :]]
            else:
                colors["team2"] = [int(c) for c in p[16, 15, :]]
        return True, colors
    return False, colors
