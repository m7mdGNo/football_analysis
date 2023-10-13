import cv2
import numpy as np



def remove_green_color(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([36,0,0])  
    upper_green = np.array([86, 255, 255])  
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    result = cv2.bitwise_and(image, image, mask=~green_mask)

    return result


def predict_team(player, frame, cluster_model):
    x1, y1, x2, y2 = player
    player_img = frame[y1:y2, x1:x2]
    player_img = cv2.resize(player_img, (32, 32))
    # player_img = remove_green_color(player_img)
    player_img = player_img.reshape(1, -1)
    label = cluster_model.predict(player_img)

    if label == 0:
        return "team1"
    return "team2"


def cluster_players_team(frame, players, cluster_model):
    colors = {"team1": (255, 0, 0), "team2": (0, 255, 0)}
    if len(players) >= 300:
        players_imgs = []
        for cords in players:
            x1, y1, x2, y2 = cords
            player_img = cv2.resize(frame[y1:y2, x1:x2], (32, 32))
            # player_img = remove_green_color(player_img)
            players_imgs.append(player_img)
        players_imgs = np.array(players_imgs).reshape(len(players), -2)
        cluster_model.fit(players_imgs)

        for p in players_imgs:
            label = cluster_model.predict(p.reshape(1, -1))
            p = p.reshape(32, 32, 3)

            if label == 0 and colors['team1']==(255, 0, 0):
                colors["team1"] = [int(c) for c in p[16, 14, :]]
            elif label == 1 and colors['team2']==(0, 255, 0):
                colors["team2"] = [int(c) for c in p[16, 14, :]]
            
        return True, colors
    return False, colors
