import random
import cv2
import numpy as np
from sklearn.cluster import KMeans,DBSCAN,MeanShift
from .detection_package import detect_corners, detect_game_objects, detect_all_game_objects,detect_ball
from .homography_package import calculate_homography, get_transformed_point
from .utils.drawing import draw_player_point, draw_player_rect, draw_point
from .utils.utils import get_player_point
from .CenterNet import centernet
import torch 


class BaseModel:
    def __init__(self, corners_model,game_objects_model) -> None:
        # corners coords in top view image
        self.est_points = {
            "1": (589, 48),
            "2": (588, 160),
            "3": (588, 242),
            "4": (588, 379),
            "5": (589, 462),
            "6": (589, 574),
            "7": (564, 242),
            "8": (563, 379),
            "9": (507, 160),
            "10": (507, 257),
            "11": (505, 365),
            "12": (507, 462),
            "13": (320, 48),
            "14": (319, 244),
            "15": (320, 379),
            "16": (320, 392),
            "23": (51, 48),
            "24": (51, 160),
            "25": (51, 242),
            "26": (51, 379),
            "27": (51, 462),
            "28": (51, 574),
            "21": (77, 242),
            "22": (77, 379),
            "17": (134, 160),
            "18": (134, 257),
            "19": (133, 365),
            "20": (134, 462),
            "29": (274, 312),
            "30": (366, 312),
            "31": (320, -31774.965),
            "32": (532, 313),
            "33": (107, 313),
            "34": (563, 162),
            "35": (563, 464),
            "36": (76, 162),
            "37": (76, 464),

            "38": (506,51),
            "39": (506,574),
            "40": (131,51),
            "41": (131,574),

            "42": (364,51), 
            "43": (364,574),
            "44": (274,51),
            "45": (274,574),


        }

        # top view image
        self.est = cv2.imread("./images/field_est.jpg")

        # tracking and smoothing (working in video part)
        self.homo_history = []
        self.homo_history_mrg = []

        # setting device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loading models
        # self.all_objects_model = all_objects_model
        self.corners_model = corners_model
        self.game_objects_model = game_objects_model
        self.ball_model = centernet()
        self.ball_model.load_state_dict(torch.load('models/soccer.pth'))
        self.ball_model.to(self.device)
        self.ball_model.eval()


        # initialize colors for teams 'will change after clustering the teams'
        self.colors = {"team1": (255, 0, 0), "team2": (0, 255, 0)}

        # cluster model
        self.cluster_model = KMeans(n_clusters=2, random_state=0)
        self.clustered = False

        # params for savgol filter that used in smoothing
        self.window_size = 15
        self.poly_order = 1


        self.official_width = 109.728 #in meter
        self.official_height = 73.152 #in meter

        self.width_percentage = self.official_width/(self.est_points['1'][0]-self.est_points['23'][0])
        self.height_percentage = self.official_height/(self.est_points['1'][1]-self.est_points['28'][1])

    def detect_pitch_corners(self, img, conf=0.15):
        """Detects corners of the pitch"""
        return detect_corners(img, self.corners_model, conf=conf)
    
    def detect_game_objects(self, img, conf=0.5):
        """Detects players in the pitch"""
        return detect_game_objects(img, self.game_objects_model, conf=conf)

    def detect_ball(self,img,last_loc):
        """Detects ball in the pitch"""
        return detect_ball(img,self.ball_model,last_loc)

    # def detect_all_game_objects(self, img, conf=0.5):
    #     """Detects players in the pitch"""
    #     return detect_all_game_objects(img, self.all_objects_model, conf=conf)
    
    def warp_top_view(self,img,H):
        top_view = cv2.warpPerspective(
                    img, H, (self.est.shape[1], self.est.shape[0])
                )
        top_view = cv2.addWeighted(self.est, 0.8, top_view, 1, 1)
        return top_view
    
    def warp_merge_view(self,img,H):
        merge_view = cv2.warpPerspective(
                    self.est, H, (img.shape[1], img.shape[0])
                )
        merge_view = cv2.addWeighted(img, 1, merge_view, 0.5, 1)
        return merge_view



    def get_top_view_homography(self, corners, thresh=80):
        """Returns top-down view of the field"""
        if len(corners) >= 4:
            pts1 = []
            pts2 = []

            for corner in corners:
                pts1.append(corners[corner])
                pts2.append(self.est_points[str(corner)])

            m1 = calculate_homography(np.float32(pts2), np.float32(pts1), thresh)
            try:
                m = np.linalg.inv(m1)
            except:
                m = calculate_homography(np.float32(pts1), np.float32(pts2), thresh)
            return m
        return np.full((3, 3), None)

    def get_merge_view_homography(self, corners, thresh=80):
        """return the prespective 2d field merged with the original image"""
        if len(corners) >= 4:
            pts1 = []
            pts2 = []

            for corner in corners:
                pts1.append(corners[corner])
                pts2.append(self.est_points[str(corner)])

            m = calculate_homography(np.float32(pts2), np.float32(pts1), thresh)
            return m
        return np.full((3, 3), None)

    def draw_corners(self, frame, corners):
        """
        draws a circle on each of the detected corners,
        with corner number.
        """
        return draw_point(frame, corners)

    def draw_players_rect(self, frame, players, color=None):
        """
        draws a rectangle on each of the detected players.
        """
        for player in players:
            frame = draw_player_rect(frame, player, color if color else (255, 0, 0))

        return frame

    def draw_estimated_players_points(self, dst, players, H, color=None):
        """
        draws estimated points of players on the given 2d top-down view,
        field image.
        """
        for player in players:
            player_point = get_player_point(player)
            transformed_player = get_transformed_point(player_point, H)
            dst = draw_player_point(
                dst, transformed_player, color if color else (255, 0, 0)
            )

        return dst
