from .postprocess import corners_postprocess
from .utils.utils import get_middle
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from .CenterNet import centernet
import cv2
import torch
from torchvision import transforms
import numpy as np


def detect_all_game_objects(frame, model, conf=0.5):
    """
    Detects game objects (players,corners) in the frame using a trained model
    """
    names = model.names
    results = model(frame, verbose=False, conf=conf, imgsz=800)
    players_boxes = []
    corners = {}
    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            class_name = names[class_id]

            if class_name =='player' and score>conf:
                if score >= conf:
                    players_boxes.append(([x1,y1,x2-x1,y2-y1],score,class_id))

            elif class_name != "c" and class_name in [str(i) for i in range(50)]:
                corners.update({class_name: get_middle(x1, y1, x2 - x1, y2 - y1)})

            else:
                try:
                    corners["c"].append(get_middle(x1, y1, x2 - x1, y2 - y1))
                except:
                    corners.update({"c": [get_middle(x1, y1, x2 - x1, y2 - y1)]})

    corners = corners_postprocess(corners)
    return corners,players_boxes
    


def detect_game_objects(frame, model, conf=0.15):
    """
    Detects game objects (players,goalkeeper,refree,ball) in the frame using a trained model
    """
    names = model.names
    results = model(frame, verbose=False, conf=0.15, imgsz=800)
    bboxs = []
    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if names[class_id] =='player' and score>conf:
                bboxs.append(([x1,y1,x2-x1,y2-y1],score,class_id))
            elif names[class_id] =='ball':
                bboxs.append(([x1,y1,x2-x1,y2-y1],score,class_id))
            
    return bboxs


def detect_corners(frame, model, conf=0.15):
    """
    Detects field corners in the frame using a trained model
    """
    names = model.names
    results = model(frame, conf=conf, verbose=False,imgsz=800)
    corners = {}
    for result in results[0]:
        classes = result.boxes.cls.tolist()
        bboxs = result.boxes.xyxy.tolist()
        for i, bbox in enumerate(bboxs):
            index = int(classes[i])
            class_name = names[index]

            if class_name != "c":
                x1, y1, x2, y2 = [int(p) for p in bbox]
                corners.update({class_name: get_middle(x1, y1, x2 - x1, y2 - y1)})
            else:
                x1, y1, x2, y2 = [int(p) for p in bbox]
                try:
                    corners["c"].append(get_middle(x1, y1, x2 - x1, y2 - y1))
                except:
                    corners.update({"c": [get_middle(x1, y1, x2 - x1, y2 - y1)]})
    corners = corners_postprocess(corners)

    return corners


def detect_ball(frame,model,last_loc):
    """
    Detects ball in the frame using a trained model
    """

    PRED_CENTER = None
    INPUT_WIDTH = 1280
    INPUT_HEIGHT = 720
    MODEL_SCALE = 8

    DELTA = 8  # tracking const variable
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # quick
    img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))  # quick
    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    input_tensor = preprocess(img)

    with torch.no_grad():
        hm, offset = model(input_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).float().unsqueeze(0))

    hm = torch.sigmoid(hm)
    hm = hm.cpu().numpy().squeeze(0).squeeze(0)
    offset = offset.cpu().numpy().squeeze(0)

    if np.max(hm, axis=None) < 0.01:
        PRED_CENTER = None
    elif np.max(hm, axis=None) > 0.4:
        PRED_CENTER = np.unravel_index(np.argmax(hm, axis=None), hm.shape)
    else:
        PRED_CENTER = None
        if last_loc is not None:
            sub_hm = hm[
                last_loc[0] - DELTA : last_loc[0] + DELTA,
                last_loc[1] - DELTA : last_loc[1] + DELTA,
            ]
            if np.max(sub_hm, axis=None) > 0.02:
                PRED_CENTER = np.unravel_index(
                    np.argmax(sub_hm, axis=None), sub_hm.shape
                ) + (np.array(last_loc) - DELTA)
                PRED_CENTER = tuple(PRED_CENTER)
            
    if PRED_CENTER is not None:
        last_loc = PRED_CENTER
        score = hm[PRED_CENTER]

        arr = (
            np.array(
                [PRED_CENTER[1], PRED_CENTER[0]]
                + offset[:, PRED_CENTER[0], PRED_CENTER[1]]
            )
            * MODEL_SCALE
        )

        # for point, score in zip(points, scores):
        u = round(arr[0] * frame.shape[1] / INPUT_WIDTH)
        v = round(arr[1] * frame.shape[0] / INPUT_HEIGHT)

        return (u, v),last_loc
    return  (),last_loc