from .postprocess import corners_postprocess
from .utils.utils import get_middle


def detect_persons(frame, model, conf=0.5):
    """
    Detects persons (players) in the frame using a pretrained model
    """
    names = model.names
    results = model(frame, conf=conf, verbose=False)
    players = []
    for result in results[0]:
        classes = result.boxes.cls.tolist()
        bboxs = result.boxes.xyxy.tolist()
        for i, bbox in enumerate(bboxs):
            index = int(classes[i])
            class_name = names[index]
            x1, y1, x2, y2 = [int(p) for p in bbox]
            if class_name == "person":
                players.append((x1, y1, x2, y2))
    return players, [], []


def detect_corners(frame, model, conf=0.15):
    """
    Detects field corners in the frame using a trained model
    """
    names = model.names
    results = model(frame, conf=conf, verbose=False,imgsz=640)
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
