import cv2


def draw_point(frame, points):
    for point in points:
        x, y = points[point]
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            str(point),
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
    return frame


def draw_player_rect(frame, player, color):
    x1, y1, x2, y2 = player
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    return frame


def draw_player_point(frame, player, color,size=12):
    x, y = player
    cv2.circle(frame, (int(x), int(y)), size, color, -1)
    # cv2.circle(frame, (int(x), int(y)), size, (255,255,255), 2)
    return frame
