def cluster_horizontal_lines(corners: dict):
    group_a = []
    group_b = []
    groub_c = []
    groub_d = []

    for key in corners.keys():
        if not key == "c":
            if int(key) in [1, 2, 3, 4, 5, 6, 23, 24, 25, 26, 27, 28]:
                group_a.append(corners[key])
            elif int(key) in [7, 8, 21, 22]:
                group_b.append(corners[key])
            elif int(key) in [9, 10, 11, 12, 17, 18, 19, 20]:
                groub_c.append(corners[key])
            elif int(key) in [13, 14, 15, 16]:
                groub_d.append(corners[key])
    return group_a, group_b, groub_c, groub_d


def cluster_main_horizontal_lines(corners):
    line1 = []
    line2 = []
    line3 = []
    middle_line = []

    a, b, c, d = cluster_horizontal_lines(corners)
    if a:
        line1.append(a[0])
        line1.append(a[-1])
    if b:
        line2.append(b[0])
        line2.append(b[-1])
    if c:
        line3.append(c[0])
        line3.append(c[-1])
    if d:
        middle_line.append(d[0])
        middle_line.append(d[-1])
    return line1, line2, line3, middle_line


def calculate_slope(line):
    """
    Calculates the slope of a line
    """
    (x, y), (x1, y1) = line
    slope = (y1 - y) / (x1 - x)
    return slope


def get_middle(x1, y1, w, h):
    """
    Calculates the middle point of a rectangle
    """
    newx = (2 * x1 + w) / 2
    newy = (2 * y1 + h) / 2
    return [newx, newy]


def get_player_point(player):
    """
    get the point of player feet
    """
    x1, y1, x2, y2 = player
    return ((x1 + x2) // 2, y2)
