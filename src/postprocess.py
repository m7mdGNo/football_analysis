from .utils.utils import cluster_main_horizontal_lines, calculate_slope, find_intersection_point


def invert_corners(label, corners_labels, index):
    """
    Invert the corners based on if the corner in the ,
    first half or in the second half.
    """
    if label == 1:
        corners_labels[index] = 23
    elif label == 2:
        corners_labels[index] = 24
    elif label == 3:
        corners_labels[index] = 25
    elif label == 4:
        corners_labels[index] = 26
    elif label == 5:
        corners_labels[index] = 27
    elif label == 6:
        corners_labels[index] = 28
    elif label == 7:
        corners_labels[index] = 21
    elif label == 8:
        corners_labels[index] = 22
    elif label == 9:
        corners_labels[index] = 17
    elif label == 10:
        corners_labels[index] = 18
    elif label == 11:
        corners_labels[index] = 19
    elif label == 12:
        corners_labels[index] = 20
    elif label == 33:
        corners_labels[index] = 33
    elif label == 34:
        corners_labels[index] = 36
    elif label == 35:
        corners_labels[index] = 37

    return corners_labels



def find_point_34(corners):
        try:
            line1 = (corners['7'],corners['8'])
            line2 = (corners['2'],corners['9'])
            goal_point = find_intersection_point(line1,line2)
            return goal_point
        except:
            return None
    
def find_point_35(corners):
    try:
        line1 = (corners['7'],corners['8'])
        line2 = (corners['5'],corners['12'])
        goal_point = find_intersection_point(line1,line2)
        return goal_point
    except:
        return None

    
def find_point_32(corners):
    try:
        line1 = (corners['8'],corners['10'])
        line2 = (corners['7'],corners['11'])
        goal_point = find_intersection_point(line1,line2)
        return goal_point
    except:
        return None
        
def corners_postprocess(corners: dict):
    """
    Postprocess the corners to get the correct order of them
    """

    # point_32 = find_point_32(corners)
    point_34 = find_point_34(corners)
    point_35 = find_point_35(corners)

    # if point_32:
    #     corners.update({'32':point_32})
    if point_34:
        corners.update({'34':point_34})
    if point_35:
        corners.update({'35':point_35})

    corners_cords = list(corners.values())
    corners_labels = list(corners.keys())
    postprocessed_corners = {}

    mid_corners = {}
    for label in corners:
        if label != "c":
            label = int(label)

            if label in [13, 14, 15, 16]:
                mid_corners.update({label: tuple(map(int, corners[str(label)]))})

    if len(mid_corners) > 0:
        min_x = min(mid_corners.values(), key=lambda x: x[0])
        max_x = max(mid_corners.values(), key=lambda x: x[0])

        for i in range(len(corners_labels)):
            if corners_labels[i] != "c":
                label = int(corners_labels[i])
                if corners_cords[i][0] < min_x[0] or corners_cords[i][0] < max_x[0]:
                    corners_labels = invert_corners(label, corners_labels, i)

        c_corners = [
            corners_cords[i]
            for i in range(len(corners_labels))
            if corners_labels[i] == "c"
        ]

        if c_corners:
            c_corners = c_corners[0]

        c_corners.sort(key=lambda x: x[0])
        if c_corners:
            if len(c_corners) > 1:
                postprocessed_corners.update({"29": c_corners[0]})
                postprocessed_corners.update({"30": c_corners[-1]})

            elif len(c_corners) == 1:
                if c_corners[0][0] < max_x[0]:
                    postprocessed_corners.update({"29": c_corners[0]})
                else:
                    postprocessed_corners.update({"30": c_corners[0]})

    else:
        main_horizontal_lines = cluster_main_horizontal_lines(corners)
        for line in main_horizontal_lines:
            if len(line) > 1:
                try:
                    slope = calculate_slope(line)
                    if slope < 0:
                        for i in range(len(corners_labels)):
                            if corners_labels[i] != "c":
                                label = int(corners_labels[i])
                                corners_labels = invert_corners(
                                    label, corners_labels, i
                                )
                except:
                    pass
                break

        c_corners = [
            corners_cords[i]
            for i in range(len(corners_labels))
            if corners_labels[i] == "c"
        ]
        try:
            max_x = max(
                [
                    corners_cords[i]
                    for i in range(len(corners_labels))
                    if corners_labels[i] != "c"
                ],
                key=lambda x: x[0],
            )

            if c_corners:
                c_corners = c_corners[0]

            c_corners.sort(key=lambda x: x[0])
            if c_corners:
                if len(c_corners) > 1:
                    postprocessed_corners.update({"29": c_corners[0]})
                    postprocessed_corners.update({"30": c_corners[-1]})

                elif len(c_corners) == 1:
                    if c_corners[0][0] < max_x[0]:
                        postprocessed_corners.update({"30": c_corners[0]})
                    else:
                        postprocessed_corners.update({"29": c_corners[0]})
        except:
            pass

    for i in range(len(corners_labels)):
        if (
            not corners_labels[i] in postprocessed_corners.keys()
            and corners_labels[i] != "c"
        ):
            postprocessed_corners.update({str(corners_labels[i]): corners_cords[i]})

    return postprocessed_corners
