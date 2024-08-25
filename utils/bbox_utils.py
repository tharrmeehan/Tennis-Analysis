def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


# Measure the distance between two points
def measure_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)


def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    min_distance = float("inf")
    closest_keypoint_index = keypoint_indices[0]
    for keypoint_index in keypoint_indices:
        keypoint = keypoints[keypoint_index * 2], keypoints[keypoint_index * 2 + 1]
        distance = abs(point[1] - keypoint[1])

        if distance < min_distance:
            min_distance = distance
            closest_keypoint_index = keypoint_index

    return closest_keypoint_index


def get_height_of_bbox(bbox):
    return bbox[3] - bbox[1]


def measure_xy_distance(p1, p2):
    return abs(p2[0] - p1[0]), abs(p2[1] - p1[1])


def get_center_position_bbox(bbox):
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
