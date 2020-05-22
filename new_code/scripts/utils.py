import numpy as np

# constants 
L = 5 # size of the box


def distance_fun(pos1, pos2):
    """
    Calculate the distance between the points
    """
    # get the two arrays as np arrays, easier to do calculations
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)

    # get the distance
    distance = pos2 - pos1

    # distance is the same as the magnitude
    dist = np.sqrt(distance.dot(distance))

    return dist

def centroid(pts):
    'Location of centroid.'

    # check if the last point is the same as the first, if nots so 'close' the polygon
    if pts[0] != pts[-1]:
        pts = pts + pts[:1]

    # get the x and y points
    x = [c[0] for c in pts]
    y = [c[1] for c in pts]

    # initialise the x and y centroid to 0 and get the area of the polygon
    sx = sy = 0
    a = area(pts)

    for i in range(len(pts) - 1):
        sx += (x[i] + x[i+1])*(x[i]*y[i+1] - x[i+1]*y[i])
        sy += (y[i] + y[i+1])*(x[i]*y[i+1] - x[i+1]*y[i])

    return [sx/(6*a), sy/(6*a)]

def area(pts):
    'Area of cross-section.'

    if pts[0] != pts[-1]:
      pts = pts + pts[:1]

    x = [ c[0] for c in pts ]
    y = [ c[1] for c in pts ]
    s = 0

    for i in range(len(pts) - 1):
        s += x[i]*y[i+1] - x[i+1]*y[i]

    return s/2

def rescale(magnitude, vector):
    """
    Changes the length of a  given vector to that of the magnitude given.
    """
    # make the vector a numpy array
    vec = np.array(vector)

    # get the magnitude
    mag = np.sqrt(vec.dot(vec))

    # multiply to rescale and make it a list
    new_vec = (magnitude / mag) * vec
    new_vec = list(new_vec)

    return new_vec



def per_boun_distance(i, j):
    """
    Calculates the minimum distance  between two particles in a box with periodic
    boundries.
    """
    # calculate the minimum x distance
    in_distance_x = j[0] - i[0]
    out_distance_x = L - in_distance_x
    distance_x = min(in_distance_x, out_distance_x)


    # calculate the minimum y distance
    in_distance_y = j[1] - i[1]
    out_distance_y = L - in_distance_y
    distance_y = min(in_distance_y, out_distance_y)

    return distance_x, distance_y
