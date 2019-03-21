import numpy as np
import matplotlib.pyplot as plt

import pymunk

L = 10

def main():

    poly = polygon()
    cn = centroid(poly)
    ar = area(poly)
    ine = inertia(poly)

    moi = pymunk.moment_for_poly(100, poly)
    moi_real = (100 * 4)/6

    print(poly)
    print(moi)
    print(moi_real)
    return 0

def polygon():
    """
    Define the polygon from the points on the verticies.
    """
    # regular polygon for testing
    # lenpoly = 5
    # polygon = np.array([[random.random() + L/2, random.random() + L/2] for x in range(4)])

    polygon =[[L/2 - 1, L/2 - 1], [L/2 + 1, L/2 - 1], [L/2 + 1, L/2 + 1], [L/2 - 1, L/2 + 1]]

    return polygon

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

def inertia(pts):
    'Moments and product of inertia about centroid.'

    if pts[0] != pts[-1]:
      pts = pts + pts[:1]

    x = [c[0] for c in pts]
    y = [c[1] for c in pts]

    sxx = syy = sxy = 0
    a = area(pts)
    cx, cy = centroid(pts)

    for i in range(len(pts) - 1):
      sxx += (y[i]**2 + y[i]*y[i+1] + y[i+1]**2)*(x[i]*y[i+1] - x[i+1]*y[i])
      syy += (x[i]**2 + x[i]*x[i+1] + x[i+1]**2)*(x[i]*y[i+1] - x[i+1]*y[i])
      sxy += (x[i]*y[i+1] + 2*x[i]*y[i] + 2*x[i+1]*y[i+1] + x[i+1]*y[i])*(x[i]*y[i+1] - x[i+1]*y[i])

    return [sxx/12 - a*cy**2, syy/12 - a*cx**2]


main()
