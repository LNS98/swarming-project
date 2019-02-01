from shapely.geometry import LinearRing, Polygon, Point
import matplotlib.pyplot as plt


poly = Polygon([(0, 0), (1,0), (1, 1), (0, 1)])
point_coord = (1, 1)
point = Point(point_coord)

pol_ext = LinearRing(poly.exterior.coords)
d = pol_ext.project(point)
p = pol_ext.interpolate(d)
closest_point_coords = list(p.coords)[0]


print(d)
print(p)


# plot the shape
x,y = pol_ext.coords.xy

plt.plot(x, y)
plt.scatter(point_coord[0], point_coord[1])
plt.show()
