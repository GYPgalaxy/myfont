import math


class Point(object):
    def __init__(self, xParam=0.0, yParam=0.0):
        self.x = xParam
        self.y = yParam

    def __str__(self):
        return "%.2f %.2f" % (self.y, self.x)

    def distance(self, pt):
        xDiff = self.x - pt.x
        yDiff = self.y - pt.y
        return math.sqrt(xDiff ** 2 + yDiff ** 2)

    def sum(self, pt):
        # newPt = Point()
        xNew = self.x + pt.x
        yNew = self.y + pt.y
        return Point(xNew, yNew)
