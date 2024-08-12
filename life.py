import cv2
import numpy as np
import random as r
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from enum import Enum


class Color(Enum):
    #BGR
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    PURPLE = (255, 0, 255)
    YELLOW = (3, 236, 252)


WIDTH = 600.0
HEIGHT = 600.0
RADIUS = 3.0
AMOUNT = 100.0
HORIZ_SAMPLE_SIZE = WIDTH/AMOUNT
VERT_SAMPLE_SIZE = WIDTH/AMOUNT
MAX_FORCE_DIST = RADIUS * 10.0

MAX_V = 3.0
FRICTION = 0.05

ForceMatrix = {}

for p in Color._member_names_:
    ForceMatrix[p] = {}
    for p2 in Color._member_names_:
        ForceMatrix[p][p2] = r.random()*(-1 if r.random()>0.5 else 1)

print(ForceMatrix)

# print(HORIZ_SAMPLE_SIZE)
# print(VERT_SAMPLE_SIZE)

FACTOR = 2

NUM_PARTICLES = 200

empty = []

for i in range(0, int(AMOUNT*AMOUNT)):
    empty.append(0)

# print(len(empty))

colorMap = mpl.colormaps['plasma']
colorSpace = []
for i, color in enumerate(colorMap(np.linspace(0, 1, int(math.pow(10, FACTOR))))):
    colorSpace.append(color)

testIdx = -1
# print((colorSpace[testIdx][0]*255, colorSpace[testIdx][1]*255, colorSpace[testIdx][2]*255))
# plt.figure(facecolor= colorSpace[testIdx])
# plt.scatter(
#     x = (0, 2),
#     y = (0, 2) ,
#     c = (0, 1),
#     cmap='plasma'
# )
# plt.colorbar()
distL = []
forceL = []

for dist in range(0, int(MAX_FORCE_DIST+RADIUS)):
    if dist < RADIUS:
        distL.append(dist)
        forceL.append(-0.5 * (RADIUS-dist)/RADIUS)
    # if not close force calc off matrix up until fall off distance
    elif dist < MAX_FORCE_DIST+RADIUS:
        distL.append(dist)
        forceL.append(ForceMatrix[Color.RED.name][Color.RED.name] * (1-abs((MAX_FORCE_DIST/2.0)-(dist-RADIUS))/(MAX_FORCE_DIST/2.0)))
plt.figure()
plt.plot(distL, forceL)
plt.show()



class Particle:
    x, y, dx, dy, mass, color = None, None, None, None, None, None
    def __init__(self, mass=1.0, x=0.0, y=0.0, dx=1.0, dy=1.0, color=Color.RED):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.mass = mass
        self.color = color

    def update(self, ddx=0.0, ddy=0.0):
        self.x += self.dx
        self.y += self.dy
        self.dx += ddx - math.copysign(self.dx * FRICTION, self.dx)
        self.dy += ddy - math.copysign(self.dy * FRICTION, self.dy)

        self.x = self.x%WIDTH
        self.y = self.y%HEIGHT

    def getColor(self):
        return self.color.value

class MathUtil:
    sign = lambda x: math.copysign(1, x)


    @staticmethod
    def calcCollisionVelocity(m1=0.0, dx1=0.0, dy1=0.0, m2=0.0, dx2=0.0, dy2=0.0):
        dx = (m1*dx1+ m2*dx2 - m2*(dx1-dx2))/(m1+m2)
        dy = (m1*dy1+ m2*dy2 - m2*(dy1-dy2))/(m1+m2)
        return (dx, dy, dx+dx1-dx2, dy+dy1-dy2)
    
    @staticmethod
    def calcDist(x1=0.0, y1=0.0, x2=0.0, y2=0.0):
        return math.sqrt(math.pow(x2-x1,2)+math.pow(y2-y1,2))

def tryValue(count, index=0):
    if(index<0):
        return -1
    try:
        return count[int(index)]
    except Exception as error:
        # print(f'error with index: {index} -- {error} -- {count}')
        # print(count)
        return -1

cv2.namedWindow("life", cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("life", int(WIDTH), int(HEIGHT))

particles = [
    Particle(
        x=20.0,
        y=10.0
    )
]

# print(Color[Color._member_names_[r.randint(0, 2)]])

for i in range(0, NUM_PARTICLES-1):
    particles.append(
        Particle(x=r.random()*WIDTH, y=r.random()*HEIGHT, dx=r.random()*MAX_V, dy=r.random()*MAX_V, color=Color[Color._member_names_[r.randint(0, len(Color._member_names_)-1)]])
    )


while True:
    window = cv2.getWindowImageRect("life")
    # print(window)
    if window[2] != WIDTH or window[3] != HEIGHT:
        WIDTH = window[2]
        HEIGHT = WIDTH
        # cv2.resizeWindow("life", WIDTH, WIDTH)

        HORIZ_SAMPLE_SIZE = WIDTH/AMOUNT
        VERT_SAMPLE_SIZE = HEIGHT/AMOUNT

        # print(f'width {WIDTH} height {HEIGHT}')
        cv2.resizeWindow("life", WIDTH, HEIGHT)
        # print((cv2.getWindowImageRect("life")[2], cv2.getWindowImageRect("life")[3]))

        empty = []

        for i in range(0, int(AMOUNT*AMOUNT)):
            empty.append(0)
    #create an empty image that represents a 5m by 5m box
    img = np.zeros((int(WIDTH), int(HEIGHT)), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    count = empty.copy()
    for i in range(0, len(particles)):
        p = particles[i]
        cv2.circle(img, (int(p.x), int(p.y)), int(RADIUS), p.getColor(), -1)
        # print(f'x box {p.x//HORIZ_SAMPLE_SIZE} y box {AMOUNT*(max((0,(p.y//VERT_SAMPLE_SIZE)-1)))}')
        box = min((len(count)-1, int(p.x//HORIZ_SAMPLE_SIZE) + int(AMOUNT*(min((AMOUNT-1, max((0,(p.y//VERT_SAMPLE_SIZE)))))))))
        # print(f'box: {box} len: {len(count)}')
        count[box] += 1

        ddx = 0.0
        ddy = 0.0

        for n in range(0, len(particles)):
            if(n!=i):
                p2 = particles[n]
                # calc distance for force calc
                dist = MathUtil.calcDist(p.x, p.y, p2.x, p2.y)
                # determine compentents of accels
                theta = math.atan2(p2.y-p.y, p2.x-p.x)
                dds = (math.cos(theta), math.sin(theta))

                # if close always repel
                if dist < RADIUS:
                    ddx += -0.5 * (RADIUS-dist)/RADIUS * dds[0]
                    ddy += -0.5 * (RADIUS-dist)/RADIUS * dds[1]
                # if not close force calc off matrix up until fall off distance
                elif dist < MAX_FORCE_DIST+RADIUS:
                    ddx += ForceMatrix[p.color.name][p2.color.name] * (1-abs((MAX_FORCE_DIST/2.0)-(dist-RADIUS))/(MAX_FORCE_DIST/2.0)) * dds[0]
                    ddy += ForceMatrix[p.color.name][p2.color.name] * (1-abs((MAX_FORCE_DIST/2.0)-(dist-RADIUS))/(MAX_FORCE_DIST/2.0)) * dds[1]

        
        p.update(ddx=ddx, ddy=ddy)

    for i in range(0, len(count)):
        nums = (
            tryValue(count, i-1-int(AMOUNT)), tryValue(count, i-int(AMOUNT)), tryValue(count, i+1-int(AMOUNT)),
            tryValue(count, i-1), tryValue(count, i), tryValue(count, i+1),
            tryValue(count, i-1+int(AMOUNT)),tryValue(count, i+int(AMOUNT)), tryValue(count, i+1+int(AMOUNT))
        )

        total = 0.0
        sum = 0.0

        for num in nums:
            if num != -1:
                sum += num
                total += 1.0

        value = sum/total
        color = colorSpace[int(value)]
        p1 = (int((i%AMOUNT)*HORIZ_SAMPLE_SIZE), int((i//AMOUNT)*VERT_SAMPLE_SIZE))
        p2 = (int(((i%AMOUNT)+1)*HORIZ_SAMPLE_SIZE), int(((i//AMOUNT)+1)*VERT_SAMPLE_SIZE))

        # cv2.rectangle(img, p1, p2, (color[2]*255, color[1]*255, color[0]*255), -1) # BGR
    

    cv2.imshow("life", img)

    key = cv2.waitKey(1)
    if(key == ord('q')):
        break

cv2.destroyAllWindows()
