import cv2
import numpy as np
import random as r
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import threading


WIDTH = 400.0
HEIGHT = 400.0
RADIUS = 1.0
AMOUNT = 100.0
HORIZ_SAMPLE_SIZE = WIDTH/AMOUNT
VERT_SAMPLE_SIZE = WIDTH/AMOUNT

MAX_V = 3.0

# print(HORIZ_SAMPLE_SIZE)
# print(VERT_SAMPLE_SIZE)

FACTOR = 2

NUM_PARTICLES = 500

empty = []

for i in range(0, int(AMOUNT*AMOUNT)):
    empty.append(0)

# print(len(empty))

colorMap = mpl.colormaps['plasma']
colorSpace = []
for i, color in enumerate(colorMap(np.linspace(0, 1, int(math.pow(10, FACTOR))))):
    colorSpace.append(color)

testIdx = -1
print((colorSpace[testIdx][0]*255, colorSpace[testIdx][1]*255, colorSpace[testIdx][2]*255))
plt.figure(facecolor= colorSpace[testIdx])
plt.scatter(
    x = (0, 2),
    y = (0, 2) ,
    c = (0, 1),
    cmap='plasma'
)
plt.colorbar()
plt.show()


class Particle:
    x, y, dx, dy, mass = None, None, None, None, None
    def __init__(self, mass=1.0, x=0.0, y=0.0, dx=1.0, dy=1.0):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.mass = mass

    def update(self, ddx=0.0, ddy=0.0):
        self.x += self.dx
        self.y += self.dy
        self.dx += ddx
        self.dy += ddy

        if(self.x+RADIUS>=WIDTH or self.x-RADIUS<=0.0):
            self.dx *= -1

        if(self.y+RADIUS>=HEIGHT or self.y-RADIUS<=0.0):
            self.dy *= -1

class ParticalUpdater(threading.Thread):
    def __init__(self, particles, count, name='particle-thread'):
        self.particles = particles
        self.count = count
        self.atIndex = 0
        self.norm = 0
        self.ready = False
        self.go = True
        super(ParticalUpdater, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            if not self.go:
                print("waiting to go")
            else:
                self.ready = False
                self.count = empty.copy()
                for i in range(0, len(particles)):
                    p = particles[i]
                    # cv2.circle(img, (int(p.x), int(p.y)), int(RADIUS), (255, 255, 255), -1)
                    # print(f'x box {p.x//HORIZ_SAMPLE_SIZE} y box {AMOUNT*(max((0,(p.y//VERT_SAMPLE_SIZE)-1)))}')
                    box = min((len(self.count)-1, int(p.x//HORIZ_SAMPLE_SIZE) + int(AMOUNT*(min((AMOUNT-1, max((0,(p.y//VERT_SAMPLE_SIZE)))))))))
                    # print(f'box: {box} len: {len(self.count)}')
                    self.count[box] += 1

                    for n in range(0, len(particles)):
                        if(n!=i):
                            p2 = particles[n]
                            if(MathUtil.calcDist(p.x, p.y, p2.x, p2.y) <= RADIUS*2.0):
                                
                                dx1, dy1, dx2, dy2 = MathUtil.calcCollisionVelocity(p.mass, p.dx, p.dy, p2.mass, p2.dx, p2.dy)
                                p.dx = dx1
                                p.dy = dy1
                                p2.dx = dx2
                                p2.dy = dy2
                self.ready = True
                self.go = False

class MathUtil:
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

cv2.namedWindow("fluid", cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("fluid", int(WIDTH), int(HEIGHT))

particles = [
    Particle(
        x=20.0,
        y=10.0
    )
]

for i in range(0, NUM_PARTICLES-1):
    particles.append(
        Particle(x=r.random()*WIDTH, y=r.random()*HEIGHT, dx=r.random()*MAX_V, dy=r.random()*MAX_V)
    )

thread = ParticalUpdater(particles=particles, count=empty.copy())


while True:
    window = cv2.getWindowImageRect("fluid")
    # print(window)
    if window[2] != WIDTH or window[3] != HEIGHT:
        WIDTH = window[2]
        HEIGHT = WIDTH
        # cv2.resizeWindow("fluid", WIDTH, WIDTH)

        HORIZ_SAMPLE_SIZE = WIDTH/AMOUNT
        VERT_SAMPLE_SIZE = HEIGHT/AMOUNT

        # print(f'width {WIDTH} height {HEIGHT}')
        cv2.resizeWindow("fluid", WIDTH, HEIGHT)
        # print((cv2.getWindowImageRect("fluid")[2], cv2.getWindowImageRect("fluid")[3]))

        empty = []

        for i in range(0, int(AMOUNT*AMOUNT)):
            empty.append(0)
    #create an empty image that represents a 5m by 5m box
    img = np.zeros((int(WIDTH), int(HEIGHT)), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if not thread.ready:
        print("waiting")
    else:
        count = thread.count.copy()
        thread.go = True

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

            cv2.rectangle(img, p1, p2, (color[2]*255, color[1]*255, color[0]*255), -1) # BGR
        

        cv2.imshow("fluid", img)

        key = cv2.waitKey(1)
        if(key == ord('q')):
            break

cv2.destroyAllWindows()
