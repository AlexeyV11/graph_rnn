import random
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


import math

def frange(start, stop, step):
    x = start
    while x < stop:
        yield x
        x += step

class Graph(ABC):

    def plot(self, **kwargs):
        points = self.generate(**kwargs)
        plt.plot(points)
        plt.waitforbuttonpress()

    @abstractmethod
    def generate(self, x_range, step):
        pass


class SinGraph(Graph):
    def generate(self, x_range=[- math.pi * 2 * 8, math.pi * 2 * 8], step=0.1):
        return [math.sin(x) for x in frange(x_range[0], x_range[1] + step, step)]

class SinGraphRandom(Graph):
    def __init__(self):
        self.a = max(1, random.random() * 5)
        self.b = max(random.random(), 0.2) * 2

        self.shift = random.random() * 2 * math.pi

    def generate(self, x_range=[- math.pi * 2 * 4, math.pi * 2 * 4], step=0.05):
        return [self.a * math.sin(self.b * (x + self.shift)) for x in frange(x_range[0], x_range[1] + step, step)]


if __name__  == "__main__":
    graph = SinGraph()
    graph.plot()

    for x in range(10):
        graph = SinGraphRandom()
        graph.plot()





