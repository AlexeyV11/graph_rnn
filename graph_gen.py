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
        x, y = zip(*points)
        plt.plot(x, y)
        plt.waitforbuttonpress()

    @abstractmethod
    def generate(self, x_range=[- math.pi * 2, math.pi * 2], step=0.001):
        pass


class SinGraph(Graph):
    def generate(self, x_range=[- math.pi * 2, math.pi * 2], step=0.001):
        return [(x, math.sin(x)) for x in frange(x_range[0], x_range[1] + step, step)]

class SinGraphRandom(Graph):
    def __init__(self):
        self.a = random.random() * 10
        self.b = random.random() * 10

        self.shift = random.random() * 2 * math.pi

    def generate(self, x_range=[- math.pi * 2, math.pi * 2], step=0.001):
        return [(x, self.a * math.sin(self.b * (x + self.shift))) for x in frange(x_range[0], x_range[1] + step, step)]


if __name__  == "__main__":
    graph = SinGraph()
    graph.plot()

    for x in range(10):
        graph = SinGraphRandom()
        graph.plot()





