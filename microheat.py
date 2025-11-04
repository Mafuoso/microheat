import numpy as np
import math



class Particle():

    def __init__(self,x,y):
        self.x: float 
        self.y: float 
        self.vx: float
        self.vy: float
        self.r: float = 1.0 #particle radius 
        self.m: float = 1.0 #particle mass
    
class Box():
    
    def __init__(self,width,height):
        self.width: float
        self.height: float
        self.gravity_strength: float = 9.8 
        
    def make_grid(self,N:int):
        x_array = np.arrange(self.width)
        y_array = np.arrange(self.height)
        X,Y = np.meshgrid(x_array,y_array)
        #need to change so that it can accomodate N, and we have even spacing at init
        return X,Y 
    
def initialize(N:int,width:float,height:float):
    particle_dict = {}
    box = Box(width,height)
    X,Y = box.make_grid()
    for i in range(N):
        particle_dict["particle{0}".format(i)] = Particle()