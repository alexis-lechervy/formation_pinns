import numpy as np


from math import cos, sin, radians

EARTH_RADIUS = 6378e3
class Station():
    """Class qui modélise une station de capture

        Parameters
        ----------
        param : ndarray double [2]
            tableau qui contient les positions de la station (en terme de longitude/latitude ).
            
    """
    def __init__(self, lat_lon):
        self.lat_lon = lat_lon
        self.pos_car = np.array([EARTH_RADIUS  * cos(radians(lat_lon[0])) * cos(radians(lat_lon[1])), EARTH_RADIUS * cos(radians(lat_lon[0])) * sin(radians(lat_lon[1])), EARTH_RADIUS  * sin(radians(lat_lon[0]))])
        return 
    def update(self,t):
        # self.pos_t  = self.getStationPos(t)
        return

    def getStationPos(self,t ):
        # return  np.array([[np.cos(ROTATION_SPEED*t),-np.sin(ROTATION_SPEED*t),0],[np.sin(ROTATION_SPEED*t),np.cos(ROTATION_SPEED*t),0],[0,0,1]])@self.pos_car
        return  self.pos_car

    def getVelocityVec(self,t):
        return np.array([0.,0.,0.])
        # return  np.array([[-ROTATION_SPEED * np.sin(ROTATION_SPEED*t),ROTATION_SPEED *np.cos(ROTATION_SPEED*t),0],[ROTATION_SPEED*np.cos(ROTATION_SPEED*t),-ROTATION_SPEED*np.sin(ROTATION_SPEED*t),0],[0,0,0]])@self.pos_car    
       
