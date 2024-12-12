import numpy as np 
import torch

G = 6.67430e-11
M = 5.972e24
EARTH_RADIUS = 6378e3 #m
GEO_ORBIT_RADIUS = 42164e3 #m
g = 9.81
ROTATION_SPEED = 7.292115e-5 #deg.s-1
c = 2.99792458e8
epsilon = 1e-40


class orbite_eliptique(): 
    def __init__(self, apogee, exentricity, inclination, argperigee, raan, device ="cpu" ):
        
        self.apogee = apogee.to(device)
        self.exentricity = exentricity.to(device)
        self.inclination = inclination.to(device)
        self.argperigee = argperigee.to(device)
        self.raan = raan.to(device)
        self.device = device
        
    def getPos(self ,t ):
        self.b = self.apogee*torch.sqrt((1-(self.exentricity**2)))
        self.T = torch.sqrt(4*torch.pi**2/(G*M)*self.apogee**3)


        # Calcul de E_10
        M_ = 2*torch.pi/self.T * t
        E_n =  M_
        for i in range(10):
            E_n = E_n-((E_n-self.exentricity*torch.sin(E_n)-M_)/(1-self.exentricity*torch.cos(E_n)))
   
        
        
        m1 = torch.zeros(3,3).to(self.device)
        m1.requires_grad = False
        m1[0,0] = torch.cos(self.argperigee)
        m1[0,1] = -torch.sin(self.argperigee)
        m1[1,0] = torch.sin(self.argperigee)
        m1[1,1] = torch.cos(self.argperigee)
        m1[2,2] = 1
        
        m3 = torch.zeros(3,3).to(self.device)
        m3.requires_grad = False
        m3[0,0] = torch.cos(self.raan)
        m3[0,1] = -torch.sin(self.raan)
        m3[1,0] = torch.sin(self.raan)
        m3[1,1] = torch.cos(self.raan)
        m3[2,2] = 1
        
        
        m2 = torch.zeros(3,3).to(self.device)
        m2.requires_grad = False
        m2[0,0] = 1
        m2[1,1] = torch.cos(self.inclination)
        m2[1,2] = -torch.sin(self.inclination)
        m2[2,1] = torch.sin(self.inclination)
        m2[2,2] = torch.cos(self.inclination)
        
        
        # vec = torch.tensor([0. , 0., 0.]).to(self.device)
        vec = torch.zeros((len(t),3)).to(self.device)
        vec.requires_grad = False 
        vec[:,0] = -torch.sqrt(self.apogee**2-self.b**2) + self.apogee*torch.cos(E_n)
        vec[:,1] = self.b*torch.sin(E_n)
     
        return (m3@m2@m1@(vec.T)).T 
    
    def getPosTensor(self, size  = 100):
        
        d = torch.zeros((size,3))
        d.requires_grad = False
        self.T = torch.sqrt(4*torch.pi**2/(G*M)*self.apogee**3)
        # t = torch.linspace(0,self.T,size).to(self.device)
        t = torch.linspace(0,self.T,size)  
        # for i in range(size):
        #     d[i] = self.getPos(t[i])          
        return self.getPos(t)  
    
    
    def getPosTensorGeocentred(self, size  = 100):


        self.T = torch.sqrt(4*torch.pi**2/(G*M)*self.apogee**3)
        # t = torch.linspace(0,self.T,size).to(self.device)
        t = torch.linspace(0,2*self.T,size).to(self.device)
        # for i in range(size):
        #     d[i] = self.getPos(t[i])   
        mat = torch.zeros((size,3,3)).to(self.device)
        mat[:,2,2] = 1
        mat[:,0,0] = torch.cos(-t*ROTATION_SPEED) 
        mat[:,0,1] = -torch.sin(-t*ROTATION_SPEED)
        mat[:,1,0] = torch.sin(-t*ROTATION_SPEED)
        mat[:,1,1] = torch.cos(-t*ROTATION_SPEED)
        return (mat@(self.getPos(t)[...,None]))[:,:,0]
    
    

    
class orbite_eliptique_batched(): 
    def __init__(self, apogee, exentricity, inclination, argperigee, raan, device ="cpu" ):
        
        self.apogee = apogee.to(device)
        self.exentricity = exentricity.to(device)
        self.inclination = inclination.to(device)
        self.argperigee = argperigee.to(device)
        self.raan = raan.to(device)
        self.device = device
        self.nbr_orb = len(apogee)
     
    def getPos(self ,t ):
        self.b = self.apogee*torch.sqrt((1-(self.exentricity**2)))
        self.T = torch.sqrt(4*torch.pi**2/(G*M)*self.apogee**3)
     

        # Calcul de E_10
        M_ = 2*torch.pi/self.T * t.T
 
        E_n =  M_
        for i in range(10):
            E_n = E_n-((E_n-self.exentricity*torch.sin(E_n)-M_)/(1-self.exentricity*torch.cos(E_n)))
        E_n = E_n.T
      
        
        
        m1 = torch.zeros(len(self.argperigee),3,3).to(self.device)
        m1.requires_grad = False
        m1[:,0,0] = torch.cos(self.argperigee)
        m1[:,0,1] = -torch.sin(self.argperigee)
        m1[:,1,0] = torch.sin(self.argperigee)
        m1[:,1,1] = torch.cos(self.argperigee)
        m1[:,2,2] = 1
        
        m3 = torch.zeros(len(self.raan),3,3).to(self.device)
        m3.requires_grad = False
        m3[:,0,0] = torch.cos(self.raan)
        m3[:,0,1] = -torch.sin(self.raan)
        m3[:,1,0] = torch.sin(self.raan)
        m3[:,1,1] = torch.cos(self.raan)
        m3[:,2,2] = 1
        
        
        m2 = torch.zeros(len(self.inclination),3,3).to(self.device)
        m2.requires_grad = False
        m2[:,0,0] = 1
        m2[:,1,1] = torch.cos(self.inclination)
        m2[:,1,2] = -torch.sin(self.inclination)
        m2[:,2,1] = torch.sin(self.inclination)
        m2[:,2,2] = torch.cos(self.inclination)
        
        vec = torch.zeros((len(self.inclination),t.shape[1],3)).to(self.device)
        vec.requires_grad = False 
       
        vec[:,:,0] = (-torch.sqrt(self.apogee**2-self.b**2+epsilon) + self.apogee*torch.cos(E_n).T).T
        vec[:,:,1] = (self.b*torch.sin(E_n).T).T
 
        return torch.bmm(torch.bmm(torch.bmm(m3,m2),m1),vec.permute(0, 2, 1)).permute(0, 2, 1)
        # return torch.bmm(torch.bmm(m3,m1),vec.permute(0, 2, 1)).permute(0, 2, 1)
      
    def getPosTensor(self, size  = 100):
        
        d = torch.zeros((size,3))
        d.requires_grad = False
        self.T = torch.sqrt(4*torch.pi**2/(G*M)*self.apogee**3)
        # t = torch.linspace(0,self.T,size).to(self.device)
        t = torch.linspace(0,self.T,size)  
        # for i in range(size):
        #     d[i] = self.getPos(t[i])          
        return self.getPos(t)  
    
    
    def getPosTensorGeocentred(self, size  = 100):


        self.T = torch.sqrt(4*torch.pi**2/(G*M)*self.apogee**3)
      
      
        t = (self.T *torch.linspace(0,2,size).to(self.device).repeat(self.nbr_orb,1).T).T
   
        mat = torch.zeros((self.nbr_orb,size,3,3)).to(self.device)
        mat[:,:,2,2] = 1
        mat[:,:,0,0] =  torch.cos(-t*ROTATION_SPEED)
        mat[:,:,0,1] = -torch.sin(-t*ROTATION_SPEED)
        mat[:,:,1,0] =  torch.sin(-t*ROTATION_SPEED)
        mat[:,:,1,1] =  torch.cos(-t*ROTATION_SPEED)
        return (mat@(self.getPos(t)[...,None]))[...,0]
    
