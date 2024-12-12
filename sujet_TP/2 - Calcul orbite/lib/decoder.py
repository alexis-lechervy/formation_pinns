import torch 
import numpy as np 



import lib.utils as utils
import lib.orbite  as orbite
import lib.station as station
from lib.translate import  gen_point_proj_operator_batched



EARTH_RADIUS = 6378e3 #m
GEO_ORBIT_RADIUS = 42164e3 #m
g = 9.81
ROTATION_SPEED = 7.292115e-5 #deg.s-1

def decoder_single_img(orb, perspective, view, img_size, device ):
    n = orb.shape[1]
    orb = orbite.orbite_eliptique_batched( apogee = torch.ones(n*len(orb))*GEO_ORBIT_RADIUS, exentricity = orb[:,:,0].flatten(), inclination = orb[:,:,1].flatten(), argperigee = orb[:,:,2].flatten(), raan = torch.zeros(n*len(orb)), device = device)
    pos = orb.getPosTensorGeocentred()

    pos = torch.cat((pos, torch.ones((pos.shape[:2]), device = torch.device(device))[...,None]), dim=-1)
   
    projection = (perspective@view@(pos[...,None]))[...,0]

    rtn = torch.stack((projection[:,:,0]/ projection[:,:,3],projection[:,:,1]/ projection[:,:,3]) , dim = -1)

    d = gen_point_proj_operator_batched(1, img_size ,device)(img_size//2*rtn+img_size//2)
    img = torch.sum(d.reshape(-1,n,img_size**2), dim = 1)
    return torch.minimum(img, 100*torch.ones_like(img))/100






def decoder(img_size, device = "cpu"):
    """ Initialise the physical model with a fixed image output size 
      Parameters:
       img_size (int): Size of the produced squared images
       device (string): Device on which the computation is done ("cpu" for cpu and "cuda:0" for gpu 0)
      Returns:
       func : Return a function corresponding to the physical model 

    """

    # Sensor parameters -> MVP projection model 
    station_ = station.Station([0,0])
    perspective = torch.from_numpy(utils.getPerspectiveMat(np.deg2rad(120), 1.0, 0.1, 100)).float().to(device)
    view = torch.from_numpy(utils.getViewMat(station_.pos_car,2*station_.pos_car, np.array([0,1,0]))).float().to(device)


    def decoder_func(orb_list):

      """ Simulate an image from a given orbit defined by his keplerian elements
      Parameters:
       orb_list (torch.Tensor): Tensor of size (B,N,3)
             B : batch size
             N : number of orbit per image
             
             For each orbit, we have 3 argument. Respectively the exentricity, the inclination and the argument of periapsis.
      Returns:
       torch.Tensor : Return the assiciated images.

      """


      return decoder_single_img(orb_list,perspective, view, img_size,device)


    return decoder_func 
