import torch 
import numpy as np



def gen_point_proj_operator_batched(sigma,size, device = "cpu"):
    class point_projection(torch.autograd.Function):


        @staticmethod
        def forward(ctx, inputs):
            points_pos = inputs

            x = torch.arange(size).to(device)
         
            grid_x, grid_y = torch.meshgrid(x, x, indexing='ij')

         
   
            c1 = points_pos[:,:,0].reshape((len(points_pos),100,1,1))
            c2 = points_pos[:,:,1].reshape((len(points_pos),100,1,1))
            a = torch.sum(torch.exp(-((grid_x-c1)**2/(2*sigma**2)+(grid_y-c2)**2/(2*sigma**2))),dim=1)

            ctx.save_for_backward(points_pos)
            ctx.sigma = sigma


            return torch.flatten(a, start_dim=1)

        @staticmethod
        def backward(ctx, grad_output):


            inputs, = ctx.saved_tensors

            points_pos = inputs

            g =  torch.zeros((len(points_pos),2,100,size,size)).float().to(device)



            
            x = torch.arange(size).to(device)
            y = torch.arange(size).to(device)
  

            grid_x, grid_y =torch.meshgrid(x, y, indexing='ij')
        
            c1 = points_pos[:,:,0].reshape((len(points_pos),100,1,1))
            c2 = points_pos[:,:,1].reshape((len(points_pos),100,1,1))

            g[:,0] =  (-c1 + grid_x) *torch.exp(-((grid_x-c1)**2/(2*sigma**2)+(grid_y-c2)**2/(2*sigma**2)))
            g[:,1] = (-c2 + grid_y) *torch.exp(-((grid_x-c1)**2/(2*sigma**2)+(grid_y-c2)**2/(2*sigma**2)))
           
            ret = torch.einsum('baij,bj->bai', torch.reshape(g, (len(points_pos),2,100,-1)), grad_output).permute(0,2,1)
            
            return ret.clone()
    return  point_projection.apply

