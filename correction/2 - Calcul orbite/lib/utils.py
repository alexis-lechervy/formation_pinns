import numpy as np

def getPerspectiveMat(fov, aspect, near, far):
    return np.array([[1/(aspect*np.tan(fov/2)),0,0,0],[0,1/np.tan(fov/2),0,0],[0,0, -(far+near)/(far-near),-(2*far*near)/(far-near)],[0,0,-1,0]])

# See https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluLookAt.xml
def getViewMat(eye, center, up):
    f  = center - eye 

    f = f*1/np.linalg.norm(f)
    up_ = up*1/np.linalg.norm(up)

    s = np.cross(f , up_)
    s *= 1/np.linalg.norm(s)
    
    u = np.cross(s,f)
    u *=  1/np.linalg.norm(u)
    
    
    M = np.array([[s[0], s[1], s[2],-np.dot(s, eye)], [u[0], u[1], u[2], -np.dot(u, eye)], [-f[0], -f[1], -f[2], -np.dot(f, eye)],[0,0,0,1]])
    # trans = np.eye(4)
    # trans[:3,3] = -eye
    return M


# def getModelMat()