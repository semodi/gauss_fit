import numpy as np
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

par_b_r = []
par_b_phi = []
par_lp_r = []
par_lp_phi = []
par_lp_theta = []
par_b_width = []
par_lp_width = []
orders = []

with open(dir_path + '/monomer_parameters.gauss', 'r') as par_file:
    orders.append([int(p) for p in par_file.readline().split(sep =',')][1])
    par_b_r += [float(p) for p in par_file.readline().split(sep=',')]
    orders.append([int(p) for p in par_file.readline().split(sep =',')][1])
    par_b_phi += [float(p) for p in par_file.readline().split(sep=',')]
    orders.append([int(p) for p in par_file.readline().split(sep =',')][1])
    par_lp_r += [float(p) for p in par_file.readline().split(sep=',')]
    orders.append([int(p) for p in par_file.readline().split(sep =',')][1])
    par_lp_phi += [float(p) for p in par_file.readline().split(sep=',')]
    orders.append([int(p) for p in par_file.readline().split(sep =',')][1])
    par_lp_theta += [float(p) for p in par_file.readline().split(sep=',')]
    orders.append([int(p) for p in par_file.readline().split(sep =',')][1])
    par_b_width += [float(p) for p in par_file.readline().split(sep=',')]
    orders.append([int(p) for p in par_file.readline().split(sep =',')][1])
    par_lp_width += [float(p) for p in par_file.readline().split(sep=',')]

def get_gauss_euclid(coord_in):

    samples = int(len(coord_in)/3)
    coord = np.array(coord_in.reshape(samples,3,3))

    loc_axes = local_axes(coord)
    origin = coord[:,0,:]

    coord_loc = change_cs(coord, loc_axes, origin).reshape(samples*3,3)

    coord_s = to_spherical(coord_loc).reshape(samples,3,3)

    g_s = get_gauss_spherical(coord_s,True)

    g_e_loc = np.zeros_like(g_s)

    g_e_loc[:,:,0] = g_s[:,:,0] * np.cos(g_s[:,:,1]) * np.sin(g_s[:,:,2])
    g_e_loc[:,:,1] = g_s[:,:,0] * np.sin(g_s[:,:,1]) * np.sin(g_s[:,:,2])
    g_e_loc[:,:,2] = g_s[:,:,0] * np.cos(g_s[:,:,2])

    g_e = np.zeros_like(g_e_loc)

    for i in range(4): #Gaussian centers
        for a in range(3):  #Axes
            for j in range(3): #Cart. coordinates
                g_e[:,i,j] += g_e_loc[:,i,a] * loc_axes[:,a*3+j]

        g_e[:,i,:3] += origin

    g_e[:,:,3] = g_s[:,:,3]
    return g_e



def get_gauss_spherical(coord_in, is_prepared = False):
    coord = np.array(coord_in)
    if not is_prepared:
        if coord.shape[1] != 3 :
            raise Exception("Atomic coordinates have to have shape (,3)")

        samples = int(len(coord)/3)
        coord = coord.reshape(samples,3,3)

        # Move to CS that is local to molecule
        coord_s = change_cs(coord, local_axes(coord), coord[:,0,:]).reshape(len(new_coord)*3,3)

        # Switch to spherical coordinates
        coord_s = to_spherical(coord_s)

        # Group into H2O molecules
        coord_s = coord_s.reshape(samples,3,3)
    else:
        coord_s = coord
    g_s = np.zeros([len(coord),4,4])


    #b_r
    g_s[:,0,0] = fit_gauss(coord_s, par_b_r, orders[0], False)
    g_s[:,1,0] = fit_gauss(coord_s, par_b_r, orders[0], True)
    #b_phi
    g_s[:,0,1] = fit_gauss(coord_s, par_b_phi, orders[1], False)
    g_s[:,1,1] = -fit_gauss(coord_s, par_b_phi, orders[1], True)
    #b_theta
    g_s[:,[0,1],2] = np.pi*.5
    #b_width
    g_s[:,0,3] = fit_gauss(coord_s, par_b_width, orders[5], False)
    g_s[:,1,3] = fit_gauss(coord_s, par_b_width, orders[5], True)

    #lp_r
    g_s[:,2,0] = fit_gauss(coord_s, par_lp_r, orders[2], False)
    g_s[:,3,0] = g_s[:,2,0]
    #lp_phi
    g_s[:,2,1] = fit_gauss(coord_s, par_lp_phi, orders[3], False) + np.pi
    g_s[:,3,1] = g_s[:,2,1]
    #lp_theta
    g_s[:,2,2] = fit_gauss(coord_s, par_lp_theta, orders[4], False) + np.pi*.5
    g_s[:,3,2] = -g_s[:,2,2] + np.pi
    #lp_width
    g_s[:,2,3] = fit_gauss(coord_s, par_lp_width, orders[6], False)
    g_s[:,3,3] = g_s[:,2,3]


    return g_s



def local_axes(X):

    oh1 = X[:,1,:] - X[:,0,:]
    oh2 = X[:,2,:] - X[:,0,:]
    x_axis = normalize((normalize(oh1)+ normalize(oh2)))

    z_axis = np.cross(oh1, x_axis, axis = 1)
    z_axis = normalize(z_axis)

    y_axis = np.cross(z_axis,x_axis,axis = 1)
    y_axis = normalize(y_axis)

    axes = np.concatenate((x_axis,y_axis,z_axis), axis = 1 )

    return axes

def normalize(X):

    result = np.zeros_like(X)
    norm = np.linalg.norm(X, axis =1)
    for i in range(3):
        result[:,i] = X[:,i]/norm
    return result

def to_spherical(X):

    r = np.linalg.norm(X,axis=1)
    phi = np.arctan2(X[:,1],X[:,0])
    theta = np.arccos(X[:,2]/r)

    return np.concatenate((r.reshape(len(X),1),
                           phi.reshape(len(X),1),
                           theta.reshape(len(X),1)),
                           axis = 1)


def change_cs(coord_in, axes, origin):
    coord = np.array(coord_in)
    for i in range(coord.shape[1]):
        coord[:,i,:] = coord[:,i,:] - origin

    new_coord = np.zeros(np.shape(coord))

    for i in range(coord.shape[1]):
        for a in range(3):
            new_coord[:,i,a] = np.diag(np.dot(coord[:,i,:],axes[:,a*3:a*3+3].T))

    return new_coord


def fit_gauss(coord, par, order = 2, switch = False):

    phi_b = np.zeros(len(coord))
    cnt = 0


    phi = coord[:,1,1]
    if switch:
        r1 = coord[:,1,0]
        r2 = coord[:,2,0]
    else:
        r1 = coord[:,2,0]
        r2 = coord[:,1,0]


    for i in range(order+1):
        for j in range(order+1):
            for k in range(order+1):

                phi_b +=  par[cnt] * r1**i * r2**j * phi**k

                cnt += 1

    return phi_b
