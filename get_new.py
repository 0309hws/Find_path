import numpy as np
noc = np.load('perspective_finder/no_collision_map.npy')
free = np.load('perspective_finder/freemap.npy')
free[1:,1:]=noc
np.save('perspective_finder/freemap_noc.npy',free)