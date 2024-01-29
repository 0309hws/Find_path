import numpy as np
import utils
import os
import copy
np.random.seed(2022)
HERO_RADIUS = 15
IMAGE_SIZE = 100 # the downsampling image size
PATH_STEP = 40 # the length of pixels for each steps

if __name__ == "__main__": 
    # load map and no collision map
    map_path = 'perspective_finder/freemap.npy' 
    noc_path = 'perspective_finder/no_collision_map.npy'
    map_data = np.load(map_path).astype(np.float64)
    true_map = map_data[1:, 1:] == 1
    if os.path.exists(noc_path):
        no_collision_map = np.load(noc_path)
    else:
        radius = round(HERO_RADIUS/min(round(abs(map_data[0,2]-map_data[0,1]),2), round(abs(map_data[2,0]-map_data[1,0]) ,2)), 2)
        no_collision_map = utils.compute_no_collision_map(true_map, radius)
        np.save('perspective_finder/no_collision_map.npy', no_collision_map)
    # downsample the map to find an exploration path
    down_rate = int(round(true_map.shape[0]/IMAGE_SIZE))
    dp_free_map = utils.ds_mp(true_map, down_rate)
    dp_no_collision = utils.ds_mp(no_collision_map, down_rate)
    utils.visualize_2d_map(no_collision_map, 'no_collision_map')
    utils.visualize_2d_map(dp_no_collision, 'dp_no_collision_map')
    utils.visualize_2d_map(true_map, 'free_map')
    utils.visualize_2d_map(dp_free_map, 'dp_free_map')
    # find a path to explore the environment
    path = utils.find_path(dp_free_map, no_collision_map = dp_no_collision, patience = 10)
    # if not os.path.exists('perspective_finder/path_.npy'):
    #     path = utils.find_path(dp_free_map, no_collision_map = dp_no_collision, patience = 10)
    # else:
    #     path = np.load('perspective_finder/path_.npy', allow_pickle=True)[0]
    # utils.visualize_final_path(path, dp_no_collision)
    np.save('perspective_finder/path_.npy', np.array([path], dtype=object))
    path_modi = utils.modify_path(path, dp_no_collision)
    # utils.visualize_final_path(path_modi, dp_no_collision)
    
    # restore the path
    path_ori = utils.upsample_path(path_modi, down_rate, no_collision_map)
    path_ori_ = copy.deepcopy(path_ori)
    utils.visualize_final_path(path_ori, true_map)

    for i in path_ori.keys():
        path_ori[i]['x'] = map_data[path_ori[i]['x'], 0]
        path_ori[i]['y'] = map_data[0, path_ori[i]['y']]
        
    path_interpolate = utils.interpolate_path(path_ori_, no_collision_map, interpolate_rate= PATH_STEP)
    utils.visualize_final_path(path_interpolate, true_map)
    np.save('perspective_finder/path_interpolate.npy', np.array([path_interpolate], dtype=object))
    # utils.visualize_final_path(path_interpolate, true_map)
    # draw the path
    # utils.visualize_path(path_interpolate, true_map, down_rate)
    # store the path in the coordinate of real world
    for i in path_interpolate.keys():
        path_interpolate[i]['x'] = map_data[path_interpolate[i]['x'], 0]
        path_interpolate[i]['y'] = map_data[0, path_interpolate[i]['y']]
    
    np.save('perspective_finder/path_final.npy', np.array([path_interpolate], dtype=object))
