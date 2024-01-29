from scipy.ndimage import sobel
from scipy.ndimage import distance_transform_edt
import numpy as np
import matplotlib.pyplot as plt
import os
import heapq
import copy
GOAL_PERCENTAGE = 0.99
VISION_RANGE = 10
VISION_ANGLE = np.pi/2  # twice of half angle
####################################################################################
# find path utils
####################################################################################
def get_start_nodes(input_map, no_collision_map):
    assert no_collision_map is not None
    output_nodes = []
    directions = np.array(
        [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]])
    x_map, y_map = np.meshgrid(
        np.arange(input_map.shape[0]), np.arange(input_map.shape[1]))
    x_map, y_map = x_map.T, y_map.T
    for direction in directions:
        value_map = x_map * direction[0] + y_map * direction[1]
        value_map = np.float32(value_map) * no_collision_map - \
            1000000000 * (1 - no_collision_map)
        max_value = np.max(value_map)
        max_points = np.argwhere(value_map >= max_value - 1)
        # now select a random point
        seed = np.random.choice(len(max_points))
        x, y = max_points[seed]
        new_node = {'x': x, 'y': y, 'look_around': True, 'new': True}
        output_nodes.append(new_node)
        print(output_nodes[-1])
    return output_nodes

def find_path(input_map, no_collision_map, patience=100):
    """
        input_map: 2d array, 0: void, 1: ground
        no_collision_map: 2d array, 0: void, 1: accessible ground
        interest_points_to_look: 2d array, shape (n, 2), the points to look
        add_new_seeds: bool, whether to add new seeds during the search
        max_iter: int, the maximum number of iterations
        patience: int, the maximum number of iterations without improvement
        print_interval: int, the interval to print the current best coverage
        return: a list of nodes, the path
    """
    assert no_collision_map is not None
    start_nodes = get_start_nodes(input_map, no_collision_map)
    path = search_path(start_nodes, input_map, no_collision_map, patience)
    return path

def search_path(start_points, input_map, no_collision_map=None, patience = 100):
    path = dict()
    num = 0
    map_to_explore = copy.deepcopy(input_map)
    current_point = start_points[np.random.randint(len(start_points))]
    path[str(num)] = current_point
    while np.sum(map_to_explore)/np.sum(input_map)>1-GOAL_PERCENTAGE:
        # explore current point and update the point need to explore
        explored_map = explore_point(current_point, input_map)
        map_to_explore = np.logical_and(map_to_explore, explored_map)
        if np.sum(map_to_explore) == 0:
            break
        visualize_2d_map(0.6*map_to_explore + 0.4*input_map, scatters=current_point)
        # choose the next node to explore and move to it
        possible_next_point = choose_possible_next_point(current_point, map_to_explore, no_collision_map)
        count = 0
        # utils.visualize_final_path(path, no_collision_map)
        while True:
            try:
                next_point = possible_next_point[count]
            except:
                try:
                    next_point = possible_next_point[0]
                except:
                    return path
            path_to_next, flag = dijkstra(no_collision_map, (current_point['x'], current_point['y']), tuple(next_point))
            if not flag:
                next_point = {'x': next_point[0], 'y': next_point[1], 'look_around': True, 'new': False}
                num += 1
                path[str(num)] = next_point
                current_point = next_point
                break
            if count >= patience:
                path_to_next = []
                next_point = {'x': next_point[0], 'y': next_point[1], 'look_around': True, 'new': True}
                num += 1
                path[str(num)] = next_point
                current_point = next_point
                break
            count += 1
        # Update the points need to explore after moving from current point to the next point
        map_to_explore = update_explored_map(path_to_next, map_to_explore, input_map)
    return path

def upsample_path(path, down_rate, non_collision_map):
    """
    Upsample a path from a downsampled image back to the original scale.

    Args:
    - path (dict of points): The path in the downsampled image, where each node has a coordinate (row, col).
    - downscale_factor (int): The factor by which the original image was downsampled.

    Returns:
    - dict of points: The upsampled path in the original image scale.
    """
    path_ori = {}
    for index, node in path.items():
        path_ori[index] = node
        path_ori[index]['x'] = min(int(down_rate/2 + node['x'] * down_rate), non_collision_map.shape[0])
        path_ori[index]['y'] = min(int(down_rate/2 + node['y'] * down_rate), non_collision_map.shape[1])
    return path_ori

def interpolate_path(path_ori, no_collision_map, interpolate_rate):
    path_inter = dict()
    num = 0
    for index, point in enumerate(path_ori.values()):
        if index ==len(path_ori) - 1:
            break
        x_curr, y_curr = point["x"], point["y"]
        x_next, y_next = path_ori[str(index+1)]['x'], path_ori[str(index+1)]['y']
        result, flag = dijkstra(no_collision_map, (x_curr, y_curr), (x_next, y_next))
        if not flag:
            path_ori[str(index+1)]['new'] = False
            path_inter[str(num)+'_0'] = path_ori[str(index)]
            for i, j in enumerate(range(1, len(result[:-1]), interpolate_rate)):
                path_inter[str(num)+'_'+str(i+1)] = {'x': result[j][0], 'y': result[j][1], 'look_around': False, 'new': False}
            num += 1
        else:
            path_ori[str(index+1)]['new'] = True
            path_inter[str(num) + '_0'] = path_ori[str(index)]
            num += 1
        
        path_inter = modify_path(path_inter, no_collision_map, 10)
    return path_inter

def modify_path(path, no_collision_map, range = 2):
    distance_map = distance_transform_edt(no_collision_map)
    visualize_2d_map(distance_map)
    # 调整路径点到最近的走道中心
    adjusted_points = dict()
    for index, point in enumerate(path.values()):
        x, y = point['x'], point['y']
        # 检查点是否已在走道中心
        if distance_map[x, y] == np.max(distance_map[x-range:x+range+1, y-range:y+range+1]):
            adjusted_points[str(index)] = {'x':x, 'y':y, 'look_around':point['look_around'], 'new':point['new']}
            continue

        # 搜索周围区域以找到最佳位置
        local_max = np.unravel_index(np.argmax(distance_map[x-range:x+range+1, y-range:y+range+1]), (2*range+1, 2*range+1))
        adjusted_points[str(index)] = {'x':local_max[0] + x - range, 'y':local_max[1] + y - range, 'look_around':point['look_around'], 'new':point['new']}
    return adjusted_points


def dijkstra(grid, start, end):
    flag = False
    grid = 1 - grid
    def neighbors(pos):
        x, y = pos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 上下左右
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0:
                yield nx, ny

    heap = [(0, start)]  # (距离, 节点)
    visited = set()
    distance = {start: 0}
    prev = {start: None}

    while heap:
        dist, current = heapq.heappop(heap)
        if current == end:
            break

        if current in visited:
            continue

        visited.add(current)

        for neighbor in neighbors(current):
            new_dist = dist + 1  # 假设每一步的代价是1
            if neighbor not in distance or new_dist < distance[neighbor]:
                distance[neighbor] = new_dist
                prev[neighbor] = current
                heapq.heappush(heap, (new_dist, neighbor))

    # 回溯路径
    path = []
    try:
        while end is not None:
            path.append(end)
            end = prev[end]
    except:
        flag = True
    return path[::-1], flag

####################################################################################
# utils
####################################################################################
def explore_point(point, input_map, vision_range=VISION_RANGE):
    """
        input_map: 2d array, 0: occupied, 1: free
        point: dict with keys('x', 'y', 'look_around', 'new')

        return: the round covered map of from current point, 0 covered, 1 cannot cover
    """
    dist_map, _ = get_dist_map_and_angle_map(point['x'], point['y'], input_map)
    new_map = dist_map <= vision_range
    check_points = np.column_stack(np.where(new_map))
    check_list = set()
    result = np.zeros_like(new_map)
    for points in check_points:
        if tuple(points) in list(check_list):
            continue
        line_points = bresenham_line(point['x'], point['y'], points[0], points[1], new_map)
        check_list.update(line_points)
        for i, j in line_points:
            if input_map[i, j]==1:
                result[i, j]=1
            else:
                break
    # visualize_2d_map(0.6*result + 0.4*input_map)
    return 1 - result

def choose_possible_next_point(point, map_to_explore, no_collision_map, additional=30):
    """
        point: dict with keys('x', 'y', 'look_around', 'new')
        no_collision_map: 2d array, 0: occupied, 1: free
        map_to_explore: 2d array, 0: don't need to explore, 1: still need to explore
    
        return: 2d array, the indices of possible next point to reach from near to far
    """
    dist_map, _ = get_dist_map_and_angle_map(point['x'], point['y'], no_collision_map)
    dist_map_unexplored = dist_map * map_to_explore * no_collision_map
    # calculate dijkstra
    dijkstra_map_unexplored = dist_map_unexplored * (dist_map<=(VISION_RANGE+additional))
    if np.sum(dijkstra_map_unexplored)!=0:
        # get non_zero indices
        non_zero_indices = list(np.argwhere(dijkstra_map_unexplored != 0)) 
        for index in non_zero_indices:
            path, flag = dijkstra(no_collision_map, (point['x'], point['y']), tuple(index))
            if flag:
                dijkstra_map_unexplored[index[0], index[1]]=0
            else:
                dijkstra_map_unexplored[index[0], index[1]]=len(path)
        if np.sum(dijkstra_map_unexplored)!=0:
            # get non_zero indices
            non_zero_indices = np.argwhere(dijkstra_map_unexplored != 0)
            # get non_zero elements
            non_zero_elements = dijkstra_map_unexplored[non_zero_indices[:, 0], non_zero_indices[:, 1]]
            # sort the non_zero elements and get the indices after sort
            sorted_indices = non_zero_indices[np.argsort(non_zero_elements)]
            return sorted_indices  
    # get non_zero indices
    non_zero_indices = np.argwhere(dist_map_unexplored != 0)
    # get non_zero elements
    non_zero_elements = dist_map_unexplored[non_zero_indices[:, 0], non_zero_indices[:, 1]]
    # sort the non_zero elements and get the indices after sort
    sorted_indices = non_zero_indices[np.argsort(non_zero_elements)]
    return sorted_indices
    
def update_explored_map(path, map_to_explore, input_map):
    """
        path: list, path from current point to next point
        map_to_explore: 2d array, 0: don't need to explore, 1: still need to explore
        input_map: 2d array, 0: occupied, 1: free

        return: the new map_to_explore after the move from current point to the next
    """
    new_map_to_explore = copy.deepcopy(map_to_explore)
    for i in range(len(path)-1):
        angle = np.arctan2(path[i+1][1]-path[i][1], path[i+1][0]-path[i][0])
        move = 1 - get_covered_map(input_map, path[i][0], path[i][1], angle)
        new_map_to_explore = np.logical_and(new_map_to_explore,move)
        # visualize_2d_map((1-move)*0.6+input_map*0.4)
    return new_map_to_explore

def get_covered_map(input_map: np.ndarray, x: int, y: int, theta: float, vision_range=VISION_RANGE, vision_angle=VISION_ANGLE):
    """
        prev_map: the covered input_map before the move
        x, y: the position after the move
        theta: the towards angle after the move
    """
    dist_map, angle_map = get_dist_map_and_angle_map(
        x, y, input_map, theta)
    new_map = np.logical_and(dist_map <= vision_range,
                             np.abs(angle_map) <= vision_angle/2)
    check_points = np.column_stack(np.where(new_map))
    check_list = set()
    result = np.zeros_like(new_map)
    for point in check_points:
        if tuple(point) in list(check_list):
            continue
        line_points = bresenham_line(x, y, point[0], point[1], new_map)
        check_list.update(line_points)
        for i, j in line_points:
            if input_map[i, j]==1:
                result[i, j]=1
                # visualize_2d_map(0.6*result + 0.4*input_map, scatters={'x':[i[0] for i in line_points], 'y':[i[1] for i in line_points]})
            else:
                break
    # visualize_2d_map(0.6*result + 0.4*input_map)
    return result

def bresenham_line(x0, y0, x1, y1, view_map):
    """ Bresenham's Line Algorithm
    Produces a list of tuples inside the viewmap along the direction from start and end
    
    :param x0: x of start point
    :param y0: y of start point
    :param x1: x of end point
    :param y1: y of end point
    :param view_map: the initial coverage view
    :return: points that form the line 
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    points.append((x0, y0))
    if dx > dy:
        err = dx / 2.0
        while True:
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
            if x>=0 and x<view_map.shape[0] and y>=0 and y<view_map.shape[1] and view_map[x, y] == 1:
                points.append((x, y))
            else:
                break
    else:
        err = dy / 2.0
        while True:
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
            if x>=0 and x<view_map.shape[0] and y>=0 and y<view_map.shape[1] and view_map[x, y] == 1:
                points.append((x, y))
            else:
                break
    return points

def get_dist_map_and_angle_map(x: int, y: int, input_map: np.ndarray, angle: float = 0):
    """
        x, y: the position of the hero
        angle: the towards angle of the hero
        map: the map of the scene, only its shape is used

        return: dist_map, angle_map
        dist_map: the distance to the hero
        angle_map: the additional angle hero need to rotate to face the point
    """
    x_map, y_map = np.meshgrid(np.arange(input_map.shape[0]), np.arange(input_map.shape[1]))
    x_map = x_map.T - x
    y_map = y_map.T - y
    dist_map = np.sqrt(x_map**2 + y_map**2)
    angle_map = np.arctan2(y_map, x_map) # in the range [-pi, pi]
    angle_map = subtract_angle(angle_map, angle)
    return dist_map, angle_map


def subtract_angle(a, b):
    """
        return the angle a - b, in the range [-pi, pi]
    """
    return np.mod(a - b + np.pi, 2*np.pi) - np.pi

def ds_mp(input_array, down_rate):
    """
    Perform max pooling on a 2D array (input_array) to downsample using a given down_rate.
    
    Args:
    - input_array (numpy.ndarray): The input array (image).
    - down_rate (int): downsampling_rate
    
    Returns:
    - numpy.ndarray: The array after max pooling.
    """
    # Calculate the size of the output array
    output_height = ((input_array.shape[0] - down_rate) // down_rate) + 1
    output_width = ((input_array.shape[1] - down_rate) // down_rate) + 1
    if output_height*down_rate<input_array.shape[0]:
        output_height += 1
    if output_width*down_rate<input_array.shape[1]:
        output_width += 1
    pooled_array = np.zeros((output_height, output_width))

    # Apply max pooling
    for i in range(output_height):
        for j in range(output_width):
            h_start = i * down_rate
            h_end = min(h_start + down_rate, input_array.shape[0] - 1)
            w_start = j * down_rate
            w_end = min(w_start + down_rate, input_array.shape[1] - 1) 
            window = input_array[h_start:h_end, w_start:w_end]
            pooled_array[i, j] = np.min(window)
    
    return pooled_array

def compute_no_collision_map(input_map, radius):
    print("computing non collision map...")
    result = input_map != 1 # 0 free, 1 occupied
    # find the edges
    edge_horizont = sobel(input_map, axis=0, mode='constant')
    edge_vertical = sobel(input_map, axis=1, mode='constant')
    magnitude = np.hypot(edge_horizont, edge_vertical)
    edges = np.where(magnitude > 0)
    edge_coords = np.column_stack(edges)
    # expand the obstacle map according to each point in the edges
    i=0
    for one_idx in edge_coords:
        i+=1
        expand_map = get_expand_map(one_idx[0], one_idx[1], input_map, radius)
        result = np.logical_or(result, expand_map)
    print("non collision lands num: %d" % np.sum(result))
    print("non collision land percentage: %.3f" % (np.sum(result)/np.sum(input_map) * 100))
    print("non collision lands num: %d" % np.sum(1-result))
    print("non collision land percentage: %.3f" % (np.sum(1-result)/np.sum(input_map) * 100))
    return 1 - result

def get_expand_map(x, y, input_map, radius):
    """
        x, y: the position of the obstacle
        input_map: 2d array, input_map[0, :]: world indices of y, input_map[:, 0]: world indices of x, input_map[1:, 1:]: 0: undefined 1:free 2:occupied

        return: expand_map
        expand_map: the occupied map expanded from current obstacle, 0 accessible ground, 1 occupied
    """
    expand_map = np.zeros_like(input_map)
    # determine the probable zone of the expand map
    column_max = int(min(np.ceil(y + radius), input_map.shape[1] - 1))
    column_min = int(max(np.floor(y - radius), 0))
    row_max = int(min(np.ceil(x + radius), input_map.shape[0] - 1))
    row_min = int(max(np.ceil(x - radius), 0))
    # calculate the final expand map
    x_block, y_block = np.meshgrid(np.arange(row_min, row_max + 1), np.arange(column_min, column_max + 1))
    x_block = x_block.T - x
    y_block = y_block.T - y
    dist_map = np.sqrt(x_block**2 + y_block**2)
    obstacle_map = dist_map <= radius
    expand_map[row_min:row_max+1, column_min:column_max+1] = obstacle_map
    return expand_map
####################################################################################
# visualization utils
####################################################################################
VISUALIZATION_FOLDER = "perspective_finder/visualization"
def visualize_2d_map(data, title=None, xlabel=None, ylabel=None, save=True, show=False, scatters=None):
    plt.imshow(data.T, cmap="gray", origin="lower")
    if scatters:
        plt.scatter(scatters['x'], scatters['y'], color='red', s=1)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    save_name = "2d_map.png"
    if title:
        plt.title(title)
        save_name = title + ".png"
    if save:
        plt.savefig(os.path.join(VISUALIZATION_FOLDER, save_name))
    if show:
        plt.show()
    plt.close()

def visualize_path(path, background, down_rate, path_alpha=0.2, show=True, save=True, total_time=None):
    """
        path: a dict of points
        background: 2D numpy array
        path_alpha: the show alpha value of the path
        show: whether to show the visualization
        save: whether to save the visualization
        total_time: the total time of the animation
    """
    if os.path.exists('path_with_coverag.npy'):
        path_with_coverage = np.load('path_with_coverag.npy', allow_pickle=True)[0]
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        fig, ax = plt.subplots()
        
        def update(frame):
            current_point = list(path_with_coverage.values())[frame]
            ax.clear()
            ax.imshow(background.T, origin="lower")
            ax.imshow(current_point['cover_map'].T, origin="lower", alpha=path_alpha)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            if frame > 0:
                xs = [node['x'] for node in list(path_with_coverage.values())[:frame+1]]
                ys = [node['y'] for node in list(path_with_coverage.values())[:frame+1]]
                ax.scatter(xs, ys, c="r", s=0.1)
                # ax.plot(xs, ys, c="r")
            ax.set_title("step: %d" % frame)
        
        interval = 300
        if total_time:
            interval = total_time / len(path_with_coverage)
        ani = FuncAnimation(fig, update, frames=len(path_with_coverage), interval=interval)

        if show:
            plt.show()
        if save:
            ani.save(os.path.join(VISUALIZATION_FOLDER, "path.gif"), writer="pillow")
            # also save the last frame
            update(len(path)-1)
            plt.savefig(os.path.join(VISUALIZATION_FOLDER, "path.png"))
        plt.close()
    else:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        fig, ax = plt.subplots()
        
        def update(frame):
            current_point =  list(path.values())[frame]
            next_point = list(path.values())[frame+1]
            if current_point['look_around']:
                cover_map = 1 - explore_point(current_point, background, down_rate*VISION_RANGE)
                if frame>0:
                    last_point = list(path.values())[frame-1]
                    cover_map = np.logical_or(last_point['cover_map'], cover_map)
                current_point['cover_map'] = cover_map
                path[list(path.keys())[frame]] = current_point
            else:
                angle = np.arctan2(next_point['y']-current_point['y'], next_point['x']-current_point['x'])
                cover_map = get_covered_map(background, current_point['x'], current_point['y'], angle, down_rate*VISION_RANGE)
                if frame>0:
                    last_point = list(path.values())[frame-1]
                    cover_map = np.logical_or(last_point['cover_map'], cover_map)
                current_point['cover_map'] = cover_map
                path[list(path.keys())[frame]] = current_point
            ax.clear()
            ax.imshow(background.T, origin="lower")
            ax.imshow(current_point['cover_map'].T, origin="lower", alpha=path_alpha)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            if frame > 0:
                xs = [node['x'] for node in list(path.values())[:frame+1]]
                ys = [node['y'] for node in list(path.values())[:frame+1]]
                ax.scatter(xs, ys, c="r", s=0.001)
                ax.plot(xs, ys, c="r")
            ax.set_title("step: %d" % frame)
        
        interval = 300
        if total_time:
            interval = total_time / len(path)
        ani = FuncAnimation(fig, update, frames=len(path)-1, interval=interval)

        if show:
            plt.show()
        if save:
            ani.save(os.path.join(VISUALIZATION_FOLDER, "path.gif"), writer="pillow")
            # also save the last frame
            update(len(path)-1)
            plt.savefig(os.path.join(VISUALIZATION_FOLDER, "path.png"))
        np.save('path_with_coverage.npy', np.array([path], dtype=object))
        plt.close()

def visualize_final_path(path, background):
    """
        path: a dict of location
        background: 2D numpy array
    """
    # 提取路径的坐标 
    x_coords = [point["x"] for point in path.values()]
    y_coords = [point["y"] for point in path.values()]

    # 绘制占用地图
    plt.imshow(background, cmap='gray', origin='lower')

    # 绘制路径
    plt.plot(y_coords, x_coords, marker='o', color='red') # 注意坐标轴的调整

    # 添加标签
    plt.title("Path on Occupation Map")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    # 显示图表
    plt.savefig(os.path.join(VISUALIZATION_FOLDER, "final_path.png"))
    plt.show()

if __name__=='__main__':
    map_path = 'freemap.npy' 
    noc_path = 'no_collision_map.npy'
    map_data = np.load(map_path).astype(np.float64)
    true_map = map_data[1:, 1:] == 1
    path_interpolate = np.load('path.npy', allow_pickle=True)[0]
    down_rate = int(round(true_map.shape[0]/100))
    visualize_path(path_interpolate, true_map, down_rate)

