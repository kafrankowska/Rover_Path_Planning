import sys
import numpy as np
import math
import pandas as pd
import os
import numpy as np
sys.path.append("../rover_planner")
from a_star import AStarPlanner
import mapping
from PIL import Image
import random


def get_default_params():
    robot_radius = 1
    map_dict = mapping.init_dict()
    dem_path = '../dataset/LRO_NAC_DEM_Apollo_15_26N004E_150cmp.tif'
    out_path = '../dataset/Auto_mapset'

    Image.MAX_IMAGE_PIXELS = None
    test_img = np.asarray(Image.open(dem_path), dtype=np.float16)
    width,height = test_img.shape[0],test_img.shape[1]
    
    return map_dict,dem_path,out_path,test_img, width, height

def open_og_grid(og_path):
    with open(og_path, 'rb') as f:
        og = np.load(f)
    return og

def get_random_points(width, height, min_dist=100):
    dist = 0
    while dist<min_dist:
        sx = random.randint(0,width)
        sy = random.randint(0,height)
        gx = random.randint(0,width)
        gy = random.randint(0,height)
        dist = math.dist([sx,sy],[gx,gy])
    return sx,sy,gx,gy

def check_if_path_possible(og_map,sx,sy,gx,gy, res=0.5,
                          robot_radius=1):
        a_star = AStarPlanner(resolution=res, rr=robot_radius, 
                              obstacle_map=og_map, show_animation=False,)
        a_star.set_goals(sx, sy, gx, gy)
        rx, ry = a_star.planning()
        if len(rx)>1:
            print('Path steps: {}'.format(len(rx)))
            return True
        else:
#             print('Path not posible')
            return False

def create_mapset(map_dict, out_path, test_img, dem_path, export_dict=True,
                  step=500, max_paths=100,max_iters=100, res=0.5, min_dist=100, width=500, height=500):

    i_range = math.floor(width/step)
    j_range = math.floor(height/step)

    paths = 0

    for i in range(i_range):
        for j in range(j_range):
            if paths >= max_paths:
                break
            x_min = i*step
            y_min = j*step

            try:
                img, name_idx = mapping.get_map(dem_path, default=False, 
                                        x_min=x_min,y_min=y_min,crop_width=step,crop_height=step, dem=test_img)

            except Exception as err:
    #             print('Invalid values found, omit img')
                continue 

            max_height = abs((np.min(img))) + 10
            slope = mapping.calculate_slope(img)
            safe_slope = mapping.safe_area(slope)
            width = safe_slope.shape[0]
            height = safe_slope.shape[1]

            i = 0
            found_path = False

            while not found_path:
                if i >= max_iters:
                    print('Max iterations without finding path. Omit img'.format(i))
                    break
                sx,sy,gx,gy = get_random_points(width, height, min_dist)
                found_path = check_if_path_possible(safe_slope,sx,sy,gx,gy)
                i+=1

            print('Found points after {} iterations'.format(i))
            if found_path:
                dem_name = mapping.save_dem(img, out_path=out_path, name='DEM', name_idx=name_idx)
                slope_name = mapping.save_dem(slope, out_path, name='SLOPE', name_idx=name_idx)
                og_name = mapping.save_dem(safe_slope, out_path, name='OG', name_idx=name_idx)

                start_end_name = mapping.plot_start_end(safe_slope,(sy,sx),(gy,gx), export_img=True, 
                                                out_path=out_path, name_idx=name_idx, step=math.ceil(step/10))
                mesh_plot_name = mapping.plot_mesh(img, export_img=True, out_path=out_path, name_idx=name_idx, height=max_height)
                slope_plot_name = mapping.analyze_slope_map(img, True, out_path=out_path, name_idx=name_idx, step=math.ceil(step/10))
                mapping.append_test_case(map_dict, og_name, dem_name, slope_name, sx, sy, gx, gy, res, name_idx, width, height)
                paths += 1
                
    if export_dict:
        mapping.export_test_case_table(map_dict, out_path=out_path, name='mapset_1000x1000')
    return map_dict


