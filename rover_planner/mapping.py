import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from mpl_toolkits.mplot3d.axes3d import *
import os
import pandas as pd
from math import sqrt, atan
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from matplotlib import colors


def init_dict():
    test_dict = {
        'Name_idx': [],
        'Occupancy grid path': [],
        'DEM path': [],
        'Slope aspect path': [],
        'Start x coord': [],
        'Start y coord': [],
        'Goal x coord': [],
        'Goal y coord': [],
        'Resolution': [],
        'Map width': [],
        'Map height': [],
    }
    return test_dict


def append_test_case(test_dict, og, dem, slope, sx, sy, gx, gy, res, name_idx, width, height):
    test_dict['Occupancy grid path'].append(og)
    test_dict['DEM path'].append(dem)
    test_dict['Slope aspect path'].append(slope)
    test_dict['Start x coord'].append(sx)
    test_dict['Start y coord'].append(sy)
    test_dict['Goal x coord'].append(gx)
    test_dict['Goal y coord'].append(gy)
    test_dict['Resolution'].append(res)
    test_dict['Name_idx'].append(name_idx)
    test_dict['Map width'].append(width)
    test_dict['Map height'].append(height)
    return test_dict

def export_test_case_table(test_dict, out_path='../dataset',name='test_table', name_idx=None):
    if name_idx is not None:
        name = name + '_'+ name_idx + '.csv'
        table_name = os.path.join(out_path, name)
    else:
        table_name = os.path.join(out_path, name+'.csv')
    ds = pd.DataFrame.from_dict(test_dict, orient='columns')
    ds.to_csv(table_name, index=False)
    print(table_name)
    
    return table_name


# resource: https://astrogeology.usgs.gov/search/map/Moon/LMMP/Apollo15/LRO_NAC_DEM_Apollo_15_26N004E_150cmp
# Height (Elevation m) = DN; Planetary Radius = DN + 1737400m
def get_def_params():
    Image.MAX_IMAGE_PIXELS = None
    dem_path = '../dataset/LRO_NAC_DEM_Apollo_15_26N004E_150cmp.tif'
    out_path='../dataset/Mapset'
    return dem_path, out_path

def get_map(img_path= '../dataset/LRO_NAC_DEM_Apollo_15_26N004E_150cmp.tif', default=True, x_min=0,y_min=0,crop_width=500,crop_height=500, dem=None):
    if dem is None:
        img = Image.open(img_path)
        img = np.asarray(img, dtype=np.float16)
    else:
        img = dem.copy()
    if default:
        width,height = img.shape[0],img.shape[1]
        x_min,y_min = int(width/2)-2500,int(height/2)-2500

    img = img[x_min:x_min+crop_width,y_min:y_min+crop_height]
    
    is_nan = math.isnan(np.mean(img)) or math.isinf(np.mean(img))
    if is_nan:
        raise NameError('Img value error')
    img = img - np.mean(img)

    name_idx = 'X_{}_{}_Y_{}_{}'.format(x_min,x_min+crop_width,y_min,y_min+crop_height)
    return img,name_idx

def save_dem(img, out_path='../dataset', name='dem', name_idx=None):
    if name_idx is not None:
        name = name+'_' + name_idx + '.npy'
    else: 
        name = name+'.npy'
    out_name = os.path.join(out_path, name)
    with open(out_name, 'wb') as f:
        np.save(f, img)
    print(out_name)
    return out_name

def open_dem(dem_path):
    with open(dem_path, 'rb') as f:
        dem = np.load(f)
    return dem


def plot_mesh(img, export_img=False, out_path='../dataset', name_idx=None, height=20, plot_name=None, res=1.5):
    xi = np.arange(0, img.shape[0],1) * res # Grid in meters
    yi = np.arange(0, img.shape[1],1) * res # Grid in meters
    z = img.flatten() # Height in meters
    Z = img
    X, Y = np.meshgrid(xi, yi)

    fig = plt.figure(figsize=(20,16))
    ax = Axes3D(fig)
  #  ax.scatter3D(X,Y,z,c=z,cmap=plt.cm.jet)  

    my_cmap = plt.cm.RdYlGn_r
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=my_cmap,
                       linewidth=0, antialiased=True)
    fig.colorbar(surf, ax = ax,
             shrink = 0.3,
             aspect = 10,
             format="%.1f m")
    
    cset = ax.contourf(X, Y, Z,
                   zdir ='z',
                   offset = -height,
                   cmap = 'Greys_r')
    if plot_name is None:
        if name_idx is not None: 
            plot_name = 'LRO NAC DEM Apollo 15 26N004E  \n ---------------------------------------------------------------\n Obszar: {}'.format(name_idx)
        else:
             plot_name = 'LRO NAC DEM Apollo 15 26N004E'
    else:
        if name_idx is not None: 
            plot_name = plot_name + '_' + name_idx
    
    fig.suptitle(plot_name, fontsize=20)

    ax.set_xlabel('Siatka mapy: kierunek X [m]', fontsize = 16)
    ax.set_ylabel('Siatka mapy: kierunek Y [m]', fontsize = 16)
    ax.set_zlabel('Wysokość terenu [m]', fontsize = 16)

    ax.set_zlim(-height, height)
    if export_img:
        if name_idx is not None:
            name = 'DEM_plot_' + name_idx + '.jpg'
            plot_name = os.path.join(out_path, name)
        else:
            plot_name = os.path.join(out_path, 'DEM_plot.jpg')
        plt.savefig(plot_name)
        plt.close()
        return plot_name
    else:
        plt.show()
    
def calculate_slope(img, deg=True, res=1.5):
    slope = np.zeros_like(img)
     # x,y res = 0,5 m
    for x in range(2,img.shape[0]-2):
        for y in range(2,img.shape[1]-2):
          #  mean_xp_axis = img[x+1:x+2,y].mean() 
          #  mean_xn_axis = img[x-2:x-1,y].mean()
          #  dz_dx = mean_xp_axis - mean_xn_axis/(res*2)
            
          #  mean_yp_axis = img[x,y+1:y+2].mean() 
          #  mean_yn_axis = img[x,y-2:y-1].mean()
          #  dz_dy = mean_yp_axis - mean_yn_axis/(res*2)
            dz_dx = (img[x+1,y]-img[x-1,y]) /(res*2)
            dz_dy = ((img[x,y+1]-img[x,y-1])/(res*2))
            slope[x,y]= atan(sqrt(dz_dx**2+dz_dy**2))
    if deg:
        slope = np.rad2deg(slope)
        print('Min slope: {} Median slope: {} Max slope: {}'.format(np.min(slope), np.median(slope[slope>0]), np.max(slope)))
    slope = slope[2:img.shape[0]-2,2:img.shape[1]-2]
    return slope

def safe_area(safe_slope, thresh=40):
    safe_slope[safe_slope < thresh] = 0
    safe_slope[safe_slope >= thresh] = 255
    return np.array(safe_slope,dtype=np.int8)

def plot_safe_slope(safe_slope, export_img=True, out_path='../dataset', name_idx=None, step=50, res=1.5):
    danger_color = (0.6, 0.3, 0.3, 0.6)
    safe_color = (0.1, 0.7, 0.2, 0.1)
    cmp=ListedColormap([danger_color,safe_color])
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots(figsize=(10,10), dpi=144)
        xi = np.arange(0, safe_slope.shape[0],1) * 0.5 # Grid in meters
        yi = np.arange(0, safe_slope.shape[1],1) * 0.5 # Grid in meters
        #Z = 
        #plt.pcolormesh(safe_slope, cmap='Greys')
        #plt.scatter(xi,yi, cmap='Greys')

        ax.imshow(safe_slope, cmap=cmp)
        fig.suptitle("Ocena możliwości trawersu", fontsize=16)
        plt.xlabel("Kierunek x [m]", fontsize=10)
        plt.ylabel("Kierunek Y [m]", fontsize=10)
        ax.invert_yaxis()

        locs_x = (np.arange(0, safe_slope.shape[0], step=step)) 
        labels_x = locs_x*res
        plt.xticks(ticks=locs_x, labels=labels_x)

        locs_y = (np.arange(0, safe_slope.shape[1], step=step)) 
        labels_y = locs_y*res
        plt.yticks(ticks=locs_y, labels=labels_y)
        

        legend_handles = [Patch(color=safe_color, label='Bezpieczny'),  
                          Patch(color=danger_color, label='Niebezpieczny')]  
        plt.legend(handles=legend_handles, ncol=2, bbox_to_anchor=[0.5, 1.02], loc='lower center', fontsize=8, handlelength=.8)
        #plt.legend()
        if export_img:
            if name_idx is not None:
                name = 'Safe_slope_plot_' + name_idx + '.jpg'
                plot_name = os.path.join(out_path, name)
            else:
                plot_name = os.path.join(out_path, 'Safe_slope_plot.jpg')
            plt.savefig(plot_name)
            plt.close()
            return plot_name
        else:
            plt.show()
        
def analyze_slope_map(img, export_img=True, out_path='../dataset', name_idx=None, step=50):
    slope = calculate_slope(img)
    safe_slope = safe_area(slope)
    plot_safe_slope(safe_slope,export_img, out_path=out_path, name_idx=name_idx, step=step)
    

def plot_start_end(img,start,end, export_img=False, out_path='../dataset', name_idx=None, step=50, res=1.5):

    danger_color = (0.3, 0.3, 0.3, 0.3)
    safe_color = (0.9, 1, 0.9, 0.1)

    markers_color = (0.92, 0.7, 0, 0.8)
    
    font = {'family': "monospace",
        'color':  (0.92, 0.7, 0, 1),
        'weight': 'bold',
        'size': 10,
        }

    
    cmp=ListedColormap([danger_color,safe_color])
    cmp=ListedColormap([safe_color,danger_color])
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots(figsize=(10,10), dpi=144)
        xi = np.arange(0, img.shape[0],1) * res # Grid in meters
        yi = np.arange(0, img.shape[1],1) * res # Grid in meters

        ax.imshow(img, cmap=cmp)
        
        start_point = ax.plot(start[0], start[1], marker="H", markeredgecolor=markers_color,
                            markersize=12, markerfacecolor=markers_color, label='Punkt startowy') # Start point plot 
        
        ax.text(start[0]+start[0]*0.02, 
                start[1]+start[1]*0.02, r'Punkt startowy', fontdict=font) 
        end_point = ax.plot(end[0], end[1], marker="X", label='Punkt końcowy', 
                            markersize=12, markeredgecolor=markers_color,markerfacecolor=markers_color) # End point plot )
        ax.text(end[0]+end[0]*0.02, 
                end[1]+end[1]*0.02, r'Punkt końcowy', fontdict=font) 
        
        plt.title("Punkt startu i mety dla trasy", fontsize=14)
        plt.xlabel("Kierunek x [m]", fontsize=10)
        plt.ylabel("Kierunek Y [m]", fontsize=10)
        ax.invert_yaxis()
        
        locs_x = (np.arange(0, img.shape[0], step=step)) 
        labels_x = locs_x*res
        plt.xticks(ticks=locs_x, labels=labels_x)

        locs_y = (np.arange(0, img.shape[1], step=step)) 
        labels_y = locs_y*res
        plt.yticks(ticks=locs_y, labels=labels_y)
        
            
        if export_img:
            if name_idx is not None:
                name = 'S_E_plot_' + name_idx + '.jpg'
                plot_name = os.path.join(out_path, name)
            else:
                plot_name = os.path.join(out_path, 'Start_end_plot.jpg')
            print(plot_name)
            plt.savefig(plot_name)
            plt.close()
            return plot_name
        else:
            plt.show()

def default_exporter(add_test_case=False):
    dem_name = save_dem(img, out_path, name='DEM', name_idx=name_idx)
    slope = calculate_slope(img)
    slope_name = save_dem(slope, out_path, name='SLOPE', name_idx=name_idx)
    safe_slope = safe_area(slope)

    og_name = save_dem(safe_slope, out_path, name='OG', name_idx=name_idx)
    mesh_plot_name = plot_mesh(img, export_img=True, out_path=out_path, name_idx=name_idx, height=50)
    slope_plot_name = analyze_slope_map(img, True, out_path=out_path, name_idx=name_idx, step=50)
    
    res = 0.5 # [m]
    sx = 235.0 / res  # [m]
    sy = 150.0 / res  # [m]
    gx = 50.0 / res   # [m]
    gy = 77.0 / res  # [m]


    start_end_name = plot_start_end(safe_slope,(sy,sx),(gy,gx), export_img=True, out_path=out_path, name_idx=name_idx, step=50)
    
    width = safe_slope.shape[0]
    height = safe_slope.shape[1]
    
    if add_test_case:
        append_test_case(test_dict, og_name, dem_name, 
                         slope_name, sx, sy, gx, gy, res, name_idx, width, height)

        export_test_case_table(test_dict, out_path=out_path,
                               name_idx=name_idx)
    
    print('Exported default mapset')

