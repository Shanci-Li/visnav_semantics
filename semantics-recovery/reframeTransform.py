import os
import shutil
import progressbar
import jpype
import jpype.imports
from jpype.types import *
import sys
import argparse
import numpy as np
import laspy
import rasterio
from scipy import interpolate
import time

# This tool can be used as a script or the ReframeTransform class can be imported to use functions directly
# It allows to transform rasters (vertical transform only) and points cloud in the swiss coordinate system, and to wgs84
# Note: wgs84 is actually ETRS89, which slightly differs. Fore some applications, an additional transformation might be required

# Script:
# -s_h_srs: source horizontal spatial reference system
# -s_v_srs: source vertical spatial reference system
# -t_h_srs: target horizontal """
# -t_v_srs: target vertical """
# avalaible srs:
#   -> horizontal: lv03, lv95, wgs84
#   -> vertical: ln02, lhn95, bessel, wgs84

# USAGE:
# 1. Raster transform examples (ONLY VERTICAL TRANSFORM)
    # A. Simplest:
    # python reframeTransform.py input.tif output.tif -s_h_srs lv95 -s_v_srs ln02 -t_v_srs lhn95
    # B. Less computations, more interpolation (for large files for ex.)
    # python reframeTransform.py input.tif output.tif -s_h_srs lv95 -s_v_srs bessel -t_v_srs ln02 -transform_res 100
    # C. No interpolation (practically useless, takes a lot of time)
    # python reframeTransform.py input.tif output.tif -s_h_srs lv95 -s_v_srs bessel -t_v_srs lhn95 -nointerp
# 2. Points clouds
    # A. With las files (not compressed laz)
    # python reframeTransform.py input.las output.las -s_h_srs ln03 -s_v_srs lv95 -t_h_srs wgs84 -t_v_srs wgs84
    # B. With text files (this is used if the input file does not end with '.tif','.las','.laz')
    # python reframeTransform.py input.txt output.txt -s_h_srs ln03 -s_v_srs lv95 -t_h_srs wgs84 -t_v_srs wgs84 -sep , -skip 1
    # with sep = separator character and skip = number of header lines to skip


class ReframeTransform:
    def __init__(self):
        if not jpype.isJVMStarted():
            jpype.startJVM('-ea',classpath=os.path.join(sys.path[0],'reframeLib.jar'),convertStrings=False)
        from com.swisstopo.geodesy.reframe_lib import Reframe
        from com.swisstopo.geodesy.reframe_lib.IReframe import AltimetricFrame, PlanimetricFrame, ProjectionChange
        self.reframeObj =  Reframe()
        self.proj = ProjectionChange
        self.default_v_srs = {"lv03":"ln02",
                         "lv95":"lhn95",
                         "wgs84":"wgs84"
        }
        self.vframes = {
            "ln02":AltimetricFrame.LN02,
            "lhn95":AltimetricFrame.LHN95,
            "bessel":AltimetricFrame.Ellipsoid,
            "wgs84":AltimetricFrame.Ellipsoid # we have bessel altitude before/after transformation from/to wgs84
        }
        self.hframes = {
            "lv03":PlanimetricFrame.LV03_Military,
            "lv95":PlanimetricFrame.LV95,
            "wgs84":PlanimetricFrame.LV95 # we have LV95 coordinates before/after transformation from/to wgs84
        }
        self.default_raster_transform_res = 10
    
    def check_transform_args(self,s_h_srs,s_v_srs,t_h_srs,t_v_srs):
        if s_h_srs is None:
            raise Exception("Source horizontal srs must be provided using -s_h_srs option [script] or in the function call")
        if s_v_srs is None:
            s_v_srs = self.default_v_srs[s_h_srs]
        if t_h_srs is None:
            raise Exception("Target horizontal srs must be provided using -t_h_srs option [script] or in the function call")
        if t_v_srs is None:
            t_v_srs = self.default_v_srs[t_h_srs]
        if s_h_srs == t_h_srs and s_v_srs == t_v_srs:
            raise Exception("Source and target srs are exactly the same, doing nothing")
        if (s_h_srs != 'wgs84' and s_v_srs == 'wgs84') or (t_h_srs != 'wgs84' and t_v_srs == 'wgs84'):
            print("Not recommended to use wgs84 height with swiss system planimetry")

        return s_h_srs,s_v_srs,t_h_srs,t_v_srs

    # Note: wgs84 always in lon,lat format (not lat,lon !)
    def transform(self,coord,s_h_srs,s_v_srs,t_h_srs,t_v_srs):
        if type(coord) is np.ndarray and coord.ndim > 1:
            for j,elem in enumerate(coord):
                coord[j,:] = self.transform(elem,s_h_srs,s_v_srs,t_h_srs,t_v_srs)
            return coord
        if s_h_srs == 'wgs84':
            if s_v_srs == 'wgs84':
                if t_h_srs == 'wgs84':
                    if t_v_srs != 'wgs84':
                        lv95bessel = self.reframeObj.ComputeGpsref(coord,self.proj.ETRF93GeographicToLV95)
                        if t_v_srs == 'bessel':
                            coord[2] = lv95bessel[2]
                        else:
                            coord[2] = self.reframeObj.ComputeReframe(lv95bessel,self.hframes['lv95'],self.hframes['lv95'],self.vframes['bessel'],self.vframes[t_v_srs])[2]
                else:
                    if t_v_srs == 'wgs84':
                        lv95bessel = self.reframeObj.ComputeGpsref(coord,self.proj.ETRF93GeographicToLV95)
                        if t_h_srs == 'lv95':
                            coord[0:2] = lv95bessel[0:2]
                        else:
                            coord[0:2] = self.reframeObj.ComputeReframe(lv95bessel,self.hframes['lv95'],self.hframes[t_h_srs],self.vframes['bessel'],self.vframes['bessel'])[0:2]
                    else:
                        lv95bessel = self.reframeObj.ComputeGpsref(coord,self.proj.ETRF93GeographicToLV95)
                        if t_h_srs == 'lv95' and t_v_srs == 'bessel':
                            coord = lv95bessel
                        else:
                            coord = self.reframeObj.ComputeReframe(lv95bessel,self.hframes['lv95'],self.hframes[t_h_srs],self.vframes['bessel'],self.vframes[t_v_srs])

            else: # degenerated --> approximations
                if t_h_srs == 'wgs84':
                    if t_v_srs == 'wgs84':
                        xyLV95 = self.reframeObj.ComputeGpsref(coord,self.proj.ETRF93GeographicToLV95) #approx
                        if s_v_srs == 'bessel':
                            xyLV95 = [xyLV95[0],xyLV95[1],coord[2]]
                        else:
                            xyLV95 = self.reframeObj.ComputeReframe([xyLV95[0],xyLV95[1],coord[2]],self.hframes['lv95'],self.hframes['lv95'],self.vframes[s_v_srs],self.vframes['bessel'])
                        coord[2] = self.reframeObj.ComputeGpsref(coord,self.proj.LV95ToETRF93Geographic)[2]
                    else:
                        if s_v_srs != t_v_srs:
                            xyLV95 = self.reframeObj.ComputeGpsref(coord,self.proj.ETRF93GeographicToLV95) #approx
                            xyLV95 = self.reframeObj.ComputeReframe([xyLV95[0],xyLV95[1],coord[2]],self.hframes['lv95'],self.hframes['lv95'],self.vframes[s_v_srs],self.vframes[t_v_srs])
                            coord[2] = xyLV95[2]
                else:
                    if t_v_srs == 'wgs84':
                        # source vertical is not wgs, destination horizontal is not wgs
                        xyLV95 = self.reframeObj.ComputeGpsref(coord,self.proj.ETRF93GeographicToLV95) #approx
                        if s_v_srs == 'bessel':
                            xyLV95 = [xyLV95[0],xyLV95[1],coord[2]]
                        else:
                            xyLV95 = self.reframeObj.ComputeReframe([xyLV95[0],xyLV95[1],coord[2]],self.hframes['lv95'],self.hframes['lv95'],self.vframes[s_v_srs],self.vframes['bessel'])
                        coord[2] = self.reframeObj.ComputeGpsref(xyLV95,self.proj.LV95ToETRF93Geographic)[2]
                        if t_h_srs != 'lv95':
                            coord[0:2] = self.reframeObj.ComputeReframe(xyLV95,self.hframes['lv95'],self.hframes[t_h_srs],self.vframes['bessel'],self.vframes['bessel'])[0:2]
                    else:
                        xyLV95 = self.reframeObj.ComputeGpsref(coord,self.proj.ETRF93GeographicToLV95) #approx
                        if t_h_srs == 'lv95' and t_v_srs == s_v_srs:
                            coord = [xyLV95[0],xyLV95[1],coord[2]]
                        else:
                            coord = self.reframeObj.ComputeReframe([xyLV95[0],xyLV95[1],coord[2]],self.hframes['lv95'],self.hframes[t_h_srs],self.vframes[s_v_srs],self.vframes[t_v_srs])
        else: # degenerated
            if s_v_srs == 'wgs84':
                if t_h_srs == 'wgs84':
                    if t_v_srs == 'wgs84':
                        if s_h_srs != 'lv95':
                            coord = self.reframeObj.ComputeReframe(coord,self.hframes[s_h_srs],self.hframes['lv95'],self.vframes['bessel'],self.vframes['bessel'])
                        coord[0:2] = self.reframeObj.ComputeGpsref(coord,self.proj.LV95ToETRF93Geographic)[0:2] #approx
                    else:
                        if s_h_srs != 'lv95':
                            coord = self.reframeObj.ComputeReframe(coord,self.hframes[s_h_srs],self.hframes['lv95'],self.vframes['bessel'],self.vframes['bessel'])
                        wgs84 = self.reframeObj.ComputeGpsref(coord,self.proj.LV95ToETRF93Geographic)[0:2] #approx
                        lv95bessel = self.reframeObj.ComputeGpsref([wgs84[0],wgs84[1],coord[2]],self.proj.ETRF93GeographicToLV95)
                        if t_v_srs == 'bessel':
                            h = lv95bessel[2]
                        else:
                            h = self.reframeObj.ComputeReframe(lv95bessel,self.hframes['lv95'],self.hframes['lv95'],self.vframes['bessel'],self.vframes[t_v_srs])[2]
                        coord = [wgs84[0],wgs84[1],h]
                else:
                    if t_v_srs == 'wgs84':
                        # horizontal: swiss to swiss, vertical: wgs to wgs
                        coord[0:2] = self.reframeObj.ComputeReframe(coord,self.hframes[s_h_srs],self.hframes[t_h_srs],self.vframes['lhn95'],self.vframes['lhn95'])[0:2]
                    else:
                        # horizontal: swiss to swiss, vertical: wgs to swiss
                        if s_h_srs != 'lv95':
                            lv95 = self.reframeObj.ComputeReframe(coord,self.hframes[s_h_srs],self.hframes['lv95'],self.vframes['bessel'],self.vframes['bessel'])
                        else:
                            lv95 = coord
                        wgs84 = self.reframeObj.ComputeGpsref(lv95,self.proj.LV95ToETRF93Geographic)[0:2] #approx
                        h = self.reframeObj.ComputeGpsref([wgs84[0],wgs84[1],coord[2]],self.proj.ETRF93GeographicToLV95)[2]
                        if t_v_srs == 'bessel' and t_h_srs == s_h_srs:
                            coord[2] = h
                        elif t_v_srs == 'bessel' and t_h_srs == 'lv95':
                            coord = [lv95[0],lv95[1],h]
                        else:
                            coord = self.reframeObj.ComputeReframe([coord[0],coord[1],h],self.hframes[s_h_srs],self.hframes[t_h_srs],self.vframes['bessel'],self.vframes[t_v_srs])
            else: # source = swiss system
                if t_h_srs == 'wgs84':
                    if t_v_srs == 'wgs84':
                        if s_h_srs == 'lv95' and s_v_srs == 'bessel':
                            lv95bessel = coord
                        else:
                            lv95bessel = self.reframeObj.ComputeReframe(coord,self.hframes[s_h_srs],self.hframes['lv95'],self.vframes[s_v_srs],self.vframes['bessel'])
                        coord = self.reframeObj.ComputeGpsref(lv95bessel,self.proj.LV95ToETRF93Geographic)
                    else:
                        if s_h_srs == 'lv95' and s_v_srs == 'bessel':
                            lv95bessel = coord
                        else:
                            lv95bessel = self.reframeObj.ComputeReframe(coord,self.hframes[s_h_srs],self.hframes['lv95'],self.vframes[s_v_srs],self.vframes['bessel'])
                        if t_v_srs == 'bessel':
                            coord[2] = lv95bessel[2]
                        elif t_v_srs != s_v_srs:
                            coord[2] = self.reframeObj.ComputeReframe(coord,self.hframes[s_h_srs],self.hframes[s_h_srs],self.vframes[s_v_srs],self.vframes[t_v_srs])[2]
                        coord[0:2] = self.reframeObj.ComputeGpsref(lv95bessel,self.proj.LV95ToETRF93Geographic)[0:2]
                else:
                    if t_v_srs == 'wgs84':
                        if s_h_srs == 'lv95' and s_v_srs == 'bessel':
                            lv95bessel = coord
                        else:
                            lv95bessel = self.reframeObj.ComputeReframe(coord,self.hframes[s_h_srs],self.hframes['lv95'],self.vframes[s_v_srs],self.vframes['bessel'])
                        if t_h_srs == 'lv95':
                            coord[0:2] = lv95bessel[0:2]
                        elif t_h_srs != s_h_srs:
                            coord[0:2] = self.reframeObj.ComputeReframe(coord,self.hframes[s_h_srs],self.hframes[t_h_srs],self.vframes['lhn95'],self.vframes['lhn95'])[0:2]
                        coord[2] = self.reframeObj.ComputeGpsref(lv95bessel,self.proj.LV95ToETRF93Geographic)[2]
                    else:
                        if t_h_srs != s_h_srs or t_v_srs != s_v_srs:
                            coord = self.reframeObj.ComputeReframe(coord,self.hframes[s_h_srs],self.hframes[t_h_srs],self.vframes[s_v_srs],self.vframes[t_v_srs])
        return coord

    
    def transform_txt(self,input_file,output_file,s_h_srs,s_v_srs,t_h_srs,t_v_srs,sep=',',skip=0):
        if input_file is None:
            raise Exception("Input file name must be provided")
        if output_file is None:
            raise Exception("Output file name must be provided")
        s_h_srs,s_v_srs,t_h_srs,t_v_srs = self.check_transform_args(s_h_srs,s_v_srs,t_h_srs,t_v_srs)
        resolution = 8 if t_h_srs == 'wgs84' else 2
        outstring = "{0:." + str(resolution) + "f}" + sep + "{1:." + str(resolution) + "f}" + sep + "{2:.2f}" + "{3}"

        total_file_size = os.path.getsize(input_file)
        size_read = 0
        progression = 0
        print("This might take some time... Progression:")
        bar = progressbar.ProgressBar(maxval=1000, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        with open(input_file,'r') as f_in, open(output_file,'w') as f_out:
            for _ in range(skip+1):
                line = f_in.readline()
            while line:
                size_read += len(line)
                if int(1000*size_read/total_file_size) > progression:
                    progression+=1
                    bar.update(progression)
                line = line.split(sep)
                coordinates = [float(line[0]),float(line[1]),float(line[2])]
                userdata = line[3:]
                userdata = sep + sep.join(userdata) if userdata else '\n'
                coordinates = self.transform(coordinates,s_h_srs,s_v_srs,t_h_srs,t_v_srs)
                f_out.write(outstring.format(coordinates[0],coordinates[1],coordinates[2],userdata))
                line = f_in.readline()
        bar.finish()

    def transform_las(self,input_file,output_file,s_h_srs,s_v_srs,t_h_srs,t_v_srs):
        if input_file is None:
            raise Exception("Input file name must be provided")
        if output_file is None:
            raise Exception("Output file name must be provided")
        shutil.copyfile(input_file,output_file)
        s_h_srs,s_v_srs,t_h_srs,t_v_srs = self.check_transform_args(s_h_srs,s_v_srs,t_h_srs,t_v_srs)
        f = laspy.file.File(output_file,mode="rw")
        l = len(f)
        progression = 0
        print("This might take some time... Progression:")
        bar = progressbar.ProgressBar(maxval=1000, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        scale = f.header.scale
        newscale = [0.0000001,0.0000001,0.01] if t_h_srs == 'wgs84' else [0.01,0.01,0.01]
        offset = f.header.offset
        newoffset = [0,0,0]
        for j in range(l):
            p = f[j]
            if int(1000*j/l) > progression:
                progression+=1
                bar.update(progression)
            x = p.X*scale[0]+offset[0]
            y = p.Y*scale[1]+offset[1]
            z = p.Z*scale[2]+offset[2]
            coordinates = self.transform([x,y,z],s_h_srs,s_v_srs,t_h_srs,t_v_srs)
            f.X[j] = (coordinates[0]-newoffset[0])/newscale[0]
            f.Y[j] = (coordinates[1]-newoffset[1])/newscale[1]
            f.Z[j] = (coordinates[2]-newoffset[2])/newscale[2]
        bar.finish()
        f.header.scale = newscale
        f.header.offset = newoffset
        f.close()

    # Raster tranformation: currently only vertical transform
    def transform_raster(self,input_file,output_file,s_h_srs,s_v_srs,t_v_srs,transform_res=None,nointerp=False):
        # transform_res: Don't do the computation for every points but compute height difference for a few points according to this parameter and interpolate
        # nointerp: Set this to true to explicitely ask to compute for all points (overrides transform_res)
        if input_file is None:
            raise Exception("Input file name must be provided")
        if output_file is None:
            raise Exception("Output file name must be provided")
        if transform_res is None:
            transform_res = self.default_raster_transform_res
        shutil.copyfile(input_file,output_file)
        t_h_srs = s_h_srs
        s_h_srs,s_v_srs,t_h_srs,t_v_srs = self.check_transform_args(s_h_srs,s_v_srs,t_h_srs,t_v_srs)
        with rasterio.open(input_file) as src:
            height = src.read()[0]
            # Create sampling vectors, make sure last values are included
            if nointerp:
                X = np.arange(0,height.shape[0])
                Y = np.arange(0,height.shape[1])
                newheight = np.zeros((X.size,Y.size))
            else:
                dx = round(transform_res/src.res[0])
                X = np.arange(0,height.shape[0],dx)
                if (X[-1] != height.shape[0]-1):
                    X = np.append(X,height.shape[0]-1)
                dy = round(transform_res/src.res[1])
                Y = np.arange(0,height.shape[1],dy)
                if (Y[-1] != height.shape[1]-1):
                    Y = np.append(Y,height.shape[1]-1)
                diff = np.zeros((X.size,Y.size))
            print("Computing conversion")
            bar = progressbar.ProgressBar(maxval=len(X), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            for ind_j,j in enumerate(X):
                for ind_k,k in enumerate(Y):
                    (xcoor,ycoor) = src.xy(j,k)
                    if nointerp:
                        newheight[ind_j,ind_k] = self.transform([xcoor,ycoor,height[j,k]],s_h_srs,s_v_srs,t_h_srs,t_v_srs)[2]
                    else:
                        diff[ind_j,ind_k] = self.transform([xcoor,ycoor,height[j,k]],s_h_srs,s_v_srs,t_h_srs,t_v_srs)[2] - height[j,k]
                bar.update(ind_j)
            bar.finish()
            #print("Mean of difference: ", diff.mean())
            #print("Std of difference: ", diff.std())
            print('Interpolation and writing')
            if not nointerp:
                f = interpolate.interp2d(Y,X,diff) # linear interpolation
                X_new = np.arange(0,height.shape[0])
                Y_new = np.arange(0,height.shape[1])
                newheight = f(Y_new,X_new) + height

            profile = src.profile
            with rasterio.open(output_file,'w',**profile) as trg:
                trg.write(newheight.astype(rasterio.float32), indexes = 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str,
                        help="Input file name")
    parser.add_argument("output_file", type=str, default="out.txt",
                        help="Output file name")
    parser.add_argument("-s_h_srs", type=str, choices=["lv03","lv95","wgs84"],
                        help="Set source horizontal srs")
    parser.add_argument("-s_v_srs", type=str, choices=["ln02","lhn95","bessel","wgs84"],
                        help="Set source vertical srs (default is assumed from horizontal sourve srs as: lv03 -> ln02 / lv95 -> lhn95 / wgs84 -> wgs84)")
    parser.add_argument("-t_h_srs", type=str, choices=["lv03","lv95","wgs84"],
                        help="Set target horizontal srs")
    parser.add_argument("-t_v_srs", type=str, choices=["ln02","lhn95","bessel","wgs84"],
                        help="Set target vertical srs (default is assumed from horizontal target srs as: lv03 -> ln02 / lv95 -> lhn95 / wgs84 -> wgs84)")
    parser.add_argument("-sep", type=str, default=',', help="For text points clouds: Select separator [default: ',']")
    parser.add_argument("-skip", type=int, default=0, help="For text points clouds: Skip n first lines of the file (for txt) [default:0]")
    parser.add_argument("-transform_res", type=float, help="For raster height transformation: Don't do the computation for every points but compute height difference for a few points according to this parameter and interpolate [default: 10 m]")
    parser.add_argument("-nointerp",action="store_true", default=False, help="Use this to explicitely ask to compute for all points (overrides transform_res)")

    args = parser.parse_args()
    r = ReframeTransform()

    if args.input_file.endswith('las'):
        if not args.output_file.endswith('las'):
            raise Exception("If input_file is a 'las', output_file has to be a 'las' too. Conversion not supported")
        r.transform_las(args.input_file,args.output_file,args.s_h_srs,args.s_v_srs,args.t_h_srs,args.t_v_srs)
    elif args.input_file.endswith('laz'):
        raise Exception("Laz files not supported, please uncompress")
    elif args.input_file.endswith('tif'):
        r.transform_raster(args.input_file,args.output_file,args.s_h_srs,args.s_v_srs,args.t_v_srs,args.transform_res,args.nointerp)
    else:
        r.transform_txt(args.input_file,args.output_file,args.s_h_srs,args.s_v_srs,args.t_h_srs,args.t_v_srs,args.sep,args.skip)
