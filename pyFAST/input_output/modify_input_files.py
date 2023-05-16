import os,glob,re
import numpy as np
import pandas as pd
from pyFAST.input_output import FASTInputFile as fstin

def modify_input_file(base_file,out_file,inp_dict):
    """
    Writes a new OpenFAST input files based on a basis file (in the required 
    version of OpenFAST) and a dictionary in which the keys correspond to the 
    variable names required to be changed in the input files and their desired 
    values. 
    (for the most recent version of OpenFAST check the regression test repository)

    Parameters
    ----------
    base_file : str
        Name of the base file. It can be a .fst file, .dat file for module input files (except MoorDyn), or a .inp file for TurbSim input file.
    out_file : TYPE
        Name of the output file to be written
    inp_dict : TYPE
        A dictionary that contains the keys the are to be changed in the base files and their corresponding values

    """
    f = fstin(base_file)
    KDAdd_dict = {}  #added to handle hydrodyn's additional matrices
    for key in inp_dict.keys():
        if key in f.keys():
            f[key] = inp_dict[key]
        elif (key in ['AddF0','AddCLin','AddBLin','AddBQuad']) and ('WaveMod' in f.keys()): #added to handle hydrodyn's additional matrices
            KDAdd_dict[key] = inp_dict[key]
        else:
            raise ValueError(f'Error: Key({key}) is not in the {base_file} input file.')
        f.write(out_file)
        
        if KDAdd_dict: #added to handle hydrodyn's additional matrices
            write_hydrodyn_KDAdd(out_file,KDAdd_dict=KDAdd_dict,first_row=68)
        

def modify_fst_deck(fst_dir,fst_file,out_dir='.',suffix='',env_suffix='',wave_suffix='',wind_suffix='',dof_suffix='',
                    fst_dict={},aero_dict={},hydro_dict={},elasto_dict={},inflow_dict={},servo_dict={}):
    """
    Reads a working OpenFAST model input files, modifies the desired files and writes them in the desired directory.    

    Parameters
    ----------
    fst_dir : str
        Directory of the base OpenFAST model.
    fst_file : str
        .fst file name of the base OpenFAST model.
    out_dir : str, optional
        Desired directory of modified OpenFAST input files. The default is the current working directory'.'.
    suffix : str, optional
        General case suffix to be used in all modified input files. The default is ''.
    cond_suffix : str, optional
        Environmental condition suffix to be used in modified .fst files as well as inflow and/or hydrodyn files . The default is ''.
    dof_suffix : str, optional
        DOFs suffix to be used in modified .fst files as well as Elastodyn files . The default is ''.
    fst_dict : dict, optional
        Dictionary of .fst keys to be changed. The default is {}.
    aero_dict : dict, optional
        Dictionary of aerodyn keys to be changed. The default is {}.
    hydro_dict : dict, optional
        Dictionary of hydrodyn keys to be changed.. The default is {}.
    elasto_dict : dict, optional
        Dictionary of elastodyn keys to be changed.. The default is {}.
    inflow_dict : dict, optional
        Dictionary of inflow keys to be changed.. The default is {}.
    servo_dict : dict, optional
        Dictionary of servodyn keys to be changed.. The default is {}.

    """
    
    # create output directory if it doesn't already exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # read .fst file and get the root name and out .fst file name
    fst = fstin(os.path.join(fst_dir,fst_file))
    fst_root = fst_file.strip('.fst')
    fst_out = os.path.join(out_dir,fst_root + suffix + dof_suffix + env_suffix + wind_suffix + wave_suffix + '.fst')
    
    # get base file names for all modules
    fst_base  = os.path.join(fst_dir,fst_file)
    aero_base = os.path.join(fst_dir,fst['AeroFile'].replace('"',''))
    hydro_base = os.path.join(fst_dir,fst['HydroFile'].replace('"',''))
    elasto_base = os.path.join(fst_dir,fst['EDFile'].replace('"',''))
    inflow_base = os.path.join(fst_dir,fst['InflowFile'].replace('"',''))
    servo_base = os.path.join(fst_dir,fst['ServoFile'].replace('"',''))
    moor_base = os.path.join(fst_dir,fst['MooringFile'].replace('"',''))
    
    aero_dir = os.path.dirname(aero_base)
    hydro_dir = os.path.dirname(hydro_base)
    elasto_dir = os.path.dirname(elasto_base)
    servo_dir = os.path.dirname(servo_base)
    inflow_dir = os.path.dirname(inflow_base)
    
    if aero_dict:
        ad = fstin(aero_base)
        aero_out = fst_root + '_AeroDyn' + suffix + '.dat'
        aero_dict['AFNames'] = ['"{}"'.format(os.path.join(os.path.relpath(aero_dir,out_dir),af.replace('"',''))) for af in ad['AFNames']]
        aero_dict['ADBlFile(1)'] = '"{}"'.format(os.path.join(os.path.relpath(aero_dir,out_dir),ad['ADBlFile(1)'].replace('"','')))
        aero_dict['ADBlFile(2)'] = '"{}"'.format(os.path.join(os.path.relpath(aero_dir,out_dir),ad['ADBlFile(2)'].replace('"','')))
        aero_dict['ADBlFile(3)'] = '"{}"'.format(os.path.join(os.path.relpath(aero_dir,out_dir),ad['ADBlFile(3)'].replace('"','')))
        modify_input_file(aero_base,os.path.join(out_dir,aero_out),aero_dict)
        fst_dict['AeroFile'] = '"{}"'.format(aero_out)
    else:
        fst_dict['AeroFile'] = '"{}"'.format(os.path.relpath(aero_base,out_dir))

    if hydro_dict:
        hd = fstin(hydro_base)
        hydro_out = fst_root + '_HydroDyn' + suffix + env_suffix + wave_suffix + '.dat'
        hydro_dict['PotFile'] = '"{}"'.format(os.path.join(os.path.relpath(hydro_dir,out_dir),hd['PotFile'].replace('"','')))
        modify_input_file(hydro_base,os.path.join(out_dir,hydro_out),hydro_dict)
        fst_dict['HydroFile'] = '"{}"'.format(hydro_out)
    else:
        fst_dict['HydroFile'] = '"{}"'.format(os.path.relpath(hydro_base,out_dir))

    if elasto_dict:
        ed = fstin(elasto_base)
        elasto_out = fst_root + '_ElastoDyn' + suffix + dof_suffix + '.dat'
        elasto_dict['BldFile1'] = '"{}"'.format(os.path.join(os.path.relpath(elasto_dir,out_dir),ed['BldFile1'].replace('"','')))
        elasto_dict['BldFile2'] = '"{}"'.format(os.path.join(os.path.relpath(elasto_dir,out_dir),ed['BldFile2'].replace('"','')))
        elasto_dict['BldFile3'] = '"{}"'.format(os.path.join(os.path.relpath(elasto_dir,out_dir),ed['BldFile3'].replace('"','')))
        elasto_dict['TwrFile'] = '"{}"'.format(os.path.join(os.path.relpath(elasto_dir,out_dir),ed['TwrFile'].replace('"','')))
        modify_input_file(elasto_base,os.path.join(out_dir,elasto_out),elasto_dict)
        fst_dict['EDFile'] = '"{}"'.format(elasto_out)
    else:
        fst_dict['EDFile'] = '"{}"'.format(os.path.relpath(elasto_base,out_dir))

    if inflow_dict:
        inflow_out = fst_root + '_Inflow' + suffix + env_suffix + wind_suffix + '.dat'
        modify_input_file(inflow_base,os.path.join(out_dir,inflow_out),inflow_dict)
        fst_dict['InflowFile'] = '"{}"'.format(inflow_out)
    else:
        fst_dict['InflowFile'] = '"{}"'.format(os.path.relpath(inflow_base,out_dir))

    if servo_dict:
        sr = fstin(servo_base)
        servo_out = fst_root + '_ServoDyn' + suffix + '.dat'
        servo_dict['DLL_FileName'] = '"{}"'.format(os.path.join(os.path.relpath(servo_dir,out_dir),sr['DLL_FileName'].replace('"','')))
        servo_dict['DLL_InFile'] = '"{}"'.format(os.path.join(os.path.relpath(servo_dir,out_dir),sr['DLL_InFile'].replace('"','')))

        modify_input_file(servo_base,os.path.join(out_dir,servo_out),servo_dict)
        fst_dict['ServoFile'] = '"{}"'.format(servo_out)
    else:
        fst_dict['ServoFile'] = '"{}"'.format(os.path.relpath(servo_base,out_dir))
    
    fst_dict['MooringFile'] = '"{}"'.format(os.path.relpath(moor_base,out_dir))
    
    modify_input_file(fst_base,fst_out,fst_dict)

def read_wind_files(bts_files):
    """
    Reads a list of .bts wind files and their corresponding .inp files and returns a dataframe with wind speeds, seed numbers, and file names.

    Parameters
    ----------
    bts_files : list
        A list of .bts file names.

    Returns
    -------
    df : pandas DataFrame
        A dataframe with columns 'Uw', 'seed', 'file.

    """
    wind_speeds = []
    seeds = []
    seed = 1
    for bts_file in bts_files:
        inp_file = bts_file.replace('.bts','.inp')
        inp_data = fstin(inp_file)
        Uw = inp_data['URef']
        if Uw in wind_speeds:
            seed += 1
            wind_speeds.append(Uw)
            seeds.append(seed)
        else:
            seed = 1
            wind_speeds.append(Uw)
            seeds.append(seed)
                
    df = pd.DataFrame({'Uw':wind_speeds,'seed':seeds,'file':bts_files})
    return df

def write_hydrodyn_KDAdd(hydrodyn_file,KDAdd_dict={},first_row=68):
    
    with open(hydrodyn_file, 'r') as f:
        lines = f.readlines()
        
    if 'AddF0' in KDAdd_dict.keys():
        AddF0 = []
        for i in range(6):
            if len(KDAdd_dict['AddF0'].shape) == 1: # added to handle NBodies
                KDAdd_dict['AddF0'] = np.expand_dims(KDAdd_dict['AddF0'],-1)
            AddF0.append('    '.join(["%0.*e"%(8,kij) for kij in KDAdd_dict['AddF0'][i]]))
        AddF0[-6] += '   AddF0    - Additional preload (N, N-m) [If NBodyMod=1, one size 6*NBody x 1 vector; if NBodyMod>1, NBody size 6 x 1 vectors]'
        for i,row in enumerate(range(first_row,first_row+6)):
            lines[row] = AddF0[i] + '\n'

    if 'AddCLin' in KDAdd_dict.keys():
        AddCLin = []
        for i in range(6):
            AddCLin.append('    '.join(["%0.*e"%(8,kij) for kij in KDAdd_dict['AddCLin'][i]]))
        AddCLin[-6] += '   AddCLin - Additional linear stiffness (N/m, N/rad, N-m/m, N-m/rad) [If NBodyMod=1, one size 6*NBody x 6*NBody matrix; if NBodyMod>1, NBody size 6 x 6 matrices]'
        for i,row in enumerate(range(first_row+6,first_row+12)):
            lines[row] = AddCLin[i] + '\n'
    
    if 'AddBLin' in KDAdd_dict.keys():
        AddBLin = []
        for i in range(6):
            AddBLin.append('    '.join(["%0.*e"%(8,kij) for kij in KDAdd_dict['AddBLin'][i]]))
        AddBLin[-6] += '   AddBLin - Additional linear damping(N/(m/s), N/(rad/s), N-m/(m/s), N-m/(rad/s)) [If NBodyMod=1, one size 6*NBody x 6*NBody matrix; if NBodyMod>1, NBody size 6 x 6 matrices]'
        for i,row in enumerate(range(first_row+12,first_row+18)):
            lines[row] = AddBLin[i] + '\n'
        
    if 'AddBQuad' in KDAdd_dict.keys():
        AddBQuad = []
        for i in range(6):
            AddBQuad.append('    '.join(["%0.*e"%(8,kij) for kij in KDAdd_dict['AddBQuad'][i]]))
        AddBQuad[-6] += '   AddBQuad - Additional quadratic drag(N/(m/s)^2, N/(rad/s)^2, N-m(m/s)^2, N-m/(rad/s)^2) [If NBodyMod=1, one size 6*NBody x 6*NBody matrix; if NBodyMod>1, NBody size 6 x 6 matrices]'
        for i,row in enumerate(range(first_row+18,first_row+24)):
            lines[row] = AddBQuad[i] + '\n'

    with open(hydrodyn_file, 'w') as f:
        f.writelines(lines)