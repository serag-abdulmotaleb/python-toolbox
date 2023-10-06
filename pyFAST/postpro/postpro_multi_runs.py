import os,glob,re
import numpy as np
import pandas as pd
import scipy.signal as sg
from pyFAST.input_output import FASTInputFile as fstin
from pyFAST.input_output import FASTOutputFile as fstout

def bw_filter(data, cutoff, fs, order, switch):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = sg.butter(order, normal_cutoff, btype=switch, analog=False)
    y = sg.filtfilt(b, a, data)
    return y

def out2pkl(out_files,keys='all'):
    for out_file in out_files:
        df = fstout(out_file).toDataFrame()
        if keys == 'all':
            keys = df.keys()
            df.pop('Time_[s]')
        df[keys].to_pickle('.'.join(out_file.split('.')[:-1]) + '.pkl')

def find_env_cond(fst_file):
    fst = fstin(fst_file)
    fst_dir = os.path.dirname(os.path.abspath(fst_file))
 
    if fst['CompInflow'] > 0:
        inflow_file = fst['InflowFile'].strip('"')
        inflow = fstin(os.path.join(fst_dir,inflow_file))
        if inflow['WindType'] == 1:
            Uw = inflow['HWindSpeed']
            TI = 0.
        elif inflow['WindType'] == 3:
            inflow_dir = os.path.dirname(os.path.abspath(os.path.join(fst_dir,inflow_file)))
            bts_file = inflow['FileName_BTS'].strip('"')
            inp_file = os.path.join(fst_dir,inflow_dir,bts_file)
            inp_file = inp_file.strip('.bts') + '.inp'
            inp = fstin(inp_file)
            Uw = inp['URef']
            TI = inp['IECturbc']
        else:
            raise ValueError(f'WindType {inflow["WindType"]} is not supported by this function.')
    else:
        Uw = 0
        TI = 0
    
    if fst['CompHydro']>0:
        hydro_file = fst['HydroFile'].strip('"')
        hydro = fstin(os.path.join(fst_dir,hydro_file))
        if hydro['WaveMod'] == 2:
            Hs = hydro['WaveHs']
            Tp = hydro['WaveTp']
            gamma = hydro['WavePkShp']
        elif hydro['WaveMod'] == 1:
            Hs = hydro['WaveHs']
            Tp = hydro['WaveTp']
            gamma = 0
        elif hydro['WaveMod'] == 0:
            Hs = 0
            Tp = 0
            gamma = 0
        else:
            raise ValueError(f'WaveMod {hydro["WaveMod"]} is not supported by this function.')
    else:
        Hs = 0
        Tp = 0
        gamma = 0

    return Uw,TI,Hs,Tp,gamma

def gen_conditions_df(out_files):
    df_conds = pd.DataFrame(columns=['condition','Uw','TI','Hs','Tp','gamma','seeds','files'])
    df_conds['files'].astype('object')
    condition = 0
    for out_file in out_files:
        fst_file = out_file.replace('.out','.fst')
        fst_file = out_file.replace('.outb','.fst')
        Uw,TI,Hs,Tp,gamma = find_env_cond(fst_file)
        row_bool = ((df_conds['Uw']==Uw) & (df_conds['TI'] == TI) & (df_conds['Hs'] == Hs) & (df_conds['Tp'] == Tp) & (df_conds['gamma'] == gamma))
        
        if row_bool.any():
            row_idx =df_conds.loc[row_bool, 'files'].index[0]
            df_conds.loc[row_idx,'seeds'] += 1
            df_conds.at[row_idx,'files'] += [out_file]

        else:
            condition += 1
            seeds = 1
            df_conds.loc[df_conds.shape[0]] = [condition,Uw,TI,Hs,Tp,gamma,seeds,[out_file]]
    return df_conds

def extract_key_series(out_files,keys,prefix=''):
    """
    Reads out/outb files and generates pkl files which contains the time series of selected keys in multiple conditions.
    Each column of the pkl file corresponds to a condition and the condtions are summarized and numbered in a csv file.

    Parameters
    ----------
    out_files : list
        List of out/outb files.
    keys : list
        OpenFAST keys to be read.


    """
    df_conds = gen_conditions_df(out_files)
    out_dfs = [pd.DataFrame() for _ in range(len(keys))]

    for row in range(df_conds.shape[0]):
        row_df = df_conds.iloc[row,:]
        condition = row_df['condition']
        for seed,out_file in enumerate(row_df['files']):
            seed += 1
            df = fstout(out_file).toDataFrame()
            for out_df,key in zip(out_dfs,keys):
                out_df['Time_[s]'] = df['Time_[s]']
                out_df[f'c{condition}s{seed}'] = df[key]

    df_conds.to_csv(prefix + 'conditions.csv')        
    for out_df,key in zip(out_dfs,keys):
        out_df.to_pickle(prefix + key.replace('/s','s-1')+'.pkl')

def filter_key_series(pkl_file,lpf=None,hpf=None):
    """
    Filters time series in a pkl file and writes pkl file with filtered columns.
    
    Parameters
    ----------
    pkl_file : str
        pkl file.
    lpf : float, optional
        Low pass filter frequency. The default is None.
    hpf : float, optional
        High pass filter frequency. The default is None.


    """
    df = pd.read_pickle(pkl_file)
    t = df['Time_[s]'].to_numpy()
    dt = t[1] - t[0]
    fs = 1/dt

    for col in df.columns:
        if lpf:
            x = df[col]
            df[col] = bw_filter(data=x,cutoff=lpf,fs=fs,order=2,switch='low')
        
        if hpf:
            x = df[col]
            df[col] = bw_filter(data=x,cutoff=hpf,fs=fs,order=2,switch='high')
    
        if lpf:
            lf = f'_LPF{lpf}Hz'
        else:
            lf = ''
        
        if hpf:
            hf = f'_HPF{hpf}Hz'
        else:
            hf = ''

        df.to_pickle(pkl_file.strip('.pkl') + lf + hf + '.pkl')


def process_key_series(xpkl_file,T_seg,T_trans=0.,ypkl_file=None,prefix=''):
    """
    Generates stats and PSDs for columns in a pkl file. Can also generate CSDs if another pkl file is provided.

    Parameters
    ----------
    xpkl_file : str
        pkl file that has the series for which the stats and PSDs to be generated.
    T_seg : float
        Window length for Welch's method.
    T_trans : float, optional
        Transient time to drop at the begining of the series. The default is 0..
    ypkl_file : str, optional
        pkl file that has the series to generate CSDs. The default is None.

    """
    df_x = pd.read_pickle(xpkl_file)
    df_x = df_x[df_x['Time_[s]']>=T_trans]

    t = df_x['Time_[s]'].to_numpy()
    T_sim = t[-1] - T_trans
    dt = t[1] - t[0]
    fs = 1/dt
    n_conds = max([int(re.findall(r'c\d+',col)[0].split('c')[1]) for col in df_x.columns[1:]]) #number of conditions
    stats_df = pd.DataFrame(columns = ['condition','mean','std_dev','max'])
    spectra_df = pd.DataFrame()

    if ypkl_file:
        df_y = pd.read_pickle(ypkl_file)
        df_y = df_y[df_y['Time_[s]']>=T_trans]
        crspectra_df = pd.DataFrame()

    for c in range(1,n_conds+1):
        dfc_x = df_x[[col for col in df_x.columns[1:] if int(re.findall(r'c\d+',col)[0].split('c')[1]) == c]]
        PSD = []
        if ypkl_file:
            dfc_y = df_y[[col for col in df_y.columns[1:] if int(re.findall(r'c\d+',col)[0].split('c')[1]) == c]]
            CSD=[]

        for col in dfc_x.columns:
            x = dfc_x[col].to_numpy()
            (f,Sxx) = sg.welch(x,fs,nperseg=int(T_seg/dt))
            PSD.append(Sxx)
            if ypkl_file:
                y = dfc_y[col].to_numpy()
                (f,Sxy) = sg.csd(x,y,fs,nperseg=int(T_seg/dt))
                CSD.append(Sxy)
        
        x_tot = dfc_x.stack().to_numpy()
        stats_df.loc[stats_df.shape[0],:] = [c,x_tot.mean(), x_tot.std(), np.abs(x_tot).max()]
        spectra_df['Freq_[Hz]'] = f
        spectra_df[f'c{c}'] = np.mean(np.array(PSD),axis=0)

        stats_df.to_csv(prefix + 'stats_' + xpkl_file.strip('.pkl') + '.csv')
        spectra_df.to_pickle(prefix + 'psd_' + xpkl_file.strip('.pkl') + '.pkl')
        if ypkl_file:
            crspectra_df['Freq_[Hz]'] = f
            CSD = np.array(CSD)
            crspectra_df[f'c{c}'] = np.mean(np.abs(CSD),axis=0)*np.exp(1j*np.mean(np.angle(CSD),axis=0))
            crspectra_df.to_pickle(prefix + 'csd_' + xpkl_file.strip('.pkl') + '_' + ypkl_file.strip('.pkl') + '.pkl')