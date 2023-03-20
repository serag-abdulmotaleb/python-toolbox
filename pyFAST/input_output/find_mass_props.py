# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:08:03 2023

@author: seragela
"""
import os
import numpy as np
import pandas as pd
from pyFAST.input_output.fast_input_file import FASTInputFile as fstin

def find_mass_props(fst_file,ref2rt = np.zeros([0,0,0]), azimuth = 0.):
    """
    Evaluates mass distribution of an OpenFAST model about the intersection of the tower with the MSL.
    (assumes all mass is modeled in ElastoDyn)

    Parameters
    ----------
    fst_file : str
        .fst file loctaion.

    Returns
    -------
    m : float
        Total mass in kg.
    r_cg : array
        Center of mass.
    MOI : array
        Moments of inertia around reference point.
    I_rot : float
        Rotor inertia about shaft.
    df : dataframe
        A dataframe that summarizes mass distribution.

    """
    root_folder = os.path.dirname(fst_file)
    fst = fstin(fst_file)
    ED_file = os.path.join(root_folder,fst['EDFile'].strip('"').strip("'"))
    ED = fstin(ED_file)
    
    # Platfrom
    PtfmMass = ED['PtfmMass']
    PtfmIner = np.array([ED[f'Ptfm{i}Iner'] for i in ['R','P','Y']])
    PtfmCM = np.array([ED[f'PtfmCM{i}t'] for i in ['x','y','z']])
    PtfmRefzt = ED['PtfmRefzt']
    
    # RNA
    NacMass = ED['NacMass']
    NacYIner = ED['NacYIner']
    YawBrMass = ED['YawBrMass']
    NacCM = np.array([ED[f'NacCM{i}n'] for i in ['x','y','z']])
    OverHang = ED['OverHang']
    NumBl = ED['NumBl']
    TipRad = ED['TipRad']
    HubRad = ED['HubRad']
    PreCone = np.array([ED[f'PreCone({i+1})'] for i in range(NumBl)])
    HubCM = ED['HubCM']
    ShftTilt = ED['ShftTilt']
    Twr2Shft = ED['Twr2Shft']
    TipMass = np.array([ED[f'TipMass({i+1})'] for i in range(NumBl)])
    HubMass = ED['HubMass']
    HubIner = ED['HubIner']
    GenIner = ED['GenIner']
    BldNodes = ED['BldNodes']
    BldFiles = [os.path.join(root_folder,ED[f'BldFile{i+1}'].strip('"').strip("'")) for i in range(NumBl)]
    BldMasses = np.zeros(NumBl)    
    BldCMs = np.zeros(NumBl)
    BldIners = np.zeros(NumBl)
    
    for b,BldFile in enumerate(BldFiles):
        Bld = fstin(BldFile)
        BldProp = Bld['BldProp']
        BldFract = BldProp[:,0]
        BldSecR = BldFract * (TipRad-HubRad)
        BMassDen = BldProp[:,3]
        BldMasses[b] = np.trapz(BMassDen,BldSecR)
        BldCMs[b] = np.trapz(BMassDen*BldSecR,BldSecR)/BldMasses[b]
        BldIners[b] = np.trapz(BMassDen*((BldSecR+HubRad)*np.cos(PreCone[b]*np.pi/180))**2,BldSecR)

    I_rot = np.sum(BldIners) + HubIner

    # Tower (Tower props relative to tower base)
    TowerHt = ED['TowerHt']
    TowerBsHt = ED['TowerBsHt']
    TwrNodes = ED['TwrNodes']
    TwrFile = os.path.join(root_folder,ED['TwrFile'].strip('"').strip("'"))
    Twr = fstin(TwrFile)
    TowProp = Twr['TowProp']
    HtFract = TowProp[:,0]
    TwrSecHt = HtFract*(TowerHt-TowerBsHt)
    TMassDen = TowProp[:,1]
    TwrMass = np.trapz(TMassDen,TwrSecHt)
    TwrCMz = np.trapz(TMassDen*TwrSecHt,TwrSecHt)/TwrMass + TowerBsHt
    TwrIner = np.zeros(3)
    TwrIner[0] = np.trapz(TMassDen*(TwrSecHt-TwrCMz)**2,TwrSecHt)
    TwrIner[1] = TwrIner[0]
    
    # Evaluate global mass props about tower base coords (t)
    moi_mat = np.ones([3,3]) - np.eye(3) 
    
    rn = np.array([0.,0.,TowerHt])
    rh = rn + np.array([OverHang*np.cos(ShftTilt*np.pi/180), 0., Twr2Shft + OverHang*np.sin(ShftTilt*np.pi/180) ])
    
    m_ptfm = PtfmMass
    r_ptfm = PtfmCM
    I_ptfm = PtfmIner + np.matmul(moi_mat,PtfmMass*r_ptfm**2)
    
    m_rna = NacMass + np.sum(BldMasses) + YawBrMass + HubMass
    r_rna = (NacMass*(NacCM+rn) + (HubMass + np.sum(BldMasses))*rh + YawBrMass*rn)/ m_rna
    I_rna = np.array([0.,0.,NacYIner]) + NacMass*np.matmul(moi_mat,(NacCM+rn)**2) + \
            (HubMass + np.sum(BldMasses))*np.matmul(moi_mat,rh**2) +\
            YawBrMass*np.matmul(moi_mat,rn**2)
    
    m_twr = TwrMass
    r_twr = np.array([0.,0.,TwrCMz])
    I_twr = TwrIner + m_twr * np.matmul(moi_mat,r_twr**2)
    
    m = m_ptfm + m_rna + m_twr
    r_cg = (m_ptfm*r_ptfm + m_rna*r_rna + m_twr*r_twr)/m
    MOI = I_ptfm + I_rna + I_twr
    
    cols = ['Mass','x','y','z','Ixx','Iyy','Izz']
    idxs = ['Platform','Tower','RNA','Total']
    data = np.array([[m_ptfm,m_twr,m_rna,m],
                     [r_ptfm[0],r_twr[0],r_rna[0],r_cg[0]],
                     [r_ptfm[1],r_twr[1],r_rna[1],r_cg[1]],
                     [r_ptfm[2],r_twr[2],r_rna[2],r_cg[2]],
                     [I_ptfm[0],I_twr[0],I_rna[0],MOI[0]],
                     [I_ptfm[1],I_twr[1],I_rna[1],MOI[1]],
                     [I_ptfm[2],I_twr[2],I_rna[2],MOI[2]]])
    
    df = pd.DataFrame(data = data.T, columns = cols, index = idxs)
                      
    return m,r_cg,MOI,I_rot,df
    
    

if __name__ == '__main__':
    fst_file = r'openfast_models\15MW_UMaineVolturnUS-S\IEA-15-240-RWT-UMaineSemi\IEA-15-240-RWT-UMaineSemi.fst'
    m,r_cg,MOI,I_rot,df_sum = find_mass_props(fst_file)
    