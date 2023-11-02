# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:08:03 2023

@author: seragela
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    
def get_mass_props(fst_file,csv_out=False,plot_wt=False,elev=20,azim=-90):
    """Evaluates mass distribution of an OpenFAST model about the intersection of the tower with the MSL.
    (assumes all mass is modeled in ElastoDyn)

    Parameters
    ----------
    fst_file : str
        fst file directory
    csv_out : bool, optional
        Save mass properties summary in a csv file.  by default False.
    plot_wt : bool, optional
        Plot 3D visualization of the WT. by default False.
    elev : int, optional
        Elevation angle of 3D plot by default 20
    azim : int, optional
        Azimuth angle of 3D plot, by default -90

    Returns
    -------
    M : numpy array
        mTotal ass matrix
    r_cg : numpy array
        Total COG vector
    I_rotor : float
        Rotor moment of inertia about LSS.
    R_rotor : float
        Rotor radius
    sum_df : pandas DataFrame
        A dataframe to summarize mass properties of different WT components.
    """

    N = 8 # ptfm - tower - nacelle - yawbr - hub - blade 1 - blade 2 - blade 3
    n = 0
    m_n = np.zeros((N,1)) # component mass
    r_n = np.zeros((N,3)) # component cg
    i_n = np.zeros((N,3,3)) # component local inertia matrix

    root_folder = os.path.dirname(fst_file)
    fst = fstin(fst_file)
    ED_file = os.path.join(root_folder,fst['EDFile'].strip('"').strip("'"))
    ED_dict = fstin(ED_file)

    # Platfrom
    m_n[n] += ED_dict['PtfmMass']
    i_n[n] += np.diag([ED_dict[f'Ptfm{i}Iner'] for i in ['R','P','Y']])
    r_n[n] += np.array([ED_dict[f'PtfmCM{i}t'] for i in ['x','y','z']])
    r_n[n,2] += ED_dict['PtfmRefzt']
    n += 1

    # Tower (Tower props relative to tower base)
    twr_ht = ED_dict['TowerHt']
    twr_bs = ED_dict['TowerBsHt']
    twr_file = os.path.join(root_folder,ED_dict['TwrFile'].strip('"').strip("'"))
    twr_dict = fstin(twr_file)
    twr_props = twr_dict['TowProp']
    ht_fract = twr_props[:,0]
    twr_sec_ht = ht_fract*(twr_ht-twr_bs)
    twr_mden = twr_props[:,1]*twr_dict['AdjTwMa']
    twr_mass = np.trapz(twr_mden,twr_sec_ht)
    twr_cm_z = np.trapz(twr_mden*twr_sec_ht,twr_sec_ht)/twr_mass + twr_bs
    twr_moi = np.zeros(3)
    twr_moi[0] = np.trapz(twr_mden*(twr_sec_ht-twr_cm_z)**2,twr_sec_ht)
    twr_moi[1] = twr_moi[0]

    m_n[n] = twr_mass
    r_n[n] = np.array([0,0,twr_cm_z])
    i_n[n] = np.diag(twr_moi)
    n += 1

    # Nacelle
    overhang = ED_dict['OverHang']
    shft_tilt = ED_dict['ShftTilt']*np.pi/180
    twr2shft = ED_dict['Twr2Shft']
    hub_cm = ED_dict['HubCM']

    r_apex = np.array([                    overhang * np.cos(shft_tilt),
                                                                     0.,
                       twr_ht + twr2shft + overhang * np.sin(shft_tilt)])

    m_n[n] += ED_dict['NacMass']
    i_n[n] += np.diag([0,0,ED_dict['NacYIner']])
    r_n[n] += np.array([ED_dict[f'NacCM{i}n'] for i in ['x','y','z']])
    r_n[n,2] += twr_ht
    n += 1

    # Yaw Bearing
    m_n[n] += ED_dict['YawBrMass']
    r_n[n,2] += twr_ht
    n += 1

    # Hub
    m_n[n] += ED_dict['HubMass']
    r_n[n] += r_apex + hub_cm * np.array([np.cos(shft_tilt),0.,np.sin(shft_tilt)])
    n += 1

    # Blades
    R_rotor = ED_dict['TipRad']
    blade_files = [os.path.join(root_folder,ED_dict[f'BldFile{i+1}'].strip('"').strip("'")) for i in range(ED_dict['NumBl'])]
    I_rotor = ED_dict['HubIner']
    blds_r1 = np.zeros((len(blade_files),3))
    blds_r2 = np.zeros((len(blade_files),3))

    azimuth = 0
    for b,bld_file in enumerate(blade_files):
        precone = ED_dict[f'PreCone({b+1})']*np.pi/180
        bld = fstin(bld_file)
        bld_prop = bld['BldProp']

        ## node locations
        bld_fract = bld_prop[:,0]
        bld_sec_R = bld_fract * (ED_dict['TipRad']-ED_dict['HubRad'])
        
        alpha = -shft_tilt
        theta = precone

        if azimuth < np.pi/2 or azimuth > 3*np.pi/2:
            theta = precone
        else:
            theta = np.pi - (precone)

        bld_sec_x = r_apex[0] + bld_sec_R*np.abs(np.cos(azimuth))*( np.sin(theta)*np.cos(alpha) + np.cos(theta)*np.sin(alpha))
        bld_sec_y = r_apex[1] + bld_sec_R*np.sin(azimuth)
        bld_sec_z = r_apex[2] + bld_sec_R*np.abs(np.cos(azimuth))*(-np.sin(theta)*np.sin(alpha) + np.cos(theta)*np.cos(alpha))

        blds_r1[b] = np.array([bld_sec_x[0],bld_sec_y[0],bld_sec_z[0]]) # for plotting
        blds_r2[b] = np.array([bld_sec_x[-1],bld_sec_y[-1],bld_sec_z[-1]]) # for plotting

        ## mass density
        bld_mden = bld_prop[:,3]*bld['AdjBlMs']

        ## blade mass
        bld_mass = np.trapz(bld_mden,bld_sec_R)

        ## blade center of mass
        bld_Rcm = np.trapz(bld_mden*bld_sec_R,bld_sec_R)/bld_mass # distance from rotor apex
        bld_cm = np.array([np.trapz(bld_mden*bld_sec_x,bld_sec_R)/bld_mass, # global x location
                           np.trapz(bld_mden*bld_sec_y,bld_sec_R)/bld_mass, # global y location
                           np.trapz(bld_mden*bld_sec_z,bld_sec_R)/bld_mass]) # global z location

        ## blade moments of inertia
        I_rotor += np.trapz(bld_mden*((bld_sec_R + ED_dict['HubRad']) * np.cos(precone))**2,bld_sec_R) # about rotor axis
        bld_moi = np.array([np.trapz(bld_mden * ((bld_sec_y-bld_cm[1])**2 + (bld_sec_z-bld_cm[2])**2),bld_sec_R), # about local x-axis passing through blade cm
                            np.trapz(bld_mden * ((bld_sec_x-bld_cm[0])**2 + (bld_sec_z-bld_cm[2])**2),bld_sec_R), # about local y-axis passing through blade cm
                            np.trapz(bld_mden * ((bld_sec_x-bld_cm[0])**2 + (bld_sec_y-bld_cm[1])**2),bld_sec_R)]) # about local z-axis passing through blade cm

        ## assign mass props
        m_n[n] = bld_mass
        r_n[n] = bld_cm
        i_n[n] = np.diag(bld_moi)
        n += 1
        azimuth += 2*np.pi/3

    # Evaluate global mass props about MSL
    moi_mat = np.ones((3,3)) - np.eye(3)
    m_tot = np.sum(m_n)
    r_cg = np.sum(m_n*r_n,axis=0) / m_tot

    M = np.zeros((6,6))
    M[:3,:3] += np.eye(3) * m_tot
    M[3:,3:] += np.sum(i_n,axis=0) # add local inertia
    M[3:,3:] += np.diag(np.sum(m_n[:,None] * (moi_mat[None,:,:] @ r_n[:,:,None]**2),axis=0)[:,0])
    
    M[:3,3:] += m_tot * np.array([[        0.0,     r_cg[2], -r_cg[1]],
                                  [-r_cg[2],            0.0,  r_cg[0]],
                                  [ r_cg[1], -r_cg[0],           0.0]])
    M[3:,:3] += M[:3,3:].T

    columns = ['mass','x_g','y_g','z_g','i_xx','i_yy','i_zz']
    components = np.array(['platform','tower','nacelle','yaw_bear','hub','blade_1','blade_2','blade_3'])

    data = np.vstack((m_n[:,0],r_n[:,0],r_n[:,1],r_n[:,2],i_n[:,0,0],i_n[:,1,1],i_n[:,2,2])).T

    sum_df = pd.DataFrame(columns=columns,data=data,index=components)

    if csv_out:
        sum_df.to_csv(csv_out)
    
    if plot_wt:
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(0,0,0,'+b',label='origin')
        n=0
        ax.plot(r_n[n,0],r_n[n,1],r_n[n,2],c='orange',marker='s',label='Platform cg',ls='None') # platform
        n+=1

        ax.plot([0,0],[0,0],[twr_bs,twr_ht],c='grey') # tower line
        ax.plot(r_n[n,0],r_n[n,1],r_n[n,2],c='grey',marker='x',label='Tower cg',ls='None') # tower cg
        n+=1

        ax.plot(r_n[n,0],r_n[n,1],r_n[n,2],c='blue',marker='o',label='Nacelle cg',ls='None') # tower cg
        n+=1

        ax.plot(r_n[n,0],r_n[n,1],r_n[n,2],c='red',marker='o',label='Yaw bearing cg',ls='None') # tower cg
        n+=1

        ax.plot(r_n[n,0],r_n[n,1],r_n[n,2],c='green',marker='o',label='Hub cg',ls='None') # tower cg
        n+=1

        ax.plot([0,r_apex[0]],[0,r_apex[1]],[twr_ht+twr2shft,r_apex[2]],c='black',ls='--',label='shaft') # shaft

        for b in range(len(blade_files)):
            ax.plot([blds_r1[b,0],blds_r2[b,0]],[blds_r1[b,1],blds_r2[b,1]],[blds_r1[b,2],blds_r2[b,2]],c='black',ls='-') # shaft
            ax.plot(r_n[n,0],r_n[n,1],r_n[n,2],c='black',marker='x',label='bld cg',ls='None') # tower cg
            n+=1

        fs_range = np.arange(-100,101,200)
        fs_x,fs_y = np.meshgrid(fs_range,fs_range)
        ax.plot_surface(fs_x,fs_y,np.zeros_like(fs_x),color='blue',alpha=0.25)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        plt.sca(ax)
        plt.grid()
        plt.legend()
        plt.axis('equal')

    return M,r_cg,I_rotor,R_rotor,sum_df


if __name__ == '__main__':
    fst_file = r'openfast_models\15MW_UMaineVolturnUS-S\IEA-15-240-RWT-UMaineSemi\IEA-15-240-RWT-UMaineSemi.fst'
    m,r_cg,MOI,I_rot,df_sum = find_mass_props(fst_file)
    