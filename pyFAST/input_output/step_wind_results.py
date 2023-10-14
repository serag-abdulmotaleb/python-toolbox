import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyFAST.input_output import FASTOutputFile as fstout

def read_step_wind(out_file,keys,T_sim,T_trans,steps,prefix=''):
    df = fstout(out_file).toDataFrame()
    t = df['Time_[s]'].to_numpy()
    dt = t[1] - t[0]
    
    stdy_dict = {}

    for key in keys:
        stdy_dict[key] = []
    
    for i in range(steps):
        n1 = int(i*(T_sim+T_trans)/dt)
        n2 = n1 + int(T_sim/dt)
        for key in keys:
            stdy_dict[key].append(df[key].to_numpy()[n1:n2][-1])
    stdy_df = pd.DataFrame(stdy_dict)

    stdy_df.to_csv(prefix + 'steady_state_results.csv')
    return stdy_df