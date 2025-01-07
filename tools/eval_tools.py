# Tools for evaluation of NN models

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import random

def plot_test_set_comparison(Xtest,Ytest,Ymodel,Xtestname='Input',Ytestname='Output',Ymodelname='Modelled output'):
  plt.rcParams.update({'font.size': 18})
  fig, ax = plt.subplots(2,2,figsize = (20,15),constrained_layout=True)
  px = ax[0,0].pcolormesh(Xtest, cmap = 'viridis')
  fig.colorbar(px,ax=ax[0,0])
  vmin = np.minimum(np.min(Ytest),np.min(Ymodel))
  vmax = np.maximum(np.max(Ytest),np.max(Ymodel))
  py = ax[0,1].pcolormesh(Ytest, vmin=vmin,vmax=vmax,cmap = 'plasma')
  #vmin, vmax = py.get_clim()
  ax[1,0].pcolormesh(Ymodel, vmin = vmin, vmax = vmax, cmap = 'plasma')
  pe = ax[1,1].pcolormesh((Ymodel- Ytest)/Ytest, cmap = 'plasma')
  fig.colorbar(py,ax=ax[0,1])
  fig.colorbar(py,ax=ax[1,0])
  fig.colorbar(pe,ax=ax[1,1])
  ax[0,0].set_title(Xtestname)
  ax[0,1].set_title(Ytestname)
  ax[1,0].set_title(Ymodelname)
  ax[1,1].set_title('Normalised error')

  return fig
