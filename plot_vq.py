from matplotlib.ticker import ScalarFormatter, NullFormatter
import numpy as np
import matplotlib.pyplot as plt

def set_relative_compression_curve_axes(ax):
    ax.set_xlabel("VQ vectors (log scale)")
    ax.set_ylabel("Adjusted compression ratio")
    ax.set_xscale("log", base=2, subs=np.linspace(1,2,16))
    ax.set_yscale("log", base=2, subs=np.linspace(1,2,16))
    ax.set_yticks([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16])
    ax.set_ylim(0.5, 32)
    ax.set_xlim(2, 260)    
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())

def simple_compression_curve(ax, ks, z, z_surrogate, c='C0'):
    adj_ratio = np.array(z)  / z_surrogate
    ax.plot(ks, adj_ratio, c=c)

def plot_rd_curve(ax, ks, z, z_surrogate, d, cmap=plt.cm.viridis):
    """
        Plot a scatter plot of adjusted compression ratio versus distortion.
    """
    adj_ratio = np.array(z)  / z_surrogate
    c = np.log2(ks) / 8.0
    ax.scatter(adj_ratio, 20.0*np.log10(d), c=cmap(c), marker='+')
    ax.set_ylim(-60, 5)
    ax.set_xlim(0.5, 16)
    ax.set_xscale("log", base=2, subs=np.linspace(1,2,16))
    ax.set_xlabel("Adjusted compression ratio")
    ax.set_ylabel("VQ distortion (dB)")
    # add a color bar, using the values from ks as ticks
    # instead of the default 0-1
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax)
    cb.set_ticks(np.linspace(0,1,8))
    cb.set_ticklabels([str(int(k)) for k in np.geomspace(2,256,8)])

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())


    
def plot_compression_curve(ax, ks, z, z_surrogate, d, z_c='C2', d_c='C3'):
    """
        Simple compression curve plot, showing the compression ratio versus
        the surrogate compression ratio, along with the relative distortion compared 
        to expected quantisation distortion.
    """
    # adjust for the surrogate
    adj_ratio = np.array(z)  / z_surrogate
    ax.loglog(ks, adj_ratio, c=z_c, base=2)
    ax.loglog(ks, adj_ratio, c=z_c, base=2, marker='+')    
    # expect compression = 1.0 for random data
    ax.axhline(1.0, ls=':', c=z_c)
    
    
    ax2 = ax.twinx()
    ax2.set_ylabel("Relative VQ distortion (dB)")
    # expected quantisation noise = 6.02 * quantisation_bits
    k_bits = np.log2(np.array(ks))    
    exp_dist = (6.02 * k_bits - 1.76) 
    # expect relative distortion = 0.0 for random quantisation noise
    ax2.axhline(0.0, ls=':', c=d_c)
    ax2.semilogx(ks, 20.0*np.log10(d)+exp_dist, c=d_c, alpha=0.5, base=2, subs=np.linspace(1,2,9))        
    ax2.set_ylim(-10, 10)    
    set_relative_compression_curve_axes(ax)
    


def plot_absolute_compression_curve(ax, ks, z, d, z_c='C0', d_c='C1', n_dim=1):
    """
        Simple compression curve plot, showing the absolute compression ratio
        and the absolute distortion        
    """
    ax.set_xlabel("VQ vectors (log scale)")
    
    k_bits = np.log2(np.array(ks))
    max_inf = 8.0/k_bits   
    # we always compress bytes, so compute
    # the "fraction of a byte" with this many symbols    
    # adjust for the surrogate
    ax.semilogx(ks, z, c=z_c, base=2)
    ax.semilogx(ks, z, c=z_c, base=2, marker='+')
    ax.semilogx(ks, max_inf, c=z_c, ls=':', base=2)
  
    
    # expect compression = 1.0 for random data
    ax.axhline(1.0, ls='--', c=z_c, alpha=0.5)
    ax2 = ax.twinx()
    ax2.set_ylabel("Absolute distortion (dB)")
    if n_dim == 1:
        exp_dist =  6.02 * k_bits - 1.76
    else:
        # for n_dim > 1, we have to account for the fact that
        # we have n_dim independent quantisation noises
        # this is an odd empirical approximation...
        exp_dist =  (6/n_dim - 0.6)*k_bits - np.pi * np.log2(k_bits) - np.pi/2

    ax2.semilogx(ks, -exp_dist, c=d_c, ls=':', base=2)    
    ax2.semilogx(ks, 20.0*np.log10(d), c=d_c, alpha=0.5, base=2, subs=np.linspace(1,2,9))        
    
    
    ax2.set_ylim(-50, 50)    
    set_relative_compression_curve_axes(ax)
    ax.set_ylabel("Absolute compression ratio (log scale)")
    