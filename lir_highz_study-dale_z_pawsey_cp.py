
#
# ICRAR - International Centre for Radio Astronomy Research
# (c) UWA - The University of Western Australia, 2018
# Copyright by UWA (in the framework of the ICRAR)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""Size plots"""

import functools
import numpy as np
from pylab import scatter
import pylab
import matplotlib.pyplot as plt
import common
import utilities_statistics as us
import pandas as pd
from collections import OrderedDict


import statistics as stat
zlist=[0]
zlist = [0, 0.194738848008908]
#zlist = [0, 0.194738848008908, 0.909822023685613, 2.00391410007239, 3.0191633709527, 3.95972701662501, 5.02220991014863, 5.96592270612165, 7.05756323172746, 8.0235605165086]
#zlist = [0, 0.909822023685613, 2.00391410007239,3.0191633709527,3.95972701662501, 5.02220991014863]
#zlist = [0,1,1.5,2,3,4,5]

#zlist = [0, 1.0, 2.00391410007239, 3.0191633709527, 3.95972701662501, 5.02220991014863, 5.96592270612165, 7.05756323172746, 8.0235605165086, 8.94312532315157, 9.95650268434316]
#zlist = [0.359789,0.849027,1.39519, 2.00392]


#9.95655] #0.194739, 0.254144, 0.359789, 0.450678, 0.8, 0.849027, 0.9, 1.20911, 1.28174, 1.39519, 1.59696, 2.00392, 2.47464723643932, 2.76734390952347, 3.01916, 3.21899984389701, 3.50099697082904, 3.7248038025221, 3.95972, 4.465197621546, 4.73693842543988] #[5.02220991014863, 5.52950356184419, 5.96593, 6.55269895697227, 7.05756323172746, 7.45816170313544, 8.02352, 8.94312532315157, 9.95655]
    #[0.016306640039433, 0.066839636933135, 0.084236502339783, 0.119886040396529, 0.138147164704691, 0.175568857770275, 0.214221447279112, 0.23402097095238, 0.274594901875312, 0.316503156974571]

##################################
#Constants
RExp     = 1.67
MpcToKpc = 1e3
G        = 4.299e-9 #Gravity constant in units of (km/s)^2 * Mpc/Msun
c_speed  = 299792458.0 #m/s
PI       = 3.141592654

mlow = 6.5
mupp = 12.5
dm = 0.2
mbins = np.arange(mlow,mupp,dm)
xmf = mbins + dm/2.0
Lsun = 3.828 * 10**26 #W
msun = -26.74
print("This is Lsun")
print(Lsun)
def prepare_data(hdf5_data, seds, seds_bands, seds_bc, index, LFs_dust, obsdir):

    (h0, volh, mdisk, mbulge, sfr_disk, sfr_burst,typ,mb,mb,mgb,mgd) = hdf5_data
    bin_it = functools.partial(us.wmedians, xbins=xmf)

    lir_disk = seds[0]
    lir_bulge = seds[1]
    lir_total = seds[2] #total IR luminosity in units of Lsun
  
    lir_cont_bc = seds_bc[1]
    lir_total = np.array(lir_total, dtype=np.float64)
    
    Tbc = 50.0
    Tdiff = 25.0

    
    Teff = Tbc * lir_cont_bc[0] + Tdiff * (1 - lir_cont_bc[0])

    lir_total_W = lir_total[0] * Lsun  #units of W
    print("This is lir_total")
    print(lir_total[0][:4])
    print("This is lir_total_W")
    print(lir_total_W[:4])
    print("This is Lsun")
    print(Lsun)
    seds_disk = seds_bands[0]
    seds_bulge = seds_bands[1]
    seds_total = seds_bands[2] 
     
    band_14 = seds_total[12,:] #the absolute magnitude of the 1.4 GHz band
    band_30 = seds_total[13,:] #the absolute magnitude of the 3.0 GHz band

    L1p4radio = seds_total[12,:]
    #compute luminosity function

    ind = np.where((lir_total > 0) & (lir_total < 1e20))
    H, bins_edges = np.histogram(np.log10(lir_total[ind]),bins=np.append(mbins,mupp))
    LFs_dust[index,:] = LFs_dust[index,:] + H

    #median Lradio vs LIR
    lir_total = lir_total[0,:]
    ind = np.where((lir_total > 1e6) & (lir_total < 1e20))
    lir_selected = lir_total[ind]
    lradio_selected = L1p4radio[ind] 
    print(lir_selected.shape, lradio_selected.shape)
    meds_radio = bin_it(x=lir_selected, y=lradio_selected)
  
    return(volh, h0, band_14, band_30, lir_total_W, Teff)

def radio_convert(h0, zlist,seds_lir_lst,hdf5_lir_lst,seds_lir_bc_lst,seds_bands_lst, lir_total_W_lst):
    
    flux_lstlst = []
    qir_lstlst = []
    lir_total_lstlst = []
    
    qir_med_lst = []
    flux_med_lst = []
    lir_med_lst = []
    
    dist_mpc = 10 #distance in  pc
    d = 3.08567758128*10**17 #distance in m
    
    for z in range(len(zlist)):
        
        stellar_mass_lst = [a+b for a,b in zip(hdf5_lir_lst[z][2],hdf5_lir_lst[z][3])]
        stellar_mass_lst = np.array(stellar_mass_lst)
        ind0 = np.where(stellar_mass_lst != 0)
        flux_lst = []
        
        stellar_mass_lst = []
        stellar_mass_lst = [a+b for a,b in zip(hdf5_lir_lst[z][2],hdf5_lir_lst[z][3])]
        stellar_mass_lst = np.array(stellar_mass_lst)
        ind0 = np.where(stellar_mass_lst != 0)
        stellar_mass_lst = stellar_mass_lst[ind0]
        
        
        for i in seds_bands_lst[z][2][9:166]: #iterates over different redshift (z), total liuminosity (index 2 of this list) and the 1.4GHz frequency (11 index)
            flux = 10**((i+48.6)/(-2.5)) *(4*np.pi*(d * 100)**2) / 1e7 #flux in W/Hz
            flux_lst.append(flux)
        
        
    #    print("This is lir_total_W")
    #    print(lir_total_W_lst[z][:4])
    #    print("This is flux")
    #    print(flux_lst[:4])
        
        qir_lst = [np.log10(a/(3.75*10**12))-np.log10(b) for a,b in zip(lir_total_W_lst[z], flux_lst[3])] #qir
    
        qir_lst = np.array(qir_lst,dtype=np.float64)
        flux_lst = np.array(flux_lst[3],dtype=np.float64)
        
        lir_total = np.array(lir_total_W_lst[z], dtype=np.float64)
        ind109 = np.where(lir_total > 10**7 * Lsun)
        lir_total = lir_total[ind109]
        qir_lst = qir_lst[ind109]
        flux_lst = flux_lst[ind109]
        stellar_mass_lst = stellar_mass_lst[ind109]
        
        ind108 = np.where(stellar_mass_lst > 0)
        
        lir_total = lir_total[ind108]
        qir_lst = qir_lst[ind108]
        flux_lst = flux_lst[ind108]
        
        
        indinf = np.where(qir_lst != np.inf)
        
        lir_total = lir_total[indinf]
        qir_lst = qir_lst[indinf]
        flux_lst = flux_lst[indinf]
        
        
        
        
        qir_med = np.median(qir_lst)
        flux_med = np.median(flux_lst)
        lir_med = np.median(lir_total)
        
        qir_lstlst.append(qir_lst)
        flux_lstlst.append(flux_lst)
        lir_total_lstlst.append(lir_total)
        
        qir_med_lst.append(qir_med)
        flux_med_lst.append(flux_med)
        lir_med_lst.append(lir_med)
        
        
        
    qir_bress, bress_rad_lum, bress_lir_total = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,zlist)
    z4 = np.linspace(0,4,endpoint = True)
    delhaize4 = (2.88)*(1+z4)**-0.19
    delhaize_upper4 = (2.88+0.03)*(1+z4)**(-0.19+0.01)
    delhair_lower4 = (2.88-0.03)*(1+z4)**(-0.19-0.01)
    z10 = np.linspace(4,10,endpoint= True)
    delhaize10 = (2.88)*(1+z10)**-0.19
    delhaize_upper10 = (2.88+0.03)*(1+z10)**(-0.19+0.01)
    delhair_lower10 = (2.88-0.03)*(1+z10)**(-0.19-0.01)
    
    plt.plot(zlist,flux_med_lst,label = 'Dale')
    plt.plot(zlist, bress_rad_lum, label = 'Bressan')
    plt.legend()
    plt.yscale('log')
    plt.title('Median radio luminosity vs. z')
    plt.xlabel('z')
    plt.ylabel('Log median radio luminosity (W/Hz)')
    plt.show()
    
    plt.plot(zlist,lir_med_lst,label = 'Dale')
    plt.plot(zlist, bress_lir_total, label = 'Bressan')
    plt.legend()
    plt.yscale('log')
    plt.title('Median lir luminosity vs. z')
    plt.xlabel('z')
    plt.ylabel('Log median fir luminosity (W)')
    plt.show()

    plt.plot(zlist,qir_med_lst,label = 'Dale')
    plt.plot(zlist,qir_bress, label = 'Bressan')
    plt.plot(z4, delhaize4, '-',color = 'purple',label = "Delhaize et al. 2017")
    plt.fill_between(z4, delhaize_upper4, delhair_lower4, alpha=0.2, color='purple')
    plt.xlim(-0.2,4)
    plt.legend()
    plt.title('Median qir vs. z')
    plt.ylabel('Median qir')
    plt.xlabel('z')
    plt.show()
    
    
    plt.plot(zlist,qir_med_lst,label = 'Dale')
    plt.plot(zlist,qir_bress, label = 'Bressan')
    plt.plot(z4, delhaize4, '-',color = 'purple',label = "Delhaize et al. 2017")
    plt.plot(z10, delhaize10, ':' ,color = 'purple')
    plt.fill_between(z4, delhaize_upper4, delhair_lower4, alpha=0.2, color='purple')
    plt.fill_between(z10, delhaize_upper10, delhair_lower10, alpha=0.2, color='purple')
    plt.legend()
    plt.title('Median qir vs. z')
    plt.ylabel('Median qir')
    plt.xlabel('z')
    plt.show()
    
    stellar_mass_lstlst = []
    
    for z in range(len(zlist)):
        stellar_mass_lst = []
        stellar_mass_lst = [a+b for a,b in zip(hdf5_lir_lst[z][2],hdf5_lir_lst[z][3])]
        stellar_mass_lst = [i for i in stellar_mass_lst if i != 0]
        stellar_mass_lstlst.append(np.log10(stellar_mass_lst))
    
 #   print(len(stellar_mass_lstlst[0]))
 #   print(len(qir_lstlst[0]))
 #   
 #   for z in range(len(zlist)):
 #       ztemp = [zlist[z]] * len(qir_lstlst[z])
 #       plt.scatter(ztemp,qir_lstlst[z],c=stellar_mass_lstlst[z])
 #   cbar = plt.colorbar()
 #   plt.title("qir vs. z")
 #   plt.xlab1el("z")
 #   plt.ylabel("qir")
 #   cbar.set_label("Log Stellar Mass (Solar masses)")
 #   plt.show()

def dataframe(hdf5_lir_lst,zlist,seds_bands_lst,seds_lir_lst,lir_total_W_lst,h0,Teff_lst):
    print("This is h0")
    print(h0)
    dist_mpc = 10 #distance in  pc
    d = 3.08567758128*10**17 #distance in m
    lstlst = []
    q = 0
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total, freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,zlist)
    print("This is bress_rad_lum_lst")
    print(bress_rad_lum_lst)
    print("This is the length of qir_lst_bress")
    print(len(qir_lst_bress))
    print(len(qir_lst_bress[0]))
    for z in range(len(zlist)):
        
        stellar_mass_lst = []
        stellar_mass_lst = [a+b for a,b in zip(hdf5_lir_lst[z][2],hdf5_lir_lst[z][3])]
        mstars_disk_array = np.array(hdf5_lir_lst[z][2])
        mstars_bulge_array = np.array(hdf5_lir_lst[z][3])
        stellar_mass_array = np.array(stellar_mass_lst)
        sfr_disk_array = np.array(hdf5_lir_lst[z][4])
        sfr_burst_array = np.array(hdf5_lir_lst[z][5])
        type_array = np.array(hdf5_lir_lst[z][6])
        bgm_array = np.array(hdf5_lir_lst[z][7]) #bulge gas metal mass array
        dgm_array = np.array(hdf5_lir_lst[z][8]) #disk gas metal mass array
        mgas_bulge_array = np.array(hdf5_lir_lst[z][9]) #disk gas mass array
        mgas_disk_array = np.array(hdf5_lir_lst[z][10]) #bulge mass array
        print("This is length of mstars_disk_array")
        print(len(mstars_disk_array))
        ind0 = np.where(stellar_mass_array > 0)
        stellar_mass_lst = stellar_mass_array[ind0]
        mstars_disk_array = mstars_disk_array[ind0]
        mstars_bulge_array = mstars_bulge_array[ind0]
        sfr_disk_array = sfr_disk_array[ind0]
        sfr_burst_array = sfr_burst_array[ind0]
        type_array = type_array[ind0]
        bgm_array = bgm_array[ind0]
        dgm_array = dgm_array[ind0]
        mgas_disk_array = mgas_disk_array[ind0]
        mgas_bulge_array = mgas_bulge_array[ind0]
        lir_total_w = lir_total_W_lst[z]
        Teff = Teff_lst[z]        

        redshift = zlist[z]
        
        print("This is length of mstars_disk_array")
        print(len(mstars_disk_array))
        flux_lst = []
        for i in seds_bands_lst[z][2][9:16]: #iterates over different redshift (z), total liuminosity (index 2 of this list) and the 1.4GHz frequency (11 index)
            flux = 10**((i+48.6)/(-2.5)) * (4*np.pi*(d * 100)**2) / 1e7 #flux in W/Hz
            flux_lst.append(flux)
            
            
        print("This is flux_lst")
        
        print(flux_lst)
        
        print("This is the length of flux_lst")
        print(len(flux_lst))
        dale_qir_lst = [np.log10(a/(3.75*10**12))-np.log10(b) for a,b in zip(lir_total_W_lst[z], flux_lst[3])] #qir
        for j in range(len(mstars_disk_array)):
            lst = []
            mstars_disk = mstars_disk_array[j]
            mstars_bulge = mstars_bulge_array[j]
            sfr_disk = sfr_disk_array[j]
            sfr_burst = sfr_burst_array[j]
            mstars_tot = (mstars_disk + mstars_bulge)/h0
            sfr = (sfr_disk + sfr_burst) / 1e9 / h0
            sb_frac = (sfr_burst/1e9/h0)/sfr
            typ = type_array[j]
            gas_metallicity = ((bgm_array[j] + dgm_array[j])/(mgas_disk_array[j] + mgas_bulge_array[j]))/0.018 #gives gas metallicity
            lir_w = lir_total_w[j]
            teff = Teff[j]
            gas_mass = (mgas_bulge_array[j] + mgas_disk_array[j])/h0


            
           # print((z,j))
            qir_bress = qir_lst_bress[z][j]
            bress_rad_lum = bress_rad_lum_lst[z][j] #radio luminosity of bressan model in watts
            qir_dale = dale_qir_lst[j]
            fir_flux = lir_total_W_lst[z][j]
            rad_flux = flux_lst[3][j]

            freefree = freefree_lst[z][j] #freefree luminosity
            sync = sync_lst[z][j] #syncrhtotron luminosity


            lst.append(redshift)
            lst.append(mstars_disk)
            lst.append(mstars_bulge)
            lst.append(mstars_tot)
            lst.append(sfr_disk)
            lst.append(sfr_burst)
            lst.append(sfr)
            lst.append(qir_bress)
            lst.append(qir_dale)
            lst.append(fir_flux)
            lst.append(rad_flux)
            lst.append(typ)
            lst.append(gas_metallicity)
            lst.append(lir_w)
            lst.append(bress_rad_lum)
            lst.append(gas_mass)
            lst.append(freefree)
            lst.append(sync)
            lst.append(teff)
            lst.append(sb_frac)
            lstlst.append(lst)
    
    df = pd.DataFrame(lstlst, columns = ['z','mstars_disk','mstars_bulge','mstars_tot','sfr_disk','sfr_bulge','sfr','qir_bress','qir_dale','fir_flux','dale_rad_lum','type','gas_metal','lir_w','bress_rad_lum','gas_mass','freefree','sync','Teff','sb_frac'])
   
    df['sfr/q'] = 'q'
    df['sf_test'] = 1
    ab_lst = []
    for z in zlist:
        sfr_z = df.loc[((df['z'] == z)&(df['mstars_tot'] > 10**(9))&(df['mstars_tot'] < 10**(10))&(df['type'] == 0)),'sfr']
        m_z = df.loc[((df['z'] == z)&(df['mstars_tot'] > 10**(9))&(df['mstars_tot'] < 10**(10))&(df['type'] == 0)),'mstars_tot']
        
        sfr_z = np.log10(sfr_z)
        m_z = np.log10(m_z)
        
        a,b = np.polyfit(m_z,sfr_z,1)
        df['sf_test'] = abs(np.log10(df['sfr']) - (a * np.log10(df['mstars_tot']) + b))
        print(df)
        df.loc[((df['z'] == z)&(df['sf_test'] < 0.5)), 'sfr/q'] = 'sf'
        
        
        
        sf_gal_sfr = df.loc[((df['z'] == z)&(df['sfr/q'] == 'sf')),'sfr']
        sf_gal_m = df.loc[((df['z'] == z)&(df['sfr/q'] == 'sf')),'mstars_tot']
        q_gal_sfr = df.loc[((df['z'] == z)&(df['sfr/q'] == 'q')),'sfr']
        q_gal_m = df.loc[((df['z'] == z)&(df['sfr/q'] == 'q')),'mstars_tot'] 
        strz = str(z)
        tit = "z = " + strz[:6]
        
        
        
        plt.title(tit)
        x = np.linspace(8, 12, 10)
        y = a * x + b
        y1 = a*x + b + 0.3
        y2 = a*x + b - 0.3
        
        ab_lst.append((a,b))
      #  y1 = np.log10(y1)
      #  y2 = np.log10(y2)
      #  plt.plot(10**(x),10**(y),color = 'red',label = 'line fit')
      #  plt.fill_between(x,y1,y2,color = 'red')
      #  plt.scatter(sf_gal_m, sf_gal_sfr, label = 'Star forming')
      #  plt.scatter(q_gal_m, q_gal_sfr, label = 'Quenched')
      #  plt.yscale('log')
      #  plt.xscale('log')
      #  plt.xlabel("Stellar mass")
      #  plt.ylabel("SFR")
      #  plt.xlim(10**8,10**12)
      #  plt.ylim(10**(-3),10**(3))
      #  plt.legend()
      #  plt.show()     
      #  
   # df.loc[(df['sf_test'] < 10**0.3), 'sfr/q'] = 'sf'
    
  #  sf_gal_sfr = df.loc[df['sfr/q'] == 'sf','sfr']
  #  sf_gal_m = df.loc[df['sfr/q'] == 'sf','mstars_tot']
  #  q_gal_sfr = df.loc[df['sfr/q'] == 'q','sfr']
  #  q_gal_m = df.loc[df['sfr/q'] == 'q','mstars_tot'] 
  #  plt.title("All z")
  #  plt.scatter(sf_gal_m, sf_gal_sfr, label = 'Star forming')
  #  plt.scatter(q_gal_m, q_gal_sfr, label = 'Quenched')
  #  plt.yscale('log')
  #  plt.xscale('log')
  #  plt.xlabel("Stellar mass")
  #  plt.ylabel("SFR")
  #  plt.xlim(10**8,10**12)
  #  plt.ylim(10**(-3),10**(3))
  #  plt.legend()
  #  plt.show()
    return df,ab_lst


def ms_sfr_m(df,z):
    pow_lst = list((np.arange(0,30)/6)+8)
    m_lst = []
    med_sfr_lst = []
    low_sfr_lst= []
    high_sfr_lst = []
    for i in range(len(pow_lst)-1):
        try:
            m = pow_lst[i]
            n = pow_lst[i+1]
            sfr = df.loc[((df['z'] == z)&(df['mstars_tot'] > 10**(m))&(df['mstars_tot'] < 10**(n))&(df['sfr'] > 0)),'sfr']
            low_sfr_lst.append(np.percentile(sfr,16))
            high_sfr_lst.append(np.percentile(sfr,84))
            med_sfr = np.median(sfr)
            med_sfr_lst.append(med_sfr)
            mid_m = 10**(n/2+m/2)
            m_lst.append(mid_m)
        except:
            pass
    return med_sfr_lst, high_sfr_lst, low_sfr_lst, m_lst

def qir_z_plt(df):
    
    df2 = pd.read_csv('Delvecchio_data.csv')
    print(df2)
    i = 0
    z = 0
    q = 0
    h = 0
    g = 0
    mlst = [8,9,9.5,10,10.5,11,12]
    fig, axs = plt.subplots(2, 3)
    
    
    
    
    
    for i in range(6):
        m = mlst[i]
        n = mlst[i+1]
        print("This is m and n")
        print(m)
        print(n)
        qir_bress = df.loc[((df['mstars_tot'] > 10**(m)) & (df['mstars_tot'] < 10**(n))& (df['sfr/q'] == 'sf')),'qir_bress']
        redshift = df.loc[((df['mstars_tot'] > 10**(m)) & (df['mstars_tot'] < 10**(n))& (df['sfr/q'] == 'sf')),'z']
        mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(redshift,qir_bress,False,False)
        qir_nondet= df2.loc[((df2['mass_min'] == m) & (df2['mass_max'] == n)),'qir_nondet']
        qir_nondet_err = df2.loc[((df2['mass_min'] == m) & (df2['mass_max'] == n)),'dqir_nondet']
            
        qir_all= df2.loc[((df2['mass_min'] == m) & (df2['mass_max'] == n)),'qir_all']
        qir_all_err= df2.loc[((df2['mass_min'] == m) & (df2['mass_max'] == n)),'dqir_all']
        z_delv = df2.loc[((df2['mass_min'] == m) & (df2['mass_max'] == n)),'zmean']
        k = mlst[i]
        j = mlst[i+1]
        tit1 = '10^'+str(k)
        tit2 = '< M/$M_\odot$ < 10^'+str(j)
        titlst = ['$10^{8}$< M/$M_\odot$ < $10^{9}$','$10^{9}$< M/$M_\odot$ < $10^{9.5}$','$10^{9.5}$< M/$M_\odot$ < $10^{10}$','$10^{10}$< M/$M_\odot$ < $10^{10.5}$','$10^{10.5}$< M/$M_\odot$ < $10^{11}$','$10^{11}$< M/$M_\odot$ < $10^{12}$']
        tit = tit1+tit2
        titpng = tit+str('.png')
        
        axs[g, h].hexbin(redshift,qir_bress,cmap = 'rainbow',mincnt = 1, vmin = 1, vmax = 10, gridsize = 30)
        axs[g, h].plot(mid_rad_lst,med_fir_lst,'black')
        axs[g, h].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
        axs[g, h].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
        axs[g, h].errorbar(z_delv, qir_all, yerr=qir_all_err, fmt="o",label = 'avg. indv dets and stacked non-dets')
        axs[g, h].errorbar(z_delv, qir_nondet, yerr=qir_nondet_err, fmt="o",label = 'stacks undet')
        axs[g, h].set_ylim([0,4])
        axs[g, h].set_xlim([0,5])
        axs[g, h].set_title(titlst[i],fontsize = 12.5)
        for ax in axs.flat:
            ax.set(xlabel='z', ylabel='qir')
            ax.label_outer()
        
        q +=1
        g = q//3
        h = q%3
    plt.legend()
    fig.set_size_inches(9, 9)
    fig.suptitle('qir vs. z for different mass bins')
    #plt.title('qir vs. z for different mass bins')
    plt.show()
    
    for i in range(6):
        m = mlst[i]
        n = mlst[i+1]
        qir_bress_median_lst = []
        qir_dale_median_lst = []
        qir_bress_low_lst = []
        qir_bress_high_lst = []
        qir_dale_low_lst = []
        qir_dale_high_lst = []
        zlst = []
        try:
            for z in zlist:
                qir_nondet= df2.loc[((df2['mass_min'] == m) & (df2['mass_max'] == n)),'qir_nondet']
                qir_nondet_err = df2.loc[((df2['mass_min'] == m) & (df2['mass_max'] == n)),'dqir_nondet']
            
                qir_all= df2.loc[((df2['mass_min'] == m) & (df2['mass_max'] == n)),'qir_all']
                qir_all_err= df2.loc[((df2['mass_min'] == m) & (df2['mass_max'] == n)),'dqir_all']
                z_delv = df2.loc[((df2['mass_min'] == m) & (df2['mass_max'] == n)),'zmean']

                qir_bress_m = df.loc[((df['mstars_tot'] > 10**(m)) & (df['mstars_tot'] < 10**(n))& (df['sfr/q'] == 'sf')),'qir_bress']
                redshift = df.loc[((df['mstars_tot'] > 10**(m)) & (df['mstars_tot'] < 10**(n))& (df['sfr/q'] == 'sf')),'z']
                mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(xmf,np.log10(bhist),False,False)
                
                qir_bress_low_lst.append(np.percentile(qir_bress_m,16))
                qir_bress_high_lst.append(np.percentile(qir_bress_m,84))
                qir_dale_low_lst.append(np.percentile(qir_dale_m,16))
                qir_dale_high_lst.append(np.percentile(qir_dale_m,84))
                
                qir_bress_median_lst.append(np.median(qir_bress_m))
                qir_dale_median_lst.append(np.median(qir_dale_m))
                zlst.append(z)
        except:
            pass
        k = mlst[i]
        j = mlst[i+1]
        tit1 = '10^'+str(k)
        tit2 = '< M/$M_\odot$ < 10^'+str(j)
        titlst = ['$10^{8}$< M/$M_\odot$ < $10^{9}$','$10^{9}$< M/$M_\odot$ < $10^{9.5}$','$10^{9.5}$< M/$M_\odot$ < $10^{10}$','$10^{10}$< M/$M_\odot$ < $10^{10.5}$','$10^{10.5}$< M/$M_\odot$ < $10^{11}$','$10^{11}$< M/$M_\odot$ < $10^{12}$']
        tit = tit1+tit2
        titpng = tit+str('.png')
        
      #  plt.plot(zlst,qir_bress_median_lst,color = 'red',label = "Bressan")
      #  plt.plot(zlst,qir_bress_low_lst,':',color='red')
      #  plt.plot(zlst,qir_bress_high_lst,':',color='red')
      #  plt.fill_between(zlst,qir_bress_low_lst,qir_bress_high_lst,alpha = 0.2, color= 'red')
      #  plt.plot(zlst,qir_dale_median_lst,color = 'blue',label = "Dale")
      #  plt.plot(zlst,qir_dale_low_lst,':',color='blue')
      #  plt.plot(zlst,qir_dale_high_lst,':',color='blue')
      #  plt.fill_between(zlst,qir_dale_low_lst,qir_dale_high_lst,alpha = 0.2, color= 'blue')
      #  plt.errorbar(z_delv, qir_all, yerr=qir_all_err, fmt="o",label = 'stacking non-detections')
      #  plt.errorbar(z_delv, qir_nondet, yerr=qir_nondet_err, fmt="o",label = 'average of individual detections')
      #  plt.ylim([0,4])
      #  plt.xlim([0,5])
       # plt.title(tit)
      #  plt.show()

        
        
        
        g = q//3
        h = q%3
        print("This is g and h")
        print(g,h)
        q +=1
        axs[g, h].plot(zlst,qir_bress_median_lst,color = 'red',label = "Bressan")
        axs[g, h].plot(zlst,qir_bress_low_lst,':',color='red')
        axs[g, h].plot(zlst,qir_bress_high_lst,':',color='red')
        axs[g, h].fill_between(zlst,qir_bress_low_lst,qir_bress_high_lst,alpha = 0.2, color= 'red')
        axs[g, h].plot(zlst,qir_dale_median_lst,color = 'blue',label = "Dale")
        axs[g, h].plot(zlst,qir_dale_low_lst,':',color='blue')
        axs[g, h].plot(zlst,qir_dale_high_lst,':',color='blue')
        axs[g, h].fill_between(zlst,qir_dale_low_lst,qir_dale_high_lst,alpha = 0.2, color= 'blue')
        axs[g, h].errorbar(z_delv, qir_all, yerr=qir_all_err, fmt="o",label = 'avg. indv dets and stacked non-dets')
        axs[g, h].errorbar(z_delv, qir_nondet, yerr=qir_nondet_err, fmt="o",label = 'stacks undet')
        axs[g, h].set_ylim([0,4])
        axs[g, h].set_xlim([0,5])
        axs[g, h].set_title(titlst[i],fontsize = 12.5)
        for ax in axs.flat:
            ax.set(xlabel='z', ylabel='qir')
            ax.label_outer()
    fig.legend()
    #plt.savefig('qir_vs_z.png')
    fig.set_size_inches(9, 9)
    fig.suptitle('qir vs. z for different mass bins')
    #plt.title('qir vs. z for different mass bins')
    plt.show()
    
    q = 0
    x = int(len(zlist)/2)
    fig, ax = plt.subplots(2,x)
    for z in zlist:
       
        med_sfr_lst, high_sfr_lst, low_sfr_lst, m_lst = ms_sfr_m(df,z)
        strz = str(round(z,2))
        g = q//x
        h = q%x
       # m_z = df.loc[((df['z'] == z)&(df['qir_dale'] < 5)& (df['qir_dale'] > -2)&(df['mstars_tot'] > 10**9)&(df['mstars_tot'] < 10**12)&(df['sfr'] > 10**(-3))&(df['sfr'] < 10**(3))),'mstars_tot']
       # sfr_z = df.loc[((df['z'] == z)&(df['qir_dale'] < 5)& (df['qir_dale'] > -2)&(df['mstars_tot'] > 10**9)&(df['mstars_tot'] < 10**12)&(df['sfr'] > 10**(-3))&(df['sfr'] < 10**(3))),'sfr']
       # qir_bress = df.loc[((df['z'] == z)&(df['qir_dale'] < 5)& (df['qir_dale'] > -2)&(df['mstars_tot'] > 10**9)&(df['mstars_tot'] < 10**12)&(df['sfr'] > 10**(-3))&(df['sfr'] < 10**(3))),'qir_bress']
        m_z = df.loc[((df['z'] == z)&(df['mstars_tot'] > 10**8)&(df['mstars_tot'] < 10**12)&(df['sfr'] > 10**(-3))&(df['sfr'] < 10**(3))),'mstars_tot']
        sfr_z = df.loc[((df['z'] == z)&(df['mstars_tot'] > 10**8)&(df['mstars_tot'] < 10**12)&(df['sfr'] > 10**(-3))&(df['sfr'] < 10**(3))),'sfr']
        qir_bress = df.loc[((df['z'] == z)&(df['mstars_tot'] > 10**8)&(df['mstars_tot'] < 10**12)&(df['sfr'] > 10**(-3))&(df['sfr'] < 10**(3))),'qir_bress']

        
        tit = "z = " + strz[:6]
     #   plt.hexbin(m_z,sfr_z,C=qir_bress,xscale='log',yscale='log',bins = 'log',reduce_C_function=np.median,cmap = 'rainbow')
     #   plt.show()
        print(g,h)
        print('This is q')
        

        m_z = np.log10(m_z)
        sfr_z = np.log10(sfr_z)
        
        
        
        schr_9x = (np.arange(0,30)/7)+8
        schr_9x = 10**(schr_9x)

        r = np.log10(1+z)
        m = np.array(m_lst)
        m0 = 0.5
        a0 = 1.5
        a1 = 0.3
        m1 = 0.36
        a2 = 2.5
        
        b = np.zeros(30)
        mass = (np.arange(0,30)/5)-1
        massplt = (np.arange(0,30)/5)+8
        schr_9y = (mass) - m0 + a0 * r - a1 * (np.maximum(b,(mass -m1 -a2*r)))**2
        
        schr_9y_max = mass - m0+0.07 + (a0+0.15) * r - (a1-0.08) * (np.maximum(b,(mass -m1+0.3 -(a2+0.6)*r)))**2
        schr_9y_min = mass - m0-0.07 + (a0-0.15) * r - (a1+0.08) * (np.maximum(b,(mass -m1-0.3 -(a2-0.6)*r)))**2


        m_lst = np.log10(m_lst)
        med_sfr_lst = np.log10(med_sfr_lst)
        low_sfr_lst = np.log10(low_sfr_lst)
        high_sfr_lst = np.log10(high_sfr_lst)







        c = ax[g,h].hexbin(m_z,sfr_z,C=qir_bress,reduce_C_function=np.median,cmap = 'rainbow',vmin = 2, vmax = 3.5, gridsize = 30)
        ax[g,h].set_title(tit)
        ax[g,h].plot(m_lst,low_sfr_lst,':',color='black')
        ax[g,h].plot(m_lst,high_sfr_lst,':',color='black')
        #ax[q].fill_between(m_lst,low_sfr_lst,high_sfr_lst,alpha = 0.2, color= 'grey')
        ax[g,h].plot(m_lst,med_sfr_lst,color='black',label = 'Fitted Median Line')

       # ax[g,h].plot(massplt,schr_9y,color = 'silver',label = 'Schreiber et al. 2015')
       # ax[g,h].plot(massplt,schr_9y_max,":",color = 'silver')
       # ax[g,h].plot(massplt,schr_9y_min,':', color = 'silver')


        ax[g,h].set_xlim(8,12)
        ax[g,h].set_ylim(-3,3)
        
        print("Here")
        print("This is z")
        print(z)
        q += 1
   # plt.legend()
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('qir')
        
    fig.text(0.5, 0.04, 'Log10(Total Stellar Mass $M_\odot$)', ha='center')
    fig.text(0.04, 0.5, 'Log10(SFR $M_\odot$/yr)', va='center', rotation='vertical')
    fig.set_size_inches(9, 9)
    fig.suptitle('SFR vs. Stellar mass for different redshift - Bressan Model')
    #plt.title('qir vs. z for different mass bins')
    plt.show()
    
    q = 0
    x = int(len(zlist)/2)
    fig, ax = plt.subplots(2,x)
    for z in zlist:
        
        med_sfr_lst, high_sfr_lst, low_sfr_lst, m_lst = ms_sfr_m(df,z)
        
        
        strz = str(round(z,2))
        g = q//x
        h = q%x
        m_z = df.loc[((df['z'] == z)&(df['qir_dale'] < 5)& (df['qir_dale'] > -2)&(df['mstars_tot'] > 10**8)&(df['mstars_tot'] < 10**12)&(df['sfr'] > 10**(-3))&(df['sfr'] < 10**(3))),'mstars_tot']
        sfr_z = df.loc[((df['z'] == z)&(df['qir_dale'] < 5)& (df['qir_dale'] > -2)&(df['mstars_tot'] > 10**8)&(df['mstars_tot'] < 10**12)&(df['sfr'] > 10**(-3))&(df['sfr'] < 10**(3))),'sfr']
        qir_dale = df.loc[((df['z'] == z)&(df['qir_dale'] < 5)& (df['qir_dale'] > -2)&(df['mstars_tot'] > 10**8)&(df['mstars_tot'] < 10**12)&(df['sfr'] > 10**(-3))&(df['sfr'] < 10**(3))),'qir_dale']
        
        tit = "z = " + strz[:6]
     #   plt.hexbin(m_z,sfr_z,C=qir_bress,xscale='log',yscale='log',bins = 'log',reduce_C_function=np.median,cmap = 'rainbow')
     #   plt.show()

        m_z = np.log10(m_z)
        sfr_z = np.log10(sfr_z)
        
        
        
        schr_9x = (np.arange(0,30)/7)+8
        schr_9x = 10**(schr_9x)

        r = np.log10(1+z)
        m = np.array(m_lst)
        m0 = 0.5
        a0 = 1.5
        a1 = 0.3
        m1 = 0.36
        a2 = 2.5
        
        b = np.zeros(30)
        mass = (np.arange(0,30)/5)-1
        massplt = (np.arange(0,30)/5)+8
        schr_9y = (mass) - m0 + a0 * r - a1 * (np.maximum(b,(mass -m1 -a2*r)))**2
        
        schr_9y_max = mass - m0+0.07 + (a0+0.15) * r - (a1-0.08) * (np.maximum(b,(mass -m1+0.3 -(a2+0.6)*r)))**2
        schr_9y_min = mass - m0-0.07 + (a0-0.15) * r - (a1+0.08) * (np.maximum(b,(mass -m1-0.3 -(a2-0.6)*r)))**2


        
        
        m_lst = np.log10(m_lst)
        med_sfr_lst = np.log10(med_sfr_lst)
        low_sfr_lst = np.log10(low_sfr_lst)
        high_sfr_lst = np.log10(high_sfr_lst)


        c = ax[g,h].hexbin(m_z,sfr_z,C=qir_dale,reduce_C_function=np.median,cmap = 'viridis',vmin = 2.5, vmax = 2.8, gridsize = 30)
        ax[g,h].set_title(tit)
        #cb = fig.colorbar(c, ax=ax[q])
        #cb.set_label('qir')
        ax[g,h].plot(m_lst,low_sfr_lst,':',color='black')
        ax[g,h].plot(m_lst,high_sfr_lst,':',color='black')
        #ax[q].fill_between(m_lst,low_sfr_lst,high_sfr_lst,alpha = 0.2, color= 'grey')
        ax[g,h].plot(m_lst,med_sfr_lst,color='black')
       

        ax[g,h].plot(massplt,schr_9y,color = 'silver',label = 'Schreiber et al. 2015')
        ax[g,h].plot(massplt,schr_9y_max,":",color = 'silver')
        ax[g,h].plot(massplt,schr_9y_min,':', color = 'silver')

        ax[g,h].set_xlim(8,12)
        ax[g,h].set_ylim((-3),(3))
        
        
        q += 1
        
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('qir')
    fig.text(0.5, 0.04, 'Log10(Total Stellar Mass $M_\odot$)', ha='center')
    fig.text(0.04, 0.5, 'Log10(SFR $M_\odot$/yr)', va='center', rotation='vertical')

        
        
    fig.set_size_inches(9, 9)
    fig.suptitle('SFR vs. Stellar mass for different redshift - Dale Model')
#    fig.legend()
    #plt.title('qir vs. z for different mass bins')
    plt.show()



    


    
def schr_vs_med(df): #plot from schreiber et al. versus the median line found here
    
    q = 0
    x = int(len(zlist)/2)
    fig, ax = plt.subplots(2,x)
    
    for z in zlist:
        med_sfr_lst, high_sfr_lst, low_sfr_lst, m_lst = ms_sfr_m(df,z)
        g = q//x
        h = q%x        
        strz = str(round(z,2))
        
        tit = 'z = ' +  strz
        
        m_lst = np.log10(m_lst)
        med_sfr_lst = np.log10(med_sfr_lst)
        low_sfr_lst = np.log10(low_sfr_lst)
        high_sfr_lst = np.log10(high_sfr_lst)
        
        schr_9x = (np.arange(0,30)/7)+8
        schr_9x = 10**(schr_9x)

        r = np.log10(1+z)
        m = np.array(m_lst)
        m0 = 0.5
        a0 = 1.5
        a1 = 0.3
        m1 = 0.36
        a2 = 2.5
        
        b = np.zeros(30)
        mass = (np.arange(0,30)/5)-1
        massplt = (np.arange(0,30)/5)+8
        schr_9y = (mass) - m0 + a0 * r - a1 * (np.maximum(b,(mass -m1 -a2*r)))**2
        
        schr_9y_max = mass - m0+0.07 + (a0+0.15) * r - (a1-0.08) * (np.maximum(b,(mass -m1+0.3 -(a2+0.6)*r)))**2
        schr_9y_min = mass - m0-0.07 + (a0-0.15) * r - (a1+0.08) * (np.maximum(b,(mass -m1-0.3 -(a2-0.6)*r)))**2

        
        
        ax[g,h].set_title(tit)
        ax[g,h].fill_between(m_lst,high_sfr_lst,low_sfr_lst,color = 'red',alpha = 0.2)
        ax[g,h].plot(m_lst,med_sfr_lst,color='red',label = 'Fitted median line')
        ax[g,h].fill_between(massplt,schr_9y_max,schr_9y_min,color='blue',alpha = 0.2)
        ax[g,h].plot(massplt,schr_9y,color = 'blue',label = 'Schreiber et al. 2015')
        ax[g,h].set_xlim(8,12)
        ax[g,h].set_ylim(-3,3)

        
        
        q += 1
        

    fig.text(0.5, 0.04, 'Log10 (Total Stellar Mass $M_\odot$)', ha='center')
    fig.text(0.04, 0.5, 'Log10 (SFR $M_\odot$/yr)', va='center', rotation='vertical')
    fig.legend()
    fig.set_size_inches(9, 9)
    fig.suptitle('Shreiber et al 2015 vs. Fitted Median Line')
    plt.show()










def q_m_m_z(df): #qir vs. stellar mass coloured by metallicity for different redshifts
    
    
    
    
    q = 0
    x = int(len(zlist)/2)
    fig, ax = plt.subplots(2,x)
    for z in zlist:
            qir_dale_m = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')&(df['mstars_tot'] > 10**9)&(df['mstars_tot'] < 10**12) & (df['sfr'] > 0)),'qir_dale']
            qir_bress_m = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')&(df['mstars_tot'] > 10**9)&(df['mstars_tot'] < 10**12)&(df['sfr'] > 0)),'qir_bress']

            mtot = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')&(df['mstars_tot'] > 10**9)&(df['mstars_tot'] < 10**12)& (df['sfr'] > 0)),'mstars_tot']
            gas_metal = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')&(df['mstars_tot'] > 10**9)&(df['mstars_tot'] < 10**12)& (df['sfr'] > 0)),'gas_metal']
            strz = str(z)
            tit = "z = " + strz[:6]
            g = q//x
            h = q%x
         #   plt.hexbin(m_z,sfr_z,C=qir_bress,xscale='log',yscale='log',bins = 'log',reduce_C_function=np.median,cmap = 'rainbow')
         #   plt.show()
            print(g,h)
            print('This is q')
            print(q)
            c = ax[g,h].hexbin(mtot,qir_dale_m,C=gas_metal,xscale='log',gridsize = 30,reduce_C_function=np.median,cmap = 'rainbow',bins='log',vmin = 0, vmax = 1)
            ax[g,h].set_title(tit)
            ax[g,h].set_ylim([2.5,2.8])
            ax[g,h].set_xlim([10**9,10**12])

           # cb = fig.colorbar(c, ax=ax[q])
           # cb.set_label('qir')
            q += 1
        
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Gas Metallicity $Z_{gas}/Z_\odot$')

    fig.text(0.5, 0.04, "Total Stellar Mass $M_\odot$", ha='center')
    fig.text(0.04, 0.5, "qir", va='center', rotation='vertical')
        
    fig.set_size_inches(9, 9)
    fig.suptitle('qir vs. stellar mass coloured by metallicity - Dale Model')

    #plt.title('qir vs. z for different mass bins')
    plt.show()
    
    
    q = 0
    x = int(len(zlist)/2)
    fig, ax = plt.subplots(2,x)
    for z in zlist:
            qir_dale_m = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')&(df['mstars_tot'] > 10**9)&(df['mstars_tot'] < 10**12)),'qir_dale']
            qir_bress_m = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')&(df['mstars_tot'] > 10**9)&(df['mstars_tot'] < 10**12)&(df['qir_bress'] > 0)&(df['qir_bress'] < 4.0)),'qir_bress']

            mtot = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')&(df['mstars_tot'] > 10**9)&(df['mstars_tot'] < 10**12)&(df['qir_bress'] > 0)&(df['qir_bress'] < 4.0)),'mstars_tot']
            gas_metal = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')&(df['mstars_tot'] > 10**9)&(df['mstars_tot'] < 10**12)&(df['qir_bress'] > 0)&(df['qir_bress'] < 4.0)),'gas_metal']
            strz = str(z)
            tit = "z = " + strz[:6]
            g = q//x
            h = q%x
         #   plt.hexbin(m_z,sfr_z,C=qir_bress,xscale='log',yscale='log',bins = 'log',reduce_C_function=np.median,cmap = 'rainbow')
         #   plt.show()
            print(g,h)
            print('This is q')
            print(q)
            c = ax[g,h].hexbin(mtot,qir_bress_m,C=gas_metal,gridsize = 30,xscale='log',reduce_C_function=np.median,cmap = 'rainbow',bins = 'log',vmin= 0,vmax=1)
            ax[g,h].set_title(tit)
            ax[g,h].set_ylim([0,4])
            ax[g,h].set_xlim([10**9,10**12])
           # cb = fig.colorbar(c, ax=ax[q])
           # cb.set_label('qir')
            q += 1
        
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Gas Metallicity $Z_{gas}/Z_\odot$')
    fig.text(0.5, 0.04, "Total Stellar Mass $M_\odot$", ha='center')
    fig.text(0.04, 0.5, "qir", va='center', rotation='vertical')
        

        
    fig.set_size_inches(9, 9)
    fig.suptitle('qir vs. Stellar mass coloured by gas metallicity - Bressan Model')

    #plt.title('qir vs. z for different mass bins')
    plt.show()




def qir_vs_qir(df):
    q = 0
    x = int(len(zlist)/2)
    fig, ax = plt.subplots(2,x)
    for z in zlist:
        g = q//x
        h = q%x
        qir_dale_m = df.loc[(df['z'] == z),'qir_dale']
        qir_bress_m = df.loc[(df['z'] == z),'qir_bress']
        sfr = df.loc[(df['z'] == z),'sfr']
        m_tot = df.loc[(df['z'] == z),'mstars_tot']
        gas_metal = df.loc[((df['z'] == z)),'gas_metal']
       # gas_metal = np.log10(gas_metal)
        min_dale = min(qir_dale_m)
        max_dale = max(qir_dale_m)
       # h = np.linspace(min_dale,max_dale)
       # y = h
        strz = str(z)
        tit = "z = " + strz[:6]
        print(z)
        print("This is the length of qir_dale")
        print(len(qir_dale_m),len(qir_bress_m),len(sfr),len(m_tot))
        print("This is g and h")
        print(g,h)
        print("This is q and x")
        print(q,x)
        x1 = np.linspace(-5,20)
        y = x1
        c = ax[g,h].hexbin(qir_dale_m,qir_bress_m,C=gas_metal,gridsize = 50,cmap = 'rainbow',reduce_C_function=np.median,mincnt = 1,vmin=0,vmax=0.1)
        ax[g,h].plot(x1,y,label = "y = x", color = 'black')
       # ax[g,h].plot(h,y,label="y = x")
        ax[g,h].set_title(tit)
        ax[g,h].set_xlim(2.5,2.8)
        ax[g,h].set_ylim(-2.5,4)
        q += 1
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Gas Metallicity')
    fig.text(0.5, 0.04, "qir - Dale", ha='center')
    fig.text(0.04, 0.5, "qir - Bressan", va='center', rotation='vertical')
    fig.suptitle('Bressan qir vs. Dale qir')
  #  plt.legend()
    plt.show()

    q = 0
    x = int(len(zlist)/2)
    fig, ax = plt.subplots(2,x)
    for z in zlist:
        g = q//x
        h = q%x
        qir_dale_m = df.loc[(df['z'] == z),'qir_dale']
        qir_bress_m = df.loc[(df['z'] == z),'qir_bress']
        sfr = df.loc[(df['z'] == z),'sfr']
        m_tot = df.loc[(df['z'] == z),'mstars_tot']
        gas_mass = df.loc[((df['z'] == z)),'gas_mass']
       # gas_metal = np.log10(gas_metal)
        min_dale = min(qir_dale_m)
        max_dale = max(qir_dale_m)
       # h = np.linspace(min_dale,max_dale)
       # y = h
        strz = str(z)
        tit = "z = " + strz[:6]
        x1 = np.linspace(-5,20)
        y = x1
        ax[g,h].plot(x1,y,label = "y = x", color = 'black')
        print(z)
        print("This is the length of qir_dale")
        print(len(qir_dale_m),len(qir_bress_m),len(sfr),len(m_tot))
        print("This is g and h")
        print(g,h)
        print("This is q and x")
        print(q,x)
        c = ax[g,h].hexbin(qir_dale_m,qir_bress_m,C=gas_mass,gridsize = 50,cmap = 'rainbow',reduce_C_function=np.median,mincnt = 1,vmin=0,vmax=10**5)
       # ax[g,h].plot(h,y,label="y = x")
        ax[g,h].plot(x1,y,label = "y = x", color = 'black')
        ax[g,h].set_title(tit)
        ax[g,h].set_xlim(2.5,2.8)
        ax[g,h].set_ylim(-2.5,4)
        q += 1
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Gas Mass')
    fig.text(0.5, 0.04, "qir - Dale", ha='center')
    fig.text(0.04, 0.5, "qir - Bressan", va='center', rotation='vertical')
    fig.suptitle('Bressan qir vs. Dale qir')
  #  plt.legend()
    plt.show()




    
    q = 0
    x = int(len(zlist)/2)
    fig, ax = plt.subplots(2,x)
    
    for z in zlist:
        g = q//x
        h = q%x
        qir_dale_m = df.loc[(df['z'] == z),'qir_dale']
        qir_bress_m = df.loc[(df['z'] == z),'qir_bress']
        sfr = df.loc[(df['z'] == z),'sfr']
        m_tot = df.loc[(df['z'] == z),'mstars_tot']
        min_dale = min(qir_dale_m)
        max_dale = max(qir_dale_m)
    #    x = np.linspace(min_dale,max_dale)
   #     y = x
        strz = str(z)
        tit = "z = " + strz[:6]
        c = ax[g,h].hexbin(qir_dale_m,qir_bress_m,C=sfr,reduce_C_function=np.median,gridsize = 50,bins = 'log',cmap = 'rainbow',mincnt = 1,vmin = 0, vmax = 10**3)
     #   ax[g,h].plot(x,y,label="y = x")
        ax[g,h].set_title(tit)
        ax[g,h].set_xlim(2.5,2.8)
        ax[g,h].set_ylim(-2.5,4)
        q += 1
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('SFR')
    fig.text(0.5, 0.04, "qir - Dale", ha='center')
    fig.text(0.04, 0.5, "qir - Bressan", va='center', rotation='vertical')
    fig.suptitle('Bressan qir vs. Dale qir')
  #  plt.legend()
    plt.show()
    
    q = 0
    x = int(len(zlist)/2)
    fig, ax = plt.subplots(2,x)
    
    for z in zlist:
        g = q//x
        h = q%x
        qir_dale_m = df.loc[(df['z'] == z),'qir_dale']
        qir_bress_m = df.loc[(df['z'] == z),'qir_bress']
        sfr = df.loc[(df['z'] == z),'sfr']
        m_tot = df.loc[(df['z'] == z),'mstars_tot']
        min_dale = min(qir_dale_m)
        max_dale = max(qir_dale_m)
        x = np.linspace(min_dale,max_dale)
        y = x
        strz = str(z)
       
        tit = "z = " + strz[:6]
        c = ax[g,h].hexbin(qir_dale_m,qir_bress_m,C=m_tot,reduce_C_function=np.median,gridsize = 50,bins = 'log',cmap = 'rainbow',mincnt = 1,vmin = 10**8, vmax = 10**12)
        ax[g,h].plot(x,y,label="y = x")
        ax[g,h].set_title(tit)        
        ax[g,h].set_xlim(2.5,2.8)
        ax[g,h].set_ylim(-2.5,4)
        q += 1
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Stellar Mass')
    fig.text(0.5, 0.04, "qir - Dale", ha='center')
    fig.text(0.04, 0.5, "qir - Bressan", va='center', rotation='vertical')
    fig.suptitle('Bressan qir vs. Dale qir')
  #  plt.legend()
    plt.show()
    
    
    q = 0
    x = int(len(zlist)/2)
    fig, ax = plt.subplots(2,x)
    for z in zlist:
        g = q//x
        h = q%x
        sfr = df.loc[(df['z'] == z),'sfr']
        qir_bress_m = df.loc[(df['z'] == z),'qir_bress']
        strz = str(z)
        tit = "z = " + strz[:6]
        c = ax[g,h].hexbin(qir_bress_m,sfr,gridsize = 50,bins = 'log',yscale = 'log',cmap = 'rainbow',mincnt = 1)
        ax[g,h].set_title(tit)
        ax[g,h].set_xlim(2.5,2.8)
        ax[g,h].set_xlim(-2.5,4)
        q += 1
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Frequency')
    fig.text(0.5, 0.04, "qir - Bressan", ha='center')
    fig.text(0.04, 0.5, "SFR", va='center', rotation='vertical')
    fig.suptitle('SFR vs. Bressan qir')
#    plt.legend()
    plt.show()
    
    q = 0
    x = int(len(zlist)/2)
    fig, ax = plt.subplots(2,x)
    for z in zlist:
        g = q//x
        h = q%x
        sfr = df.loc[(df['z'] == z),'sfr']
        m_tot = df.loc[(df['z'] == z),'mstars_tot']
        qir_bress_m = df.loc[(df['z'] == z),'qir_bress']
        strz = str(z)
        tit = "z = " + strz[:6]
        c = ax[g,h].hexbin(qir_bress_m,m_tot,C=sfr,gridsize = 50,bins = 'log',yscale = 'log',cmap = 'rainbow',mincnt = 1)
        ax[g,h].set_title(tit)
        ax[g,h].set_xlim(-2.5,4)
        q += 1
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('SFR')
    fig.text(0.5, 0.04, "qir - Bressan", ha='center')
    fig.text(0.04, 0.5, "Stellar Mass", va='center', rotation='vertical')
    fig.suptitle('Stellar Mass vs. Bressan qir')
    plt.show()

def sfr_rad_lum(df):
    
    sfr = df.loc[df['z'] < 0.4, 'sfr']
    bress_rad_lum = df.loc[df['z'] < 0.4, 'bress_rad_lum']
    dale_rad_lum = df.loc[df['z'] < 0.4, 'dale_rad_lum']
    
    xpaper = np.linspace(3*10**21,4*10**23,num=2, endpoint = True)

    yfree_paper = 10**(0.66*np.log10(xpaper) - 14.02)
    yfixed_paper = 10**(np.log10(xpaper) -21.62)
    ystack_paper = 10**(0.63*np.log10(xpaper) - 13.27)

    x_extp = np.linspace(1e8,3*10**21)

    yfree_extp = 10**(0.66*np.log10(x_extp) - 14.02)
    yfixed_extp = 10**(np.log10(x_extp) -21.62)
    ystack_extp = 10**(0.63*np.log10(x_extp) - 13.27)
    
    plt.hexbin(bress_rad_lum,sfr,xscale = 'log',yscale='log',bins = 'log',cmap="Greys",mincnt=1)
    plt.title("Bressan - Radio Luminosity vs. SFR")
    plt.xscale = 'log'
    plt.yscale = 'log'
    plt.plot(xpaper,yfree_paper,'blue',label = 'Free Fit')
    plt.plot(xpaper,yfixed_paper,'red',label = 'Fixed Fit')
    plt.plot(xpaper,ystack_paper,'purple',label = 'Stack Fit')
    plt.plot(x_extp,yfree_extp,'blue',linestyle='dashed')
    plt.plot(x_extp,yfixed_extp,'red',linestyle='dashed')
    plt.plot(x_extp,ystack_extp,'purple',linestyle='dashed')
    plt.xlabel("Radio Luminosity (W/Hz)")
    plt.ylabel("SFR (M☉/yr)")
  #  plt.legend()
    plt.ylim(1e-3,1e3)
    plt.xlim(1e12,1e24)
    plt.colorbar()
    plt.show()
    
    
    plt.hexbin(dale_rad_lum,sfr,xscale = 'log',yscale = 'log',bins = 'log',cmap='Greys',mincnt=1)
    plt.xscale = 'log'
    plt.yscale = 'log'
    plt.plot(xpaper,yfree_paper,'blue',label = 'Free Fit')
    plt.plot(xpaper,yfixed_paper,'red',label = 'Fixed Fit')
    plt.plot(xpaper,ystack_paper,'purple',label = 'Stack Fit')
    plt.plot(x_extp,yfree_extp,'blue',linestyle='dashed')
    plt.plot(x_extp,yfixed_extp,'red',linestyle='dashed')
    plt.plot(x_extp,ystack_extp,'purple',linestyle='dashed')
    plt.colorbar()
    plt.title("Dale - Radio Luminosity vs. SFR")
    plt.xlabel("Radio Luminosity (W/Hz)")
    plt.ylabel("SFR (M☉/yr)")
    plt.ylim(1e-3,1e3)
    plt.xlim(1e12,1e24)
  #  plt.legend()
    plt.show()
    
    q = 0
    x = int(len(zlist)/2)
    fig, ax = plt.subplots(2,x)
    g = q//x
    h = q%x
    for z in zlist:

        g = q//x
        h = q%x
        bress_rad_lum = df.loc[((df['z'] == z)&(df['bress_rad_lum'] > 1e12)&(df['bress_rad_lum'] < 1e24) & (df['sfr'] > 1e-3) & (df['sfr'] < 1e3)),'bress_rad_lum']
        sfr = df.loc[((df['z'] == z)&(df['bress_rad_lum'] > 1e12)&(df['bress_rad_lum'] < 1e24) & (df['sfr'] > 1e-3) & (df['sfr'] < 1e3)),'sfr']

        strz = str(z)
        tit = "z = " + strz[:6]
        
        c = ax[g,h].hexbin(bress_rad_lum,sfr,xscale = 'log',yscale = 'log',bins = 'log',cmap='Greys',mincnt=1,vmin = 0, vmax = 10, gridsize = 30)
        ax[g,h].set_title(tit)
        ax[g,h].set_ylim(1e-3,1e3)
        ax[g,h].set_xlim(1e12,1e24)
        ax[g,h].plot(xpaper,yfree_paper,'blue',label = 'Free Fit')
        ax[g,h].plot(xpaper,yfixed_paper,'red',label = 'Fixed Fit')
        ax[g,h].plot(xpaper,ystack_paper,'purple',label = 'Stack Fit')
        ax[g,h].plot(x_extp,yfree_extp,'blue',linestyle='dashed')
        ax[g,h].plot(x_extp,yfixed_extp,'red',linestyle='dashed')
        ax[g,h].plot(x_extp,ystack_extp,'purple',linestyle='dashed')
        q += 1
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Frequency')
    fig.text(0.5, 0.04, "Radio Luminosity (W/Hz)", ha='center')
    fig.text(0.04, 0.5, "SFR", va='center', rotation='vertical')
    fig.suptitle('SFR vs. Radio Luminosity - Bressan')
  #  plt.legend()
    plt.show()
    
    q = 0
    x = int(len(zlist)/2)
    fig, ax = plt.subplots(2,x)
    z = 0 
    for z in zlist:
        dale_rad_lum = df.loc[((df['z'] == z)&(df['dale_rad_lum'] > 1e12)&(df['dale_rad_lum'] < 1e24) & (df['sfr'] > 1e-3) & (df['sfr'] < 1e3)),'dale_rad_lum']
        sfr = df.loc[((df['z'] == z)&(df['dale_rad_lum'] > 1e12)&(df['dale_rad_lum'] < 1e24) & (df['sfr'] > 1e-3) & (df['sfr'] < 1e3)),'sfr']

        g = q//x
        h = q%x


        strz = str(z)
        tit = "z = " + strz[:6]
        print(z)
        print("Here is the first 4 elements of dale rad lum")
        print(dale_rad_lum[:4]) 
        c = ax[g,h].hexbin(dale_rad_lum,sfr,xscale = 'log',yscale = 'log',bins = 'log',cmap='Greys',mincnt=1,vmin = 0, vmax = 5, gridsize = 30)
        ax[g,h].set_title(tit)
        ax[g,h].set_ylim(1e-3,1e3)
        ax[g,h].set_xlim(1e12,1e24)
        ax[g,h].plot(xpaper,yfree_paper,'blue',label = 'Free Fit')
        ax[g,h].plot(xpaper,yfixed_paper,'red',label = 'Fixed Fit')
        ax[g,h].plot(xpaper,ystack_paper,'purple',label = 'Stack Fit')
        ax[g,h].plot(x_extp,yfree_extp,'blue',linestyle='dashed')
        ax[g,h].plot(x_extp,yfixed_extp,'red',linestyle='dashed')
        ax[g,h].plot(x_extp,ystack_extp,'purple',linestyle='dashed')
        q += 1
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Frequency')
    fig.text(0.5, 0.04, "Radio Luminosity (W/Hz)", ha='center')
    fig.text(0.04, 0.5, "SFR", va='center', rotation='vertical')
    fig.suptitle('SFR vs. Radio Luminosity - Dale')
  #  plt.legend()
    plt.show()
    
    
def rad_lum_func_plt_2(df,h0,volh):
    vol = volh/h0**3
    mlow = 0
    mupp = 40
    dm = 0.25
    mbins = np.arange(mlow, mupp, dm)
    xmf = mbins + dm/2.0 #setting up the bins
   
    fig, ax = plt.subplots(3,1)

    df2 = pd.read_csv("bonata_data_z0.csv")
    
    bonatoy = df2.loc[(df2['ref'] == 'Bonato2020'),'log_phi']
    bonatox = df2.loc[(df2['ref'] == 'Bonato2020'),'log_L_1.4']
    bonatoerr_up = df2.loc[(df2['ref'] == 'Bonato2020'),'err_sup_phi']
    bonatoerr_down = df2.loc[(df2['ref'] == 'Bonato2020'),'err_inf_phi']
    
    butlery = df2.loc[(df2['ref'] == 'Butler2019'),'log_phi']
    butlerx = df2.loc[(df2['ref'] == 'Butler2019'),'log_L_1.4']
    butlererr_up = df2.loc[(df2['ref'] == 'Butler2019'),'err_sup_phi']
    butlererr_down = df2.loc[(df2['ref'] == 'Butler2019'),'err_inf_phi']

    ocrany = df2.loc[(df2['ref'] == 'Ocran2020'),'log_phi']
    ocranx = df2.loc[(df2['ref'] == 'Ocran2020'),'log_L_1.4']
    ocranerr_up = df2.loc[(df2['ref'] == 'Ocran2020'),'err_sup_phi']
    ocranerr_down = df2.loc[(df2['ref'] == 'Ocran2020'),'err_inf_phi']
    
    novaky = df2.loc[(df2['ref'] == 'Novak2017'),'log_phi']
    novakx = df2.loc[(df2['ref'] == 'Novak2017'),'log_L_1.4']
    novakerr_up = df2.loc[(df2['ref'] == 'Novak2017'),'err_sup_phi']
    novakerr_down = df2.loc[(df2['ref'] == 'Novak2017'),'err_inf_phi']
    
    dale_rad_lum = df.loc[(df['z'] < 0.1),'dale_rad_lum']
    
    dlum = np.log10(dale_rad_lum) #logging the dataset

    dhist,_= np.histogram(dlum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins
    
    print("This is dhist")
    print(dhist)
    
    dhist = dhist/vol/dm
    
    
    bress_rad_lum = df.loc[(df['z'] < 0.1),'bress_rad_lum']

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(xmf,np.log10(bhist),False,False)
    ax[0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    

    ax[0].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="o",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020')
    ax[0].errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019')
    ax[0].errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020')
    ax[0].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020')
    ax[0].set_title("z = 0")
    ax[0].xlabel("Log$_{10}$($L_{1.4Ghz}$) (W/Hz)")
    ax[0].ylabel("$Log_{10}$(Φ) ($Mpc^{-3} dex^{-1}$)")
    ax[0].set_xlim(19,25)
    ax[0].set_ylim(-7,0)
    #ax[0,0].legend()
    #ax[0,0].show()
    
    

    
    df2 = pd.read_csv("bonata_data_z1.csv")
    
    bonatoy = df2.loc[(df2['ref'] == 'Bonato2020'),'log_phi']
    bonatox = df2.loc[(df2['ref'] == 'Bonato2020'),'log_L_1.4']
    bonatoerr_up = df2.loc[(df2['ref'] == 'Bonato2020'),'err_sup_phi']
    bonatoerr_down = df2.loc[(df2['ref'] == 'Bonato2020'),'err_inf_phi']

    ocrany = df2.loc[(df2['ref'] == 'Ocran2020'),'log_phi']
    ocranx = df2.loc[(df2['ref'] == 'Ocran2020'),'log_L_1.4']
    ocranerr_up = df2.loc[(df2['ref'] == 'Ocran2020'),'err_sup_phi']
    ocranerr_down = df2.loc[(df2['ref'] == 'Ocran2020'),'err_inf_phi']
    
    novaky = df2.loc[(df2['ref'] == 'Novak2017'),'log_phi']
    novakx = df2.loc[(df2['ref'] == 'Novak2017'),'log_L_1.4']
    novakerr_up = df2.loc[(df2['ref'] == 'Novak2017'),'err_sup_phi']
    novakerr_down = df2.loc[(df2['ref'] == 'Novak2017'),'err_inf_phi']
    
    dale_rad_lum = df.loc[((df['z'] < 1.2)&(df['z'] > 0.8)),'dale_rad_lum']
    
    dlum = np.log10(dale_rad_lum) #logging the dataset

    dhist= np.histogram(dlum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins
    
    print("This is dhist")
    print(dhist)
    
    dhist = dhist[0]/vol/dm
    
    bress_rad_lum = df.loc[(df['z'] == 0.909822023685613),'bress_rad_lum']

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(xmf,np.log10(bhist),False,False)
    ax[1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    

    ax[1].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="o",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020')
   # ax[1].errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019')
    ax[1].errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020')
    ax[1].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020')
    ax[1].set_title("z = 1")
    ax[1].xlabel("Log$_{10}$($L_{1.4Ghz}$) (W/Hz)")
    ax[1].ylabel("$Log_{10}$(Φ) ($Mpc^{-3} dex^{-1}$)")
    ax[1].set_xlim(19,25)
    ax[1].set_ylim(-7,0)

    
    df2 = pd.read_csv("bonata_data_z2.csv")
    
    bonatoy = df2.loc[(df2['ref'] == 'Bonato2020'),'log_phi']
    bonatox = df2.loc[(df2['ref'] == 'Bonato2020'),'log_L_1.4']
    bonatoerr_up = df2.loc[(df2['ref'] == 'Bonato2020'),'err_sup_phi']
    bonatoerr_down = df2.loc[(df2['ref'] == 'Bonato2020'),'err_inf_phi']
    
    novaky = df2.loc[(df2['ref'] == 'Novak2017'),'log_phi']
    novakx = df2.loc[(df2['ref'] == 'Novak2017'),'log_L_1.4']
    novakerr_up = df2.loc[(df2['ref'] == 'Novak2017'),'err_sup_phi']
    novakerr_down = df2.loc[(df2['ref'] == 'Novak2017'),'err_inf_phi']
    
    bress_rad_lum = df.loc[((df['z'] < 2.2)&(df['z'])),'bress_rad_lum']

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(xmf,np.log10(bhist),False,False)
    ax[2].plot(mid_rad_lst,med_fir_lst,'black')
    ax[2].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[2].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    

    ax[2].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="o",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020')
   # ax[1].errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019')
    ax[2].errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020')
    ax[2].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020')
    ax[2].set_title("z = 2")
    ax[2].xlabel("Log$_{10}$($L_{1.4Ghz}$) (W/Hz)")
    ax[2].ylabel("$Log_{10}$(Φ) ($Mpc^{-3} dex^{-1}$)")
    ax[2].set_xlim(19,25)
    ax[2].set_ylim(-7,0)

    plt.legend()
    plt.show()
    
    
    
def rad_lum_func_plt(df,h0,volh):
    vol = volh/h0**3
    mlow = 0
    mupp = 40
    dm = 0.25
    mbins = np.arange(mlow, mupp, dm)
    xmf = mbins + dm/2.0 #setting up the bins
   
    fig, ax = plt.subplots(3,1)

    df2 = pd.read_csv("bonata_data_z0.csv")
    
    bonatoy = df2.loc[(df2['ref'] == 'Bonato2020'),'log_phi']
    bonatox = df2.loc[(df2['ref'] == 'Bonato2020'),'log_L_1.4']
    bonatoerr_up = df2.loc[(df2['ref'] == 'Bonato2020'),'err_sup_phi']
    bonatoerr_down = df2.loc[(df2['ref'] == 'Bonato2020'),'err_inf_phi']
    
    butlery = df2.loc[(df2['ref'] == 'Butler2019'),'log_phi']
    butlerx = df2.loc[(df2['ref'] == 'Butler2019'),'log_L_1.4']
    butlererr_up = df2.loc[(df2['ref'] == 'Butler2019'),'err_sup_phi']
    butlererr_down = df2.loc[(df2['ref'] == 'Butler2019'),'err_inf_phi']

    ocrany = df2.loc[(df2['ref'] == 'Ocran2020'),'log_phi']
    ocranx = df2.loc[(df2['ref'] == 'Ocran2020'),'log_L_1.4']
    ocranerr_up = df2.loc[(df2['ref'] == 'Ocran2020'),'err_sup_phi']
    ocranerr_down = df2.loc[(df2['ref'] == 'Ocran2020'),'err_inf_phi']
    
    novaky = df2.loc[(df2['ref'] == 'Novak2017'),'log_phi']
    novakx = df2.loc[(df2['ref'] == 'Novak2017'),'log_L_1.4']
    novakerr_up = df2.loc[(df2['ref'] == 'Novak2017'),'err_sup_phi']
    novakerr_down = df2.loc[(df2['ref'] == 'Novak2017'),'err_inf_phi']
    
    dale_rad_lum = df.loc[(df['z'] < 0.1),'dale_rad_lum']
    
    dlum = np.log10(dale_rad_lum) #logging the dataset

    dhist,_= np.histogram(dlum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins
    
    print("This is dhist")
    print(dhist)
    
    dhist = dhist/vol/dm
    
    
    bress_rad_lum = df.loc[(df['z'] < 0.1),'bress_rad_lum']

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(xmf,np.log10(bhist),False,False)
    ax[0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    

    ax[0].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="o",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020')
    ax[0].errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019')
    ax[0].errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020')
    ax[0].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020')
    ax[0].set_title("z = 0")
    ax[0].xlabel("Log$_{10}$($L_{1.4Ghz}$) (W/Hz)")
    ax[0].ylabel("$Log_{10}$(Φ) ($Mpc^{-3} dex^{-1}$)")
    ax[0].set_xlim(19,25)
    ax[0].set_ylim(-7,0)
    #ax[0,0].legend()
    #ax[0,0].show()
    
    

    
    df2 = pd.read_csv("bonata_data_z1.csv")
    
    bonatoy = df2.loc[(df2['ref'] == 'Bonato2020'),'log_phi']
    bonatox = df2.loc[(df2['ref'] == 'Bonato2020'),'log_L_1.4']
    bonatoerr_up = df2.loc[(df2['ref'] == 'Bonato2020'),'err_sup_phi']
    bonatoerr_down = df2.loc[(df2['ref'] == 'Bonato2020'),'err_inf_phi']

    ocrany = df2.loc[(df2['ref'] == 'Ocran2020'),'log_phi']
    ocranx = df2.loc[(df2['ref'] == 'Ocran2020'),'log_L_1.4']
    ocranerr_up = df2.loc[(df2['ref'] == 'Ocran2020'),'err_sup_phi']
    ocranerr_down = df2.loc[(df2['ref'] == 'Ocran2020'),'err_inf_phi']
    
    novaky = df2.loc[(df2['ref'] == 'Novak2017'),'log_phi']
    novakx = df2.loc[(df2['ref'] == 'Novak2017'),'log_L_1.4']
    novakerr_up = df2.loc[(df2['ref'] == 'Novak2017'),'err_sup_phi']
    novakerr_down = df2.loc[(df2['ref'] == 'Novak2017'),'err_inf_phi']
    
    dale_rad_lum = df.loc[((df['z'] < 1.2)&(df['z'] > 0.8)),'dale_rad_lum']
    
    dlum = np.log10(dale_rad_lum) #logging the dataset

    dhist= np.histogram(dlum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins
    
    print("This is dhist")
    print(dhist)
    
    dhist = dhist[0]/vol/dm
    
    bress_rad_lum = df.loc[(df['z'] == 0.909822023685613),'bress_rad_lum']

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(xmf,np.log10(bhist),False,False)
    ax[1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    

    ax[1].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="o",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020')
   # ax[1].errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019')
    ax[1].errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020')
    ax[1].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020')
    ax[1].set_title("z = 1")
    ax[1].xlabel("Log$_{10}$($L_{1.4Ghz}$) (W/Hz)")
    ax[1].ylabel("$Log_{10}$(Φ) ($Mpc^{-3} dex^{-1}$)")
    ax[1].set_xlim(19,25)
    ax[1].set_ylim(-7,0)

    
    df2 = pd.read_csv("bonata_data_z2.csv")
    
    bonatoy = df2.loc[(df2['ref'] == 'Bonato2020'),'log_phi']
    bonatox = df2.loc[(df2['ref'] == 'Bonato2020'),'log_L_1.4']
    bonatoerr_up = df2.loc[(df2['ref'] == 'Bonato2020'),'err_sup_phi']
    bonatoerr_down = df2.loc[(df2['ref'] == 'Bonato2020'),'err_inf_phi']
    
    novaky = df2.loc[(df2['ref'] == 'Novak2017'),'log_phi']
    novakx = df2.loc[(df2['ref'] == 'Novak2017'),'log_L_1.4']
    novakerr_up = df2.loc[(df2['ref'] == 'Novak2017'),'err_sup_phi']
    novakerr_down = df2.loc[(df2['ref'] == 'Novak2017'),'err_inf_phi']
    
    bress_rad_lum = df.loc[((df['z'] < 2.2)&(df['z'])),'bress_rad_lum']

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(xmf,np.log10(bhist),False,False)
    ax[2].plot(mid_rad_lst,med_fir_lst,'black')
    ax[2].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[2].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    

    ax[2].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="o",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020')
   # ax[1].errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019')
    ax[2].errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020')
    ax[2].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020')
    ax[2].set_title("z = 2")
    ax[2].xlabel("Log$_{10}$($L_{1.4Ghz}$) (W/Hz)")
    ax[2].ylabel("$Log_{10}$(Φ) ($Mpc^{-3} dex^{-1}$)")
    ax[2].set_xlim(19,25)
    ax[2].set_ylim(-7,0)

    plt.legend()
    plt.show()
    
    
    
    df2 = pd.read_csv("bonata_data_z3.csv")
    
    novaky = df2.loc[(df2['ref'] == 'Novak2017'),'log_phi']
    novakx = df2.loc[(df2['ref'] == 'Novak2017'),'log_L_1.4']
    novakerr_up = df2.loc[(df2['ref'] == 'Novak2017'),'err_sup_phi']
    novakerr_down = df2.loc[(df2['ref'] == 'Novak2017'),'err_inf_phi']
    
    dale_rad_lum = df.loc[((df['z'] < 3.2)&(df['z'] > 2.8)),'dale_rad_lum']
    
    dlum = np.log10(dale_rad_lum) #logging the dataset

    dhist= np.histogram(dlum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins
    
    dhist = dhist[0]/vol/dm
    
    
    bress_rad_lum = df.loc[((df['z'] < 3.2)&(df['z'] > 2.8)),'bress_rad_lum']

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm

    ax[1,1].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],ecolor='red',fmt='none',label="_nolegend_")
    ax[1,1].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],ecolor='green',fmt='none',label="_nolegend_")
    ax[1,1].scatter(bonatox,bonatoy,c='red',label = 'Bonato 2020')
    ax[1,1].scatter(novakx,novaky,c='green',label = 'Novak 2020')
    ax[1,1].plot(xmf, np.log10(dhist), linestyle='dashed', color='goldenrod', label='Dale model')
    ax[1,1].plot(xmf, np.log10(bhist), linestyle='dashed', color='silver', label='Bressan model')
    ax[1,1].set_title("z = 3.0")
   # ax[0].xlabel("Log10(L_1.4Ghz) (W/Hz)")
   # ax[1,0].ylabel("Log10(Φ) (Mpc^-3 dex^-1)")
    ax[1,1].set_xlim(19,25)
    ax[1,1].set_ylim(-7,0)
   # ax[1,0].legend()
   # ax[1,0].show()
    
    
    
    df2 = pd.read_csv("bonata_data_z4.csv")
    
    novaky = df2.loc[(df2['ref'] == 'Novak2017'),'log_phi']
    novakx = df2.loc[(df2['ref'] == 'Novak2017'),'log_L_1.4']
    novakerr_up = df2.loc[(df2['ref'] == 'Novak2017'),'err_sup_phi']
    novakerr_down = df2.loc[(df2['ref'] == 'Novak2017'),'err_inf_phi']
    
    dale_rad_lum = df.loc[((df['z'] < 4.2)&(df['z'] > 3.8)),'dale_rad_lum']
    
    dlum = np.log10(dale_rad_lum) #logging the dataset

    dhist= np.histogram(dlum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins
    
    dhist = dhist[0]/vol/dm
    
    
    bress_rad_lum = df.loc[((df['z'] < 4.2)&(df['z'] > 3.8)),'bress_rad_lum']

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm

    ax[1,2].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],ecolor='green',fmt='none',label="_nolegend_")
    ax[1,2].scatter(novakx,novaky,c='green',label = 'Novak 2020')
    ax[1,2].plot(xmf, np.log10(dhist), linestyle='dashed', color='goldenrod', label='Dale model')
    ax[1,2].plot(xmf, np.log10(bhist), linestyle='dashed', color='silver', label='Bressan model')
    ax[1,2].set_title("z = 4.0")
#    ax[1,1].xlabel("Log10(L_1.4Ghz) (W/Hz)")
#    ax[1,1].ylabel("Log10(Φ) (Mpc^-3 dex^-1)")
    ax[1,2].set_xlim(19,25)
    ax[1,2].set_ylim(-7,0)
    #ax[1,1].legend()
    #ax[1,1].show()
    
    
    df2 = pd.read_csv("bonata_data_z5.csv")
    
    novaky = df2.loc[(df2['ref'] == 'Novak2017'),'log_phi']
    novakx = df2.loc[(df2['ref'] == 'Novak2017'),'log_L_1.4']
    novakerr_up = df2.loc[(df2['ref'] == 'Novak2017'),'err_sup_phi']
    novakerr_down = df2.loc[(df2['ref'] == 'Novak2017'),'err_inf_phi']
    
    dale_rad_lum = df.loc[((df['z'] < 5.2)&(df['z'] > 4.8)),'dale_rad_lum']
    
    dlum = np.log10(dale_rad_lum) #logging the dataset

    dhist= np.histogram(dlum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins
    
    dhist = dhist[0]/vol/dm
    
    
    bress_rad_lum = df.loc[((df['z'] < 5.2)&(df['z'] > 4.8)),'bress_rad_lum']

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm

    ax[1,3].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],ecolor='green',fmt='none',label="_nolegend_")
    ax[1,3].scatter(novakx,novaky,c='green',label = 'Novak 2020')
    ax[1,3].scatter(-100,-100,c='red',label = 'Bonato 2020')
    ax[1,3].scatter(-100,-100,c='blue',label = 'Butler 2019')
    ax[1,3].scatter(-100,-100,c='purple',label = 'Ocran 2020')
   # ax[1,2].scatter(-100,-100,c='green',label = 'Novak 2020')
    ax[1,3].plot(xmf, np.log10(dhist), linestyle='dashed', color='goldenrod', label='Dale model')
    ax[1,3].plot(xmf, np.log10(bhist), linestyle='dashed', color='silver', label='Bressan model')
    ax[1,3].set_title("z = 5.0")
 #   ax[1,2].xlabel("Log10(L_1.4Ghz) (W/Hz)")
 #   ax[1,2].ylabel("Log10(Φ) (Mpc^-3 dex^-1)")
    ax[1,3].set_xlim(19,25)
    ax[1,3].set_ylim(-7,0)
    plt.legend()
    fig.text(0.5, 0.04, "Log10(L_1.4Ghz) (W/Hz)", ha='center')
    fig.text(0.04, 0.5, "Log10(Φ) (Mpc^-3 dex^-1)", va='center', rotation='vertical')
    fig.suptitle('Radio Luminosity function for different redshifts')
    plt.show()


def ms_qir_m_dale(df,z):
    pow_lst = list((np.arange(0,30)/7)+8)
    qd_lst = []
    med_qd_lst = []
    low_qd_lst= []
    high_qd_lst = []
    
    qb_lst = []
    med_qb_lst = []
    low_qb_lst= []
    high_qb_lst = []
    print("This is the pow_lst")
    print(pow_lst)
    for i in range(len(pow_lst)-1):
        m = pow_lst[i]
        n = pow_lst[i+1]
        qd = df.loc[((df['z'] == z)&(df['mstars_tot'] > 10**(m))&(df['mstars_tot'] < 10**(n))&(df['sfr/q'] == 'sf')),'qir_dale']
        try:
            low_qd_lst.append(np.percentile(qd,16))
            high_qd_lst.append(np.percentile(qd,84))
        except:
            continue
        med_qd = np.median(qd)
        med_qd_lst.append(med_qd)
        mid_qd = (n/2+m/2)
        qd_lst.append(mid_qd)
        
     #   qb = df.loc[((df['z'] == z)&(df['mstars_tot'] > 10**(m))&(df['mstars_tot'] < 10**(n))&(df['sfr/q'] == 'sf')),'qir_bress']
     #   try:
     #       low_qb_lst.append(np.percentile(qb,16))
     #       high_qb_lst.append(np.percentile(qb,84))
     #   except:
     #       continue
     #       
     #   med_qb = np.median(qb)
     #   med_qb_lst.append(med_qb)
     #   mid_qb = n/2+m/2
     #   qb_lst.append(mid_qb)
        
        
        
    return med_qd_lst, high_qd_lst, low_qd_lst, qd_lst, med_qb_lst, high_qb_lst, low_qb_lst, qb_lst

def ms_qir_m_bress(df,z):
    pow_lst = list((np.arange(0,30)/7)+8)
    qd_lst = []
    med_qd_lst = []
    low_qd_lst= []
    high_qd_lst = []

    qb_lst = []
    med_qb_lst = []
    low_qb_lst= []
    high_qb_lst = []
    print("This is the pow_lst")
    print(pow_lst)
    for i in range(len(pow_lst)-1):
        m = pow_lst[i]
        n = pow_lst[i+1]
       # qd = df.loc[((df['z'] == z)&(df['mstars_tot'] > 10**(m))&(df['mstars_tot'] < 10**(n))&(df['sfr/q'] == 'sf')),'qir_dale']
       # try:
       #     low_qd_lst.append(np.percentile(qd,16))
       #     high_qd_lst.append(np.percentile(qd,84))
       # except:
       #     continue
       # med_qd = np.median(qd)
       # med_qd_lst.append(med_qd)
       # mid_qd = (n/2+m/2)
       # qd_lst.append(mid_qd)

        qb = df.loc[((df['z'] == z)&(df['mstars_tot'] > 10**(m))&(df['mstars_tot'] < 10**(n))&(df['sfr/q'] == 'sf')),'qir_bress']
        try:
            low_qb_lst.append(np.percentile(qb,16))
            high_qb_lst.append(np.percentile(qb,84))
        except:
            continue
            
        med_qb = np.median(qb)
        med_qb_lst.append(med_qb)
        mid_qb = n/2+m/2
        qb_lst.append(mid_qb)



    return med_qd_lst, high_qd_lst, low_qd_lst, qd_lst, med_qb_lst, high_qb_lst, low_qb_lst, qb_lst




def qir_with_m_plt(df):
#    zlist = [0,0.194738848008908]
    q = 0
    x = int(len(zlist)/2)
    
    fig, ax = plt.subplots(2,x)
    for z in zlist:
        print("This is q and z")
        print(q,z)
        g = q//x
        h = q%x
        med_qd_lst, high_qd_lst, low_qd_lst, qd_lst, med_qb_lst, high_qb_lst, low_qb_lst, qb_lst = ms_qir_m_dale(df,z)
    
        qir_dale_m = df.loc[((df['qir_dale'] < 2.8) & (df['qir_dale'] > 2.5) & (df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)),'qir_dale']
       # qir_bress_m = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)),'qir_bress']
        mstars_tot = df.loc[((df['qir_dale'] < 2.8) & (df['qir_dale'] > 2.5)&(df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)),'mstars_tot']
        gas_metal = df.loc[((df['qir_dale'] < 2.8) & (df['qir_dale'] > 2.5)&(df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)),'gas_metal']
        mstars_tot = np.log10(mstars_tot)
        strz = str(z)
        tit = 'z = ' + strz[:4]
        c = ax[g,h].hexbin(mstars_tot,qir_dale_m,C=gas_metal,mincnt = 1,cmap='rainbow', bins = 'log',reduce_C_function=np.median,gridsize = 30,vmin = 0.1, vmax = 0.5)
        ax[g,h].plot(qd_lst,med_qd_lst,label = "Median Line", color = 'black')
        ax[g,h].plot(qd_lst,high_qd_lst, color = 'black',label = "_nolegend_",linestyle = 'dashed')
        ax[g,h].plot(qd_lst,low_qd_lst, color = 'black',label = "_nolegend_",linestyle = 'dashed')
        ax[g,h].set_title(tit)
     #   plt.xlabel("Log Stellar mass - Solar masses")
      #  plt.ylabel("qir")
        ax[g,h].set_xlim(8,12)
        ax[g,h].set_ylim(2.5,2.8)
        q+=1
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Gas Metallicity')
    fig.text(0.5, 0.04, "Stellar Mass $M_\odot$", ha='center')
    fig.text(0.04, 0.5, "qir", va='center', rotation='vertical')
    fig.suptitle('qir vs. Stellar Mass for Star Forming Galaxies- Dale')
    plt.show()
    
    fig, ax = plt.subplots(2,x)
    q = 0
    x = int(len(zlist)/2)
    z = 0
    for z in zlist:
        g = q//x
        h = q%x
        med_qd_lst, high_qd_lst, low_qd_lst, qd_lst, med_qb_lst, high_qb_lst, low_qb_lst, qb_lst = ms_qir_m_bress(df,z)
        
      #  qir_dale_m = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)),'qir_dale']
        qir_bress_m = df.loc[((df['qir_bress'] > 2.0) & (df['qir_bress'] < 3.5) & (df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)),'qir_bress']
        mstars_tot = df.loc[((df['qir_bress'] > 2.0) & (df['qir_bress'] < 3.5) & (df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)),'mstars_tot']
        gas_metal = df.loc[((df['qir_bress'] > 2.0) & (df['qir_bress'] < 3.5) & (df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)),'gas_metal']
        mstars_tot = np.log10(mstars_tot)
        strz = str(z)
        tit = 'z = ' + strz[:4]
        c = ax[g,h].hexbin(mstars_tot,qir_bress_m,C=gas_metal,mincnt = 1,cmap='rainbow',reduce_C_function=np.median,vmin = 0.1, vmax = 0.5,gridsize = 30,bins = 'log')
        ax[g,h].plot(qb_lst,med_qb_lst,label = "Median Line", color = 'black')
        ax[g,h].plot(qb_lst,high_qb_lst, color = 'black',label = "_nolegend_",linestyle = 'dashed')
        ax[g,h].plot(qb_lst,low_qb_lst, color = 'black',label = "_nolegend_",linestyle = 'dashed')
        ax[g,h].set_title(tit)
     #   plt.xlabel("Log Stellar mass - Solar masses")
      #  plt.ylabel("qir")
        ax[g,h].set_xlim(8,12)
        ax[g,h].set_ylim(2,3.5)
        q += 1
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Gas Metallicity')
    fig.text(0.5, 0.04, "Log Stellar Mass ($M_\odot$)", ha='center')
    fig.text(0.04, 0.5, "qir", va='center', rotation='vertical')
    fig.suptitle('qir vs. Stellar Mass for Star Forming Galaxies - Bressan')
    plt.show()



def hist(df):
    q = 0
    x = int(len(zlist)/2)
    
    fig, ax = plt.subplots(2,x)
    for z in zlist:
        g = q//x
        h = q%x
    
        qir_dale_m = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')),'qir_dale']

        strz = str(z)
        tit = 'z = ' + strz[:4]
        
        ax[g,h].hist(qir_dale_m,bins = 100)
        ax[g,h].set_title(tit)
        ax[g,h].set_ylim(0,20000)
       # ax[g,h].set_yscale['log']
        ax[g,h].set_xlim(2.5,2.8)
     #   plt.xlabel("Log Stellar mass - Solar masses")
      #  plt.ylabel("qir")

        q+=1

    fig.text(0.5, 0.04, "Frequency", ha='center')
    fig.text(0.04, 0.5, "qir", va='center', rotation='vertical')
    fig.suptitle('qir for star forming galaxies histogram - Dale')
    plt.show()
    
    q = 0
    x = int(len(zlist)/2)
    
    fig, ax = plt.subplots(2,x)
    
    for z in zlist:
        g = q//x
        h = q%x
    
        qir_bress_m = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')),'qir_bress']

        strz = str(z)
        tit = 'z = ' + strz[:4]
        
        ax[g,h].hist(qir_bress_m,bins = 100)
        ax[g,h].set_title(tit)
        ax[g,h].set_ylim(0,80000)
        ax[g,h].set_xlim(2.0,3.5)

     #   plt.xlabel("Log Stellar mass - Solar masses")
      #  plt.ylabel("qir")

        q+=1

    fig.text(0.5, 0.04, "Frequency", ha='center')
    fig.text(0.04, 0.5, "qir", va='center', rotation='vertical')
    fig.suptitle('qir for star forming galaxies histogram - Bressan')
    plt.show()


def fir_lir_mass(df):
    q = 0
    x = int(len(zlist)/2)
    fig, ax = plt.subplots(2,x)
    for z in zlist:
        g = q//x
        h = q%x
        dale_rad_lum = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')&(df['mstars_tot'] < 10**12) & (df['mstars_tot'] > 10**8)),'dale_rad_lum']
        fir_flux = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')&(df['mstars_tot'] < 10**12) & (df['mstars_tot'] > 10**8)),'fir_flux']
        mstar_tot = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')&(df['mstars_tot'] < 10**12) & (df['mstars_tot'] > 10**8)),'mstars_tot']
        
        dale_rad_lum = np.log10(dale_rad_lum)
        fir_flux = np.log10(fir_flux)
        mstar_tot = np.log10(mstar_tot)
       
        if z == 0.194738848008908:
            strz = "0.20"
        else:
            strz = round(z,2)
            strz = str(strz) 
        


        x1 = np.linspace(13,25,3)
        y = 2.34 + np.log10(3.75e12) + x1
     
        tit = "z = " + strz + "0"
        print("THis is g,h")
        print(g,h)        
        c=ax[g,h].hexbin(dale_rad_lum,fir_flux,C=mstar_tot,mincnt = 1,reduce_C_function=np.median,cmap='rainbow', gridsize = 30,vmin = 8, vmax = 12)
        ax[g,h].plot(x1,y,label = "qir = 2.34",color = 'black')
        ax[g,h].set_title(tit)
        ax[g,h].set_ylim(30,40)
        ax[g,h].set_xlim(13,25)
        q += 1
        print("This is z and q")
        print(z,q)
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Stellar Mass')
    fig.text(0.5, 0.04, "Radio Luminosity (W) at 1.4 GHz", ha='center')
    fig.text(0.04, 0.5, "Far-Infrared Luminosity (W)", va='center', rotation='vertical')
    fig.suptitle('FIR vs. Radio Luminosity coloured by stellar mass - Dale')
    plt.show()
    
    q = 0
    x = int(len(zlist)/2)
    
    fig, ax = plt.subplots(2,x)
    for z in zlist:

        g = q//x
        h = q%x
        bress_rad_lum = df.loc[((df['mstars_tot'] < 10**12) & (df['mstars_tot'] > 10**8) & (df['z'] == z) & (df['sfr/q'] == 'sf')),'bress_rad_lum']
        fir_flux = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')&(df['mstars_tot'] < 10**12) & (df['mstars_tot'] > 10**8)),'fir_flux']
        mstar_tot = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')&(df['mstars_tot'] < 10**12) & (df['mstars_tot'] > 10**8)),'mstars_tot']
        
        bress_rad_lum = np.log10(bress_rad_lum)
        fir_flux = np.log10(fir_flux)
        mstar_tot = np.log10(mstar_tot)
        
        if z == 0.194738848008908:
            strz = "0.20"
        else:
            strz = round(z,2)
            strz = str(strz)
        
        tit = "z = " + strz + "0"
        
        x1 = np.linspace(13,25,3)
        y = 2.34 + np.log10(3.75e12) + x1


        ax[g,h].plot(x1,y,label = "qir = 2.34",color = 'black')
        ax[g,h].set_ylim(30,40)
        ax[g,h].set_xlim(13,25)


        c=ax[g,h].hexbin(bress_rad_lum,fir_flux,C=mstar_tot,mincnt = 1,reduce_C_function=np.median,cmap='rainbow', gridsize = 30,vmin = 8, vmax = 12)

        ax[g,h].set_title(tit)

        q += 1
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Stellar Mass')
    fig.text(0.5, 0.04, "Radio Luminosity (W) at 1.4 GHz", ha='center')
    fig.text(0.04, 0.5, "Far-Infrared Luminosity (W)", va='center', rotation='vertical')
    fig.suptitle('FIR vs. Radio Luminosity coloured by stellar mass - Bressan')
    plt.show()

def lum_m(df,ab_lst):
    
    q = 0
    x = int(len(zlist)/2)
    fig, ax = plt.subplots(2,x)
    for z in zlist:
    
        g = q//x
        h = q%x


        bress_rad_lum = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)),'bress_rad_lum']
       # fir_flux = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')),'fir_flux']
        mstar_tot = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)),'mstars_tot']
        
        bress_rad_lum = np.log10(bress_rad_lum)
      #  fir_flux = np.log10(fir_flux)
        mstar_tot = np.log10(mstar_tot)
        ab_tup = ab_lst[q]
        a = ab_tup[0]
        b = ab_tup[1]
        
        a = ab_tup[0]
        b = ab_tup[1]
        print("This is ab_tup")
        print(ab_tup,a,b)
        x1 = np.linspace(8,12,2)

        y = a * x1 + b + 21

        z = round(z,1)
        strz = str(z)
        
        tit = "z = " + strz + "0"
        
        c=ax[g,h].hexbin(mstar_tot,bress_rad_lum,mincnt = 1,reduce_C_function=np.median,cmap='rainbow',gridsize = 30,vmin = 0, vmax = 5000)
        ax[g,h].plot(x1,y,color = 'black',label = 'Star forming fit + 22')
        ax[g,h].set_title(tit)
        
        ax[g,h].set_ylim(15,25)

        q += 1
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Frequency')
    fig.text(0.5, 0.04, "Log Stellar Mass", ha='center')
    fig.text(0.04, 0.5, "Log Radio Luminosity (W)", va='center', rotation='vertical')
    fig.suptitle('Radio Luminosity vs. Stellar mass - Bressan, star forming galaxies')
 #   plt.legend()
    plt.show()


    q = 0
    x = int(len(zlist)/2)
    fig, ax = plt.subplots(2,x)
    for z in zlist:

        g = q//x
        h = q%x


        bress_rad_lum = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)),'dale_rad_lum']
       # fir_flux = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')),'fir_flux']
        mstar_tot = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)),'mstars_tot']

        bress_rad_lum = np.log10(bress_rad_lum)
      #  fir_flux = np.log10(fir_flux)
        mstar_tot = np.log10(mstar_tot)
        ab_tup = ab_lst[q]
        a = ab_tup[0]
        b = ab_tup[1]

        a = ab_tup[0]
        b = ab_tup[1]
        print("This is ab_tup")
        print(ab_tup,a,b)
        x1 = np.linspace(8,12,2)

        y = a * x1 + b + 20


        strz = str(z)

        tit = "z = " + strz

        c=ax[g,h].hexbin(mstar_tot,bress_rad_lum,mincnt = 1,reduce_C_function=np.median,cmap='rainbow', gridsize = 30,vmin = 0, vmax = 5000)
        ax[g,h].plot(x1,y,color = 'black',label = 'Star forming fit + 20')
        ax[g,h].set_title(tit)

        ax[g,h].set_ylim(15,25)

        q += 1
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Frequency')
    fig.text(0.5, 0.04, "Log Stellar Mass", ha='center')
    fig.text(0.04, 0.5, "Log Radio Luminosity (W)", va='center', rotation='vertical')
    fig.suptitle('Radio Luminosity vs. Stellar mass - Dale, star forming galaxies')
    plt.show()





    q = 0
    x = int(len(zlist)/2)
    fig, ax = plt.subplots(2,x)


    for z in zlist:

        g = q//x
        h = q%x

        bress_rad_lum = df.loc[((df['z'] == z) & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)),'bress_rad_lum']
       # fir_flux = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')),'fir_flux']
        mstar_tot = df.loc[((df['z'] == z) & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)),'mstars_tot']

        bress_rad_lum = np.log10(bress_rad_lum)
      #  fir_flux = np.log10(fir_flux)
        mstar_tot = np.log10(mstar_tot)

        strz = str(z)

        tit = "z = " + strz

        c=ax[g,h].hexbin(mstar_tot,bress_rad_lum,mincnt = 1,reduce_C_function=np.median,cmap='rainbow', gridsize = 30,vmin = 0, vmax = 5000)

        ax[g,h].set_title(tit)

        ax[g,h].set_ylim(15,25)

        q += 1
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Frequency')
    fig.text(0.5, 0.04, "Log Stellar Mass", ha='center')
    fig.text(0.04, 0.5, "Log Radio Luminosity (W)", va='center', rotation='vertical')
    fig.suptitle('Radio Luminosity vs. Stellar mass - Bressan, All Galaxies')
    plt.show()
    
    
    q = 0
    x = int(len(zlist)/2)
    
    fig, ax = plt.subplots(2,x)
    
    for z in zlist:
        #bress_rad_lum = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)),'bress_rad_lum']
        fir_flux = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf')& (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)),'fir_flux']
        mstar_tot = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)),'mstars_tot']
        g = q//x
        h = q%x
      #  bress_rad_lum = np.log10(bress_rad_lum)
        fir_flux = np.log10(fir_flux)
        mstar_tot = np.log10(mstar_tot)
        
        strz = str(z)
        
        tit = "z = " + strz
        
        c=ax[g,h].hexbin(mstar_tot,fir_flux,mincnt = 1,reduce_C_function=np.median,cmap='rainbow', gridsize = 30,vmin = 0, vmax = 5000)

        ax[g,h].set_title(tit)
        
        ax[g,h].set_ylim(30,40)

        q += 1
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Frequency')
    fig.text(0.5, 0.04, "Log Stellar Mass", ha='center')
    fig.text(0.04, 0.5, "Log FIR Luminosity(W)", va='center', rotation='vertical')
    fig.suptitle('FIR Luminosity vs. Stellar mass - Both Models, star forming galaxies')
    plt.show()



def percentiles_qir_metals_bressan(df,z,m1,m2):
    metals_lst = np.logspace(-2, 0.5, 30)
    print("This is metals_lst")
    print(metals_lst)
    qb_lst = []
    med_qb_lst = []
    low_qb_lst= []
    high_qb_lst = []
    
    for i in range(len(metals_lst)-1):
        m = metals_lst[i]
        n = metals_lst[i+1]
        qb = df.loc[((df['z'] == z)&(df['gas_metal'] > m)&(df['gas_metal'] < n)&(df['sfr/q'] == 'sf')&(df['mstars_tot'] >10**m1)&(df['mstars_tot'] < 10**m2)),'qir_bress']
     #   print("This is m and n")
     #   print(m,n)
     #   print("This is qb")
     #   print(list(qb))
        med_qb = np.median(qb)
        print("This is med_qb")
        print(med_qb)
        med_qb_lst.append(med_qb)
        mid_qb = (n/2+m/2)
        qb_lst.append(np.log10(mid_qb))
        try:
            low_qb_lst.append(np.percentile(qb,16))
            high_qb_lst.append(np.percentile(qb,84))
        except:
            low_qb_lst.append(np.nan)
            high_qb_lst.append(np.nan)
            continue
        
    print("This is med_qb_lst and qb_lst")
    print(med_qb_lst,qb_lst)
        
        
    return med_qb_lst, high_qb_lst, low_qb_lst, qb_lst
    
    

def metal_hist(df):
    
    
    gas_metal = df.loc[(df['sfr/q'] == 'sf'),'gas_metal']
    
    plt.hist(gas_metal)
    plt.xscale('log')
    plt.title("Histogram of gas metals for star forming galaxies")
    
    plt.show()
    
def qir_metals_mass_bins(df):
    zlist1 = [0, 0.909822023685613, 2.00391410007239, 3.0191633709527, 3.95972701662501, 5.02220991014863]
    q = 0
    g = 0
    mass_bins = np.arange(8,13,1)
    fig, ax = plt.subplots(6,4)
    for z in zlist1:
        
        n = 0
        
        h = 0
        

        
        for i in range(len(mass_bins) - 1):
            m1 = mass_bins[i]
            m2 = mass_bins[i+1]
            
            
            strm = '$10^{' + str(m1) + '}$<  M/$M_\odot$ < $10^{' + str(m2) + '}$'
            strm = str(strm)
            
            gas_metal = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**m1) & (df['mstars_tot'] < 10**m2)),'gas_metal']
            qir_bress = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**m1) & (df['mstars_tot'] < 10**m2)),'qir_bress']
            gas_metal = np.log10(gas_metal)
            teff = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**m1) & (df['mstars_tot'] < 10**m2)),'Teff']
            sb_frac = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**m1) & (df['mstars_tot'] < 10**m2)),'sb_frac']

            med_qb_lst, high_qb_lst, low_qb_lst, qb_lst = percentiles_qir_metals_bressan(df,z,m1,m2)
            
            c = ax[g,h].hexbin(gas_metal,qir_bress,C=sb_frac, cmap = 'rainbow', gridsize = 30,vmin = 0, vmax = 1)
            ax[g,h].plot(qb_lst,med_qb_lst,color = 'black', label = 'Median value')
            ax[g,h].plot(qb_lst,low_qb_lst,color = 'black',linestyle='dashed')
            ax[g,h].plot(qb_lst,high_qb_lst,color = 'black',linestyle='dashed')
            
            
            ax[g,h].set_xlim(-2.2,0.7)
            ax[g,h].set_ylim(-3.2,3.2)
            ax[g,h].get_xaxis().set_visible(False)
            ax[g,h].get_yaxis().set_visible(False)
            if g == 0:
                ax[g,h].set_title(strm,fontsize = 8)
            if h == 0:
                ax[g,h].get_yaxis().set_visible(True)
                z1 = round(z,1)
                strz = 'z = ' + str(z1)
                ax[g,h].set_ylabel(strz,fontsize = 8)
            if g == 5:
                ax[g,h].get_xaxis().set_visible(True)
            h += 1
        g +=1
    fig.subplots_adjust(wspace=0, hspace=0)
    
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Star Burst Fraction')
    fig.text(0.5, 0.04, "Log Gas Metallicity", ha='center',fontsize = 12)
    fig.text(0.04, 0.5, "qir", va='center', rotation='vertical',fontsize = 12)
    fig.suptitle('qir vs. gas metallicity - Bressan',fontsize = 12)
    plt.show()






def gas_mass_median_line(df,z):
    
    gas_mass_lst = np.logspace(7, 10.5, 30)
    qb_lst = []
    med_qb_lst = []
    low_qb_lst= []
    high_qb_lst = []
    
    for i in range(len(gas_mass_lst)-1):
        
        m = gas_mass_lst[i]
        n = gas_mass_lst[i+1]
        
        
        qb = df.loc[((df['z'] == z)&(df['gas_mass'] > m)&(df['gas_mass'] < n)&(df['sfr/q'] == 'sf')&(df['mstars_tot'] >10**8)&(df['mstars_tot'] < 10**9)&(df['type'] == 0)),'qir_bress']

        med_qb = np.median(qb)
        print("This is med_qb")
        print(med_qb)
        med_qb_lst.append(med_qb)
        mid_qb = (n/2+m/2)
        qb_lst.append(np.log10(mid_qb))
        try:
            low_qb_lst.append(np.percentile(qb,16))
            high_qb_lst.append(np.percentile(qb,84))
        except:
            low_qb_lst.append(np.nan)
            high_qb_lst.append(np.nan)
            continue
        
    print("This is med_qb_lst and qb_lst")
    print(med_qb_lst,qb_lst)
        
        
    return med_qb_lst, high_qb_lst, low_qb_lst, qb_lst

def freefree_median_line(df,z):
    
    freefree_lst = np.logspace(24, 28, 30)
    qb_lst = []
    med_qb_lst = []
    low_qb_lst= []
    high_qb_lst = []
    
    for i in range(len(freefree_lst)-1):
        
        m = freefree_lst[i]
        n = freefree_lst[i+1]
        
        qb = df.loc[((df['z'] == z)&(df['freefree'] > m)&(df['freefree'] < n)&(df['sfr/q'] == 'sf')&(df['mstars_tot'] >10**8)&(df['mstars_tot'] < 10**9)&(df['type'] == 0)),'qir_bress']

        med_qb = np.median(qb)
        print("This is med_qb")
        print(med_qb)
        med_qb_lst.append(med_qb)
        mid_qb = (n/2+m/2)
        qb_lst.append(np.log10(mid_qb))
        try:
            low_qb_lst.append(np.percentile(qb,16))
            high_qb_lst.append(np.percentile(qb,84))
        except:
            low_qb_lst.append(np.nan)
            high_qb_lst.append(np.nan)
            continue
        
    print("This is med_qb_lst and qb_lst")
    print(med_qb_lst,qb_lst)
        
        
    return med_qb_lst, high_qb_lst, low_qb_lst, qb_lst

def sync_median_line(df,z):
    
    sync_lst = np.logspace(26.5, 30, 30)
    qb_lst = []
    med_qb_lst = []
    low_qb_lst= []
    high_qb_lst = []
    
    for i in range(len(sync_lst)-1):
        
        m = sync_lst[i]
        n = sync_lst[i+1]
        
        
        qb = df.loc[((df['z'] == z)&(df['sync'] > m)&(df['sync'] < n)&(df['sfr/q'] == 'sf')&(df['mstars_tot'] >10**8)&(df['mstars_tot'] < 10**9)&(df['type'] == 0)),'qir_bress']

        med_qb = np.median(qb)
        print("This is med_qb")
        print(med_qb)
        med_qb_lst.append(med_qb)
        mid_qb = (n/2+m/2)
        qb_lst.append(np.log10(mid_qb))
        try:
            low_qb_lst.append(np.percentile(qb,16))
            high_qb_lst.append(np.percentile(qb,84))
        except:
            low_qb_lst.append(np.nan)
            high_qb_lst.append(np.nan)
            continue
        
    print("This is med_qb_lst and qb_lst")
    print(med_qb_lst,qb_lst)
        
        
    return med_qb_lst, high_qb_lst, low_qb_lst, qb_lst
    
def qir_v_dust_ff_sync(df):




    zlist1 = [0, 0.909822023685613, 2.00391410007239, 3.0191633709527, 3.95972701662501, 5.02220991014863]
   
    q = 0
    g = 0
    mass_bins = np.arange(8,13,1)
    fig, ax = plt.subplots(6,3)
    for z in zlist1:
        n = 0
        
        h = 0
        
        gas_mass = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**9)&(df['gas_mass'] > 10**7) & (df['gas_mass'] < 10**10.5)&(df['type'] == 0)),'gas_mass']
        qir_bress_g = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**9)&(df['gas_mass'] > 10**7) & (df['gas_mass'] < 10**10.5)&(df['type'] == 0)),'qir_bress']
        freefree = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**9) & (df['freefree'] > 10**24) & (df['freefree'] < 10**28)&(df['type'] == 0)),'freefree']
        qir_bress_f = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**9) & (df['freefree'] > 10**24) & (df['freefree'] < 10**28)&(df['type'] == 0)),'qir_bress']
        sync = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**9)& (df['sync'] > 10**26.5) & (df['sync'] < 10**30)&(df['type'] == 0)),'sync']
        qir_bress_s = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**9)&(df['sync'] > 10**26.5) & (df['sync'] < 10**30)&(df['type'] == 0)),'qir_bress']
        teff_s = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**9)&(df['sync'] > 10**26.5) & (df['sync'] < 10**30)&(df['type'] == 0)),'Teff']
        teff_g = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**9)&(df['gas_mass'] > 10**7) & (df['gas_mass'] < 10**10.5)&(df['type'] == 0)),'Teff']
        teff_f = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**9) & (df['freefree'] > 10**24) & (df['freefree'] < 10**28)&(df['type'] == 0)),'Teff']
        
        sb_frac_s = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**9)&(df['sync'] > 10**26.5) & (df['sync'] < 10**30)&(df['type'] == 0)),'sb_frac']
        sb_frac_g = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**9)&(df['gas_mass'] > 10**7) & (df['gas_mass'] < 10**10.5)&(df['type'] == 0)),'sb_frac']
        sb_frac_f = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**9) & (df['freefree'] > 10**24) & (df['freefree'] < 10**28)&(df['type'] == 0)),'sb_frac']


        gas_mass = np.log10(gas_mass)
        freefree = np.log10(freefree)
        sync = np.log10(sync)
        
        
        mgas_med_qb_lst, mgas_high_qb_lst, mgas_low_qb_lst, mgas_qb_lst = gas_mass_median_line(df,z)
       
        c = ax[g,0].hexbin(gas_mass,qir_bress_g,C=sb_frac_g, cmap = 'rainbow', gridsize = 30,vmin = 0,vmax = 1)
        ax[g,0].plot(mgas_qb_lst,mgas_med_qb_lst,color = 'black', label = 'Median value')
        ax[g,0].plot(mgas_qb_lst,mgas_low_qb_lst,color = 'black',linestyle='dashed')
        ax[g,0].plot(mgas_qb_lst,mgas_high_qb_lst,color = 'black',linestyle='dashed')
        ax[g,0].get_xaxis().set_visible(False)
        ax[g,0].get_yaxis().set_visible(True)
        
        z1 = round(z,1)
        strz = 'z = ' + str(z1)
        ax[g,0].set_ylabel(strz,fontsize = 8)
        
        
        ax[g,0].set_ylim(-0.2,3.2)
        ax[g,0].set_xlim(6.8,10.7)

        ax[0,0].set_title('Gas Mass',fontsize = 8)
        
        
        ff_med_qb_lst, ff_high_qb_lst, ff_low_qb_lst, ff_qb_lst = freefree_median_line(df,z)
        
        ax[g,1].hexbin(freefree,qir_bress_f, C = sb_frac_f,cmap = 'rainbow', gridsize = 30,vmin = 0, vmax = 1)
        ax[g,1].plot(ff_qb_lst,ff_med_qb_lst,color = 'black', label = 'Median value')
        ax[g,1].plot(ff_qb_lst,ff_low_qb_lst,color = 'black',linestyle='dashed')
        ax[g,1].plot(ff_qb_lst,ff_high_qb_lst,color = 'black',linestyle='dashed')
        
        ax[g,1].get_xaxis().set_visible(False)
        ax[g,1].get_yaxis().set_visible(False)
        
        ax[g,1].set_ylim(-0.2,3.2)
        ax[g,1].set_xlim(23.8,28.2)
        
        ax[0,1].set_title('Free Free Emission',fontsize = 8)
        
        sc_med_qb_lst, sc_high_qb_lst, sc_low_qb_lst, sc_qb_lst = sync_median_line(df,z)
        
        
        c = ax[g,2].hexbin(sync,qir_bress_s,C=sb_frac_s, cmap = 'rainbow', gridsize = 30,vmin = 0, vmax = 1)
        ax[g,2].plot(sc_qb_lst,sc_med_qb_lst,color = 'black', label = 'Median value')
        ax[g,2].plot(sc_qb_lst,sc_low_qb_lst,color = 'black',linestyle='dashed')
        ax[g,2].plot(sc_qb_lst,sc_high_qb_lst,color = 'black',linestyle='dashed')
        
        ax[g,2].set_ylim(-0.2,3.2)
        ax[g,2].set_xlim(26.2,30.2)
        
        ax[g,2].get_xaxis().set_visible(False)
        ax[g,2].get_yaxis().set_visible(False)
        
        ax[0,2].set_title('Synchrotron Emission',fontsize = 8)


        ax[5,0].get_xaxis().set_visible(True)
        ax[5,0].set_xlabel('Log Total Gas Mass',fontsize = 8)
        
        ax[5,1].get_xaxis().set_visible(True)
        ax[5,1].set_xlabel('Log Free Free Luminosity (W)',fontsize = 8)
        
        ax[5,2].get_xaxis().set_visible(True)
        ax[5,2].set_xlabel('Log Synchrotron Luminosity (W)',fontsize = 8)
            
        
        
        
        g +=1
    fig.subplots_adjust(wspace=0, hspace=0)
    
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Star-Burst Fraction')

    fig.text(0.04, 0.5, "qir", va='center', rotation='vertical',fontsize = 12)
    fig.suptitle('qir vs. gas mass, free free and synchrotron emission - Bressan Model, Star Forming Galaxies in stellar mass range $10^{8}$ - $10^{9}$ $M_\odot$. Coloured by Star-Burst Fraction',fontsize = 12)
    plt.show()


def gas_metal_mass_median_line(df,z):
    
    mass_lst = np.logspace(8, 12, 30)
    qb_lst = []
    med_qb_lst = []
    low_qb_lst= []
    high_qb_lst = []
    
    for i in range(len(mass_lst)-1):
        
        m = mass_lst[i]
        n = mass_lst[i+1]
        print("This is m and n")
        print(m,n)
        gas_metal = df.loc[((df['z'] == z)&(df['mstars_tot'] > m)&(df['mstars_tot'] < n)&(df['sfr/q'] == 'sf')&(df['type'] == 0)),'gas_metal']
        print("This is gas metal")
        print(gas_metal)
        gas_metal = np.log10(gas_metal)
        med_gm = np.median(gas_metal)
        print("This is med_gm")
        print(med_gm)
        med_qb_lst.append(med_gm)
        mid_qb = (n/2+m/2)
        qb_lst.append(np.log10(mid_qb))
        try:
            low_qb_lst.append(np.percentile(qb,16))
            high_qb_lst.append(np.percentile(qb,84))
        except:
            low_qb_lst.append(np.nan)
            high_qb_lst.append(np.nan)
            continue
        
    print("This is med_qb_lst and qb_lst")
    print(med_qb_lst,qb_lst)
        
        
    return med_qb_lst, high_qb_lst, low_qb_lst, qb_lst    

def gas_metal_vs_stellar_mass(df):
    q = 0
    

    x = int(len(zlist)/2)
    fig, ax = plt.subplots(2,x)
    for z in zlist:
        g = q//x
        h = q%x    




        med_qb_lst, high_qb_lst, low_qb_lst, qb_lst  = gas_metal_mass_median_line(df,z)
    
        gas_metal = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)&(df['type'] == 0)),'gas_metal']
        mstars_tot = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)&(df['type'] == 0)),'mstars_tot']
        typ = df.loc[((df['z'] == z) & (df['sfr/q'] == 'sf') & (df['mstars_tot'] > 10**8) & (df['mstars_tot'] < 10**12)&(df['type'] == 0)),'qir_bress'] 
        gas_metal = np.log10(gas_metal)
        mstars_tot = np.log10(mstars_tot)
        
        
      #  mstars_tot = np.log10(mstars_tot)
        strz = round(z)
        strz = str(strz)
        tit = 'z = ' + strz[:4]
        c = ax[g,h].hexbin(mstars_tot,gas_metal,C=typ,cmap = 'rainbow', gridsize = 30,vmin = -2 , vmax = 2)

        ax[g,h].plot(qb_lst,med_qb_lst,label = "Median Line", color = 'red')
        ax[g,h].plot(qb_lst,high_qb_lst, color = 'red',label = "_nolegend_",linestyle = 'dashed')
        ax[g,h].plot(qb_lst,low_qb_lst, color = 'red',label = "_nolegend_",linestyle = 'dashed')
        ax[g,h].set_title(tit)
     #   plt.xlabel("Log Stellar mass - Solar masses")
      #  plt.ylabel("qir")
        ax[g,h].set_xlim(8,12)
        ax[g,h].set_ylim(-2.5,1)
        q+=1
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('qir bressan')
    fig.text(0.5, 0.04, "Log Stellar Mass", ha='center')
    fig.text(0.04, 0.5, "Log Gas Metallicity", va='center', rotation='vertical')
    fig.suptitle('Gas Metallicity vs. Stellar Mass')
    plt.show()

def median_line(xlist,ylist,xlog,ylog):
    
    
    
    med_y_lst = []
    low_y_lst = []
    high_y_lst = []
    mid_x_lst = []
    
    
    
    if xlog == True:
        xlist = np.log10(np.array(xlist))
    else:
        xlist = np.array(xlist)
        
    if ylog == True:
        ylist = np.log10(np.array(ylist))
    else:
        ylist = np.array(ylist)
        
    xbin_lst = list(np.arange(min(xlist),max(xlist),(max(xlist)-min(xlist))/30))


    for i in range(len(xbin_lst)-1):
        
        m = xbin_lst[i]
        n = xbin_lst[i+1]

        x_bins = np.where(((xlist > m)&(xlist < n)))


        y_bins = ylist[x_bins]
        
        
        med_y = np.median(y_bins)
        med_y_lst.append(med_y)
        mid_x = (n/2+m/2)
        mid_x_lst.append(mid_x)
        try:
            low_y_lst.append(np.percentile(y_bins,16))
            high_y_lst.append(np.percentile(y_bins,84))
        except:
            low_y_lst.append(np.nan)
            high_y_lst.append(np.nan)
            continue
    if ylog == True:
        med_y_lst = 10**np.array(med_y_lst)
        low_y_lst = 10**np.array(low_y_lst)
        high_y_lst = 10**np.array(high_y_lst)
    if xlog == True:
        mid_x_lst = 10**np.array(mid_x_lst)
    
    return mid_x_lst,med_y_lst,low_y_lst,high_y_lst



def lo_faro_plots(df):
    

    Lsun = 3.828 * 10**26 #W
    df_lf = pd.read_csv('french_paper_2_pg16.csv')
    df_lf['LIR/W'] = df_lf['LIR'] * Lsun

    df_lf['qir'] = np.log10((df_lf['LIR/W']/(3.75*10**12))/df_lf['L1.4'])
    
    q = 0
    
    df_lf['L1.4_err'] = df_lf['L1.4']*(0.1 *np.log(10) - 10**(0.13)/df_lf['LIR/W'])
    
    sfr10_1 = df_lf.loc[(df_lf['z'] < 1.5),'SFR10']
    sfr10_2 = df_lf.loc[(df_lf['z'] > 1.5),'SFR10']
    m_1 = df_lf.loc[(df_lf['z'] < 1.5),'M_star']
    m_2 = df_lf.loc[(df_lf['z'] > 1.5),'M_star']

    LIR1 = df_lf.loc[(df_lf['z'] < 1.5),'LIR/W']
    LIR2 = df_lf.loc[(df_lf['z'] > 1.5),'LIR/W']

    Lrad1 = df_lf.loc[(df_lf['z'] < 1.5),'L1.4']
    Lrad2 = df_lf.loc[(df_lf['z'] > 1.5),'L1.4']

    qir1 = df_lf.loc[(df_lf['z'] < 1.5),'qir']
    qir2 = df_lf.loc[(df_lf['z'] > 1.5),'qir']


    LIR1 = np.log10(LIR1)
    LIR2 = np.log10(LIR2)

    Lrad1 = np.log10(Lrad1)
    Lrad2 = np.log10(Lrad2)

    sfr10_1 = np.log10(sfr10_1)
    sfr10_2 = np.log10(sfr10_2) 
    
    m_2 = np.log10(m_2) 
    m_1 = np.log10(m_1) 
    
    plt.hist(np.log10(df.loc[((df['z'] < 1.05)&(df['z']>0.8)),'fir_flux']))
    plt.show()
    
    rad_bress_1 = df.loc[((df['z'] < 1.05)&(df['z']>0.8)&(df['fir_flux']>7.88568e+37)),'bress_rad_lum']
    fir_bress_1 = df.loc[((df['z'] < 1.05)&(df['z']>0.8)&(df['fir_flux']>7.88568e+37)),'fir_flux']
    
    rad_bress_2 = df.loc[((df['z'] < 2.2)&(df['z']>1.5)&(df['fir_flux']>7.88568e+37)),'bress_rad_lum']
    fir_bress_2 = df.loc[((df['z'] < 2.2)&(df['z']>1.5)&(df['fir_flux']>7.88568e+37)),'fir_flux']
    
    rad_bress_1 = np.log10(rad_bress_1)
    
    rad_bress_2 = np.log10(rad_bress_2)
    fir_bress_2 = np.log10(fir_bress_2)
    fir_bress_1 = np.log10(fir_bress_1)
    
    


    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(rad_bress_1,fir_bress_1,False,False)



    fig, ax = plt.subplots(1,2)
    
    c = ax[0].hexbin(rad_bress_1,fir_bress_1,mincnt = 1, cmap='rainbow', gridsize = 30,vmin = 0, vmax = 50)
    ax[0].scatter(Lrad1,LIR1,marker="o",color='white', edgecolors='black')
    ax[0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')
    ax[0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0].set_xlim(20,26)
    ax[0].set_ylim(36,40)
    ax[0].set_title("z ~ 1")
    

    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(rad_bress_2,fir_bress_2,False,False)

    
    ax[1].hexbin(rad_bress_2,fir_bress_2,mincnt = 1, cmap='rainbow', gridsize = 30,vmin = 0, vmax = 50)
    ax[1].scatter(Lrad2,LIR2,marker="o",color='white', edgecolors='black')
    ax[1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')
    ax[1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1].set_xlim(20,26)
    ax[1].set_ylim(36,40)
    ax[1].set_title("z ~ 2")
    
    
    
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Frequency')
    fig.text(0.5, 0.04, "Log10(L 1.4 GHz) W/Hz", ha='center')
    fig.text(0.04, 0.5, "Log10(LFIR) W/Hz", va='center', rotation='vertical')
    fig.suptitle('Infrared Luminosity vs. Radio Luminosity - Lo Faro et al.')
    
    
    plt.show()
    
    sfr_1 = df.loc[((df['z'] < 1.05)&(df['z']>0.8)&(df['fir_flux']>7.88568e+37)&(df['fir_flux']<1.74174e+39)),'sfr']
    
    m_2 = df.loc[((df['z'] < 2.2)&(df['z']>1.5)&(df['fir_flux']>7.88568e+37)&(df['fir_flux']<1.74174e+39)),'mstars_tot']
    sfr_2 = df.loc[((df['z'] < 2.2)&(df['z']>1.5)&(df['fir_flux']>7.88568e+37)&(df['fir_flux']<1.74174e+39)),'sfr']
    
    m_1 = np.log10(m_1)
    m_2 = np.log10(m_2)
    
    sfr_1 = np.log10(sfr_1)
    sfr_2 = np.log10(sfr_2)
    
    qir1 = df.loc[((df['z'] < 1.05)&(df['z']>0.8)&(df['fir_flux']>7.88568e+37)&(df['fir_flux']<1.74174e+39)),'qir_bress']
    qir2 = df.loc[((df['z'] < 2.2)&(df['z']>1.5)&(df['fir_flux']>7.88568e+37)&(df['fir_flux']<1.74174e+39)),'qir_bress']
    
    
    m_lf_1 = np.log10(df_lf.loc[(df_lf['z'] < 1.5),'M_star'])
    m_lf_2 = np.log10(df_lf.loc[(df_lf['z'] > 1.5),'M_star'])
    
    sfr_lf_1 = np.log10(df_lf.loc[(df_lf['z'] < 1.5),'SFR10'])
    sfr_lf_2 = np.log10(df_lf.loc[(df_lf['z'] > 1.5),'SFR10'])
    
    qir_lf_1 = df_lf.loc[(df_lf['z'] < 1.5),'qTIR']
    qir_lf_2 = df_lf.loc[(df_lf['z'] > 1.5),'qTIR']
    
    a,b = np.polyfit(m_lf_1,qir_lf_1,1)
    x = np.linspace(8,12,len(m_lf_1))
    y = a*x +b
    
    m1_lf_up = m_lf_1 + 0.2
    qir1_up = qir_lf_1 + 0.1
    m1_lf_down = m_lf_1 - 0.2
    qir1_down = qir_lf_1 - 0.1
    
    m2_lf_up = m_lf_2 + 0.2
    qir2_up = qir_lf_2 + 0.1
    m2_lf_down = m_lf_2 - 0.2
    qir2_down = qir_lf_2 - 0.1
    
    a1,b1 = np.polyfit(m1_lf_up,qir1_up,1)
    x1 = np.linspace(8,12,len(m_lf_1))
    y1 = a1*x1 +b1
    
    a2,b2 = np.polyfit(m1_lf_down,qir1_down,1)
    x2 = np.linspace(8,12,len(m_lf_1))
    y2 = a2*x2 +b2
    
    a3,b3 = np.polyfit(m_lf_2,qir_lf_2,1)
    x3 = np.linspace(8,12,len(m_lf_2))
    y3 = a3*x3 +b3
    
    
    
    a4,b4 = np.polyfit(m2_lf_up,qir2_up,1)
    x4 = np.linspace(8,12,len(m_lf_2))
    y4 = a4*x4 +b4
    
    a5,b5 = np.polyfit(m2_lf_down,qir2_down,1)
    x5 = np.linspace(8,12,len(m_lf_2))
    y5 = a5*x5 +b5
    
    fig, ax = plt.subplots(1,2)
    
    c = ax[0].hexbin(m_1,qir1,mincnt = 1, cmap='Greys', gridsize = 30,vmin = 0, vmax = 50)
    
    ax[0].scatter(m_lf_1,qir_lf_1,color = 'red')
    ax[0].plot(x,y,color = 'blue')
    ax[0].plot(x1,y1,color='blue',linestyle='dashed')
    ax[0].plot(x2,y2,color='blue',linestyle='dashed')
    ax[0].set_xlim(8,12)
    ax[0].set_ylim(0,3.5)
    ax[0].set_title("z ~ 1")
    
    ax[1].hexbin(m_2,qir2,mincnt = 1, cmap='Greys', gridsize = 30,vmin = 0, vmax = 50)
    ax[1].scatter(m_lf_2,qir_lf_2,color = 'red')
    ax[1].plot(x3,y3,color = 'blue')
    ax[1].plot(x4,y4,color='blue',linestyle='dashed')
    ax[1].plot(x5,y5,color='blue',linestyle='dashed')
    ax[1].set_xlim(8,12)
    ax[1].set_ylim(0,3.5)
    ax[1].set_title("z ~ 2")
    
    
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    ax[1].set_ylim(0,3.5)
    ax[1].set_title("z ~ 2")
    
    
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Number Count')
    fig.text(0.5, 0.04, "Log10(SFR/$M_\odot$/yr) ", ha='center')
    fig.text(0.04, 0.5, "qir", va='center', rotation='vertical')
    fig.suptitle('qir vs. SFR - Lo Faro et al.')
    plt.show()

    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(m_1,sfr_1,False,False)

    c = ax[0].hexbin(m_1,sfr_1,mincnt = 1, cmap='rainbow', gridsize = 30,vmin = 0, vmax = 50)
    ax[0].scatter(m_lf_1,sfr_lf_1,marker="o",color='white', edgecolors='black')
    ax[0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')
    ax[0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0].set_xlim(8,12)
    ax[0].set_ylim(0,3)
    ax[0].set_title("z ~ 1")


    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(m_2,sfr_2,False,False)


    c = ax[1].hexbin(m_2,sfr_2,mincnt = 1, cmap='rainbow', gridsize = 30,vmin = 0, vmax = 50)
    ax[1].scatter(m_lf_2,sfr_lf_2,marker="o",color='white', edgecolors='black')
    ax[1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')
    ax[1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1].set_xlim(8,12)
    ax[1].set_ylim(0,3.0)
    ax[1].set_title("z ~ 2")



    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Number Count')
    fig.text(0.5, 0.04, "Log10(Stellar Mass/$M_\odot$) ", ha='center')
    fig.text(0.04, 0.5, "Log10(SFR/$M_\odot$/yr) ", va='center', rotation='vertical')
    fig.suptitle('SFR vs. Stellar Mass - Lo Faro et al.')
    plt.show()









def GAMA_plots(df):

    gd = pd.read_csv('GAMA_data_1.csv') #GAMA Data
    

    gd['qir'] = (np.log10(gd['DustLum_50']*Lsun/3.75*10**12) - np.log10(gd['radioLum2']))*10**(-1)
    gd['qir_err'] = (1/np.log(10)) * ((gd['DustLum_84'] - gd['DustLum_16'])/gd['DustLum_50'] + gd['radioLum2_err']/gd['radioLum2']) * (gd['radioLum2']/((gd['DustLum_50']*Lsun)/3.75*10**12))
    qir = gd['qir']
    qir_err = gd['qir_err']
    m = gd['StellarMass_50']
    sfr = gd['SFR_50']
    radlum = gd['radioLum2']
    firlum = gd['DustLum_50']*Lsun
    radlum_err = gd['radioLum2_err']
    firlum_err = ((gd['DustLum_16']*Lsun),(gd['DustLum_84']*Lsun))
    
    
    
    m_err = (gd['StellarMass_50']-gd['StellarMass_16'],gd['StellarMass_84']-gd['StellarMass_50'])
    sfr_err = (gd['SFR_50']-gd['SFR_16'],gd['SFR_84']-gd['SFR_50'])
    
 #   qir_bress = df.loc[((df['z'] < 0.1)&(df['fir_flux']>498972038.2)&(df['qir_bress']>2)&(df['qir_bress']<3)&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)&(df['sfr']>0.01)),'qir_bress']
 #   rad_bress = df.loc[((df['z'] < 0.1)&(df['fir_flux']>498972038.2)&(df['qir_bress']>2)&(df['qir_bress']<3)&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)&(df['sfr']>0.01)),'bress_rad_lum']
 #   fir_bress = df.loc[((df['z'] < 0.1)&(df['fir_flux']>498972038.2)&(df['qir_bress']>2)&(df['qir_bress']<3)&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)&(df['sfr']>0.01)),'fir_flux']
#    m_shark = df.loc[((df['z'] < 0.1)&(df['fir_flux']>498972038.2)&(df['qir_bress']>2)&(df['qir_bress']<3)&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)&(df['sfr']>0.01)),'mstars_tot']
 #   sfr_shark = df.loc[((df['z'] < 0.1)&(df['fir_flux']>498972038.2)&(df['qir_bress']>2)&(df['qir_bress']<3)&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)&(df['sfr']>0.01)),'sfr']
    
  #  m_shark = df.loc[((df['z'] < 0.5)&(df['bress_rad_lum']>5.60906e+19)&(df['fir_flux']>498972038.2)),'mstars_tot']
  #  sfr_shark = df.loc[((df['z'] < 0.5)&(df['bress_rad_lum']>5.60906e+19)&(df['fir_flux']>498972038.2)),'sfr']


    qir_bress = df.loc[((df['z'] == 0)&(df['sfr/q']=='sf')&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)),'qir_bress']
    rad_bress = df.loc[((df['z'] == 0)&(df['sfr/q']=='sf')&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)),'bress_rad_lum']
    fir_bress = df.loc[((df['z'] == 0)&(df['sfr/q']=='sf')&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)),'fir_flux']
    m_shark = df.loc[((df['z'] == 0)&(df['sfr/q']=='sf')&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)),'mstars_tot']
    sfr_shark = df.loc[((df['z'] == 0)&(df['sfr/q']=='sf')&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)),'sfr']  

    qir_bress1 = df.loc[((df['z'] == 0)&(df['sfr/q']=='sf')&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)&(df['qir_bress'] > 2)),'qir_bress']
    rad_bress1 = df.loc[((df['z'] == 0)&(df['sfr/q']=='sf')&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)&(df['qir_bress'] > 2)),'bress_rad_lum']
    fir_bress1 = df.loc[((df['z'] == 0)&(df['sfr/q']=='sf')&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)&(df['qir_bress'] > 2)),'fir_flux']
    m_shark1 = df.loc[((df['z'] == 0)&(df['sfr/q']=='sf')&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)&(df['qir_bress'] > 2)),'mstars_tot']
    sfr_shark1 = df.loc[((df['z'] == 0)&(df['sfr/q']=='sf')&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)&(df['qir_bress'] > 2)),'sfr']



    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(rad_bress,fir_bress,True,True)
    
    c = plt.hexbin(rad_bress,fir_bress,mincnt = 1, xscale='log',yscale='log',cmap='rainbow', gridsize = 30)
    plt.plot(mid_rad_lst,med_fir_lst,'black')
    plt.plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    plt.plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    plt.errorbar(radlum,firlum, xerr = radlum_err,yerr = firlum_err,fmt="o",markerfacecolor='white', markeredgecolor='black',label = 'GAMA Data',alpha=1,ecolor='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("GAMA Data Comparison - LIR vs. Lrad")
    plt.xlabel("Log10(Lrad (1.4GHz)) W/Hz")
    plt.ylabel("Log10(LIR) W")
    cb = plt.colorbar(c)
    cb.set_label('Number count')
    plt.show()
    
    mid_m_lst,med_qir_lst,low_qir_lst,high_qir_lst = median_line(m_shark1,qir_bress1,True,False)
    

    
    c = plt.hexbin(m_shark1,qir_bress1,mincnt = 1, xscale='log',cmap='rainbow', gridsize = 30) 
    plt.plot(mid_m_lst,med_qir_lst,'black')
    plt.plot(mid_m_lst,low_qir_lst,'black',linestyle='dashed')   
    plt.plot(mid_m_lst,high_qir_lst,'black',linestyle='dashed')
    plt.errorbar(m,qir, xerr = m_err,yerr = qir_err,fmt="o",label = 'GAMA Data',markerfacecolor='white', markeredgecolor='black',ecolor = 'black',alpha=1)
    plt.xscale('log')
    plt.title("GAMA Data Comparison - qir vs. Stellar mass")
    plt.xlabel("Log10(Stellar Mass/$M_\odot$)")
    plt.ylabel("qir")
    cb = plt.colorbar(c)
    cb.set_label('Number count')
    plt.show()
    
    mid_sfr_lst,med_qir_lst,low_qir_lst,high_qir_lst = median_line(sfr_shark1,qir_bress1,True,False)
    
    c = plt.hexbin(sfr_shark1,qir_bress1,mincnt = 1, xscale='log',cmap='rainbow', gridsize = 30)
    
    plt.plot(mid_sfr_lst,med_qir_lst,'black')
    plt.plot(mid_sfr_lst,low_qir_lst,'black',linestyle='dashed')   
    plt.plot(mid_sfr_lst,high_qir_lst,'black',linestyle='dashed')
    
    
    plt.errorbar(sfr,qir, xerr =sfr_err,yerr = qir_err,fmt="o",label = 'GAMA Data',markerfacecolor='white', markeredgecolor='black',ecolor = 'black',alpha=1)
    plt.xscale('log')
    plt.title("GAMA Data Comparison - qir vs. SFR")
    plt.xlabel("Log10(SFR/$M_\odot$/yr)")
    plt.ylabel("qir")
    plt.xlim(0,1000) 
    cb = plt.colorbar(c)
    cb.set_label('Number count')

    plt.show()

    mid_m_lst,med_sfr_lst,low_sfr_lst,high_sfr_lst = median_line(m_shark,sfr_shark,True,True)
    
    c = plt.hexbin(m_shark,sfr_shark,mincnt = 1, xscale='log',yscale = 'log',cmap='rainbow', gridsize = 30)
    
    plt.plot(mid_m_lst,med_sfr_lst,'black')
    plt.plot(mid_m_lst,low_sfr_lst,'black',linestyle='dashed')   
    plt.plot(mid_m_lst,high_sfr_lst,'black',linestyle='dashed')
    

    plt.errorbar(m,sfr, xerr =m_err,yerr = sfr_err,fmt="o",label = 'GAMA Data',markerfacecolor='white', markeredgecolor='black',ecolor = 'black')
    plt.title("GAMA Data Comparison - SFR vs. Stellar Mass")
    plt.xlabel("Log10(Stellar Mass/$M_\odot$)")
    plt.ylabel("Log10(SFR/$M_\odot$/yr)")
    cb = plt.colorbar(c)
    cb.set_label('Number count')



    plt.show()

    



def bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,zlist):
    Lsunwatts = 3.846e26
    h = 6.6261e-27 #cm2 g s-1
    d10pc = 3.086e+19
    dfac = 4 * PI * d10pc**2
    qir_bress = []
    bress_rad_lum = []
    bress_lir_total = []
    qir_lst = []
    IncludeSelfAbsorption = False
    freefree_lst = []
    sync_lst = []



    for z in range(len(zlist)):
        sfr_disk = np.array(hdf5_lir_lst[z][4])
        sfr_disk_2 = []
        sfr_bulge = np.array(hdf5_lir_lst[z][5])
        sfr_bulge_2 = []
        sfr_tot = []
        sfr_tot = sfr_bulge + sfr_disk
        mdisk = np.array(hdf5_lir_lst[z][2])
        mbulge = np.array(hdf5_lir_lst[z][3])
        print("THis is mdisk")
        print(mdisk)

        ind0 = np.where(np.asarray(mdisk) + np.asarray(mbulge) > 0)
        ms = (mdisk[ind0] + mbulge[ind0])/h0 #in Msun
        sfr_tot = (sfr_tot[ind0])/h0/1e9
        #ionising mag
        lir_total = np.array(lir_total_W_lst[z], dtype=np.float64)
       # ind109 = np.where(lir_total > 10**7 * Lsun)
      #  print("This is the length of sfr before removing 10^9")
      #  print(len(sfr_tot))
      #  lir_total = lir_total[ind109]
      # ms = ms[ind109]
       # sfr_tot= sfr_tot[ind109]
        ind108 = np.where(ms > 0)
        print("This is length of lir_total (based on seds file)")
        print(len(lir_total))
        print("This is length of stellar mass (based on galaxies file)")
        print(len(ms))
        lir_total = lir_total[ind108]
        ms = ms[ind108]
        sfr_tot= sfr_tot[ind108]
        
        
        
        print("This is length of sfr")
        print(len(sfr_tot))

        n = 0
        

        def ionising_photons(m, wave):
            #m is a vector of AB absolute magnitudes in a band with central wavelength wave
            #wavelength input has to be in angstrom

            wave_m = wave * 1e-10 #wavelength in m
            wave_cm = wave_m  * 1e2 #wavelength in cm
            freq = c_speed / wave_m #Hz
            hc =  h * (c_speed * 1e2) #h*c in cgs

            lum = 10.0**((m + 48.6) / (-2.5)) * dfac * freq * wave_cm #we want to convert from Luminosity to int(lambda*Lum_lambda*dlambda)
            Q = lum / hc #rate of ionising photons in s^-1.

            return Q

        def freefree_lum(Q, nu):

        #Q is the rate of ionising photos in s^-1
        #nu is the frequency in GHz
        #output in erg/s/Hz

            T = 1e4 #temperature in K
            lum = Q/6.3e25 * (T/1e4)**0.45 * (nu)**(-0.1)

            return lum

        def synchrotron_lum(SFR, nu):

        #SFR in Msun/yr
        #nu is the frequency in GHz
            #output in erg/s/Hz

            ENT = 1.44
            ESNR = 0.06 * ENT
            alpha = -0.8

            comp1 = ESNR * (nu / 1.49)**(-0.5) + ENT * (nu / 1.49)**(alpha)
            nuSNCC = SFR * 0.015
            lum = comp1 * 1e30
            lum = lum * nuSNCC
    


            if(IncludeSelfAbsorption == True):
                for i in (range(100)):
                    print("SELF ABSORPTION IS INCLUDED")
                    
     
                    
                
                tau = 1
                lum = lum * np.e**(-tau)
            
            
            return lum







        total_mags_nod = np.array(seds_bands_lst[z][2], dtype=np.float64)
        ion_mag = total_mags_nod[1,:]
     #   ion_mag = ion_mag[ind109]
        ion_mag = ion_mag[ind108]
        q_ionis = ionising_photons(ion_mag, 912.0) #in s^-1
 
        selection_freq = (8.4, 5.0, 3.0, 1.4, 0.61, 0.325, 0.15)
        lum_radio = np.zeros(shape = (len(selection_freq), len(q_ionis)), dtype=np.float64)

        for i, nu in enumerate(selection_freq):
            lum_radio[i,:] = freefree_lum(q_ionis[:], nu) + synchrotron_lum(sfr_tot[:], nu)

            freefree = freefree_lum(q_ionis[:], nu)
            sync = synchrotron_lum(sfr_tot[:], nu)
        

        qIR_bressan = np.log10(lir_total/3.75e12) - np.log10(lum_radio[3,:]/1e7)
        
        freefree_lst.append(freefree)
        sync_lst.append(sync)
        
        print("This is qIR_bressan")
        print(qIR_bressan)
        inf_lst = np.where(qIR_bressan == np.inf)
        notinf_lst = np.where(qIR_bressan != np.inf)
        qir_lst.append(qIR_bressan)
        qir_bress.append(np.median(qIR_bressan[notinf_lst]))
        
        bress_rad_lum.append(lum_radio[3][notinf_lst]/1e7)

        bress_lir_total.append(np.median(lir_total[notinf_lst]))



    
    return qir_lst, qir_bress, bress_rad_lum, bress_lir_total, freefree_lst, sync_lst

def plot_lir_lf(plt, outdir, obsdir, LFs_dust, file_name):

    fig = plt.figure(figsize=(6,6))
    ytit = "$\\rm log_{10} (\\rm \\phi/\\, cMpc^{-3}\\, dex^{-1})$"
    xtit = "$\\rm log_{10} (L_{\\rm TIR}/L_{\\odot})$"
    xmin, xmax, ymin, ymax = 8.5, 12.8, -6, -2
    xleg = xmax - 0.3 * (xmax - xmin)
    yleg = ymax - 0.1 * (ymax - ymin)

    cols = ('Indigo','purple','Navy','MediumBlue','Green','MediumAquamarine','LightGreen','YellowGreen','Gold','Orange','Coral','OrangeRed','red','DarkRed','FireBrick','Crimson','IndianRed','LightCoral','Maroon','brown','Sienna','SaddleBrown','Chocolate','Peru','DarkGoldenrod','Goldenrod','SandyBrown')
    ax = fig.add_subplot(111)

    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.5, 0.5, 0.5, 0.5))

    for b in range(0,len(zlist)):
        inp = np.where(LFs_dust[b,:] != 0)
        x = xmf[inp]
        y = LFs_dust[b,inp]
        ax.plot(x, y[0]/dm, linestyle='solid',color=cols[b], label='z=%s' % str(zlist[b]))
        print('#redshift: %s' % str(zlist[b]))

        for a,b in zip(x,y[0]):
            print (a,b)
    common.prepare_legend(ax, cols, loc=3)
    common.savefig(outdir, fig, 'LIR_total_highz_'+file_name+'.pdf')



def main(model_dir, outdir, redshift_table, subvols, obsdir):
    print("This is the subvolume")
    print(subvols[0])

    plt = common.load_matplotlib()

    file_name = "eagle-rr14-radio-only"
    file_hdf5_sed = "Shark-SED-" + file_name + ".hdf5"

    fields_sed = {'SED/lir_dust': ('disk','bulge_t','total'),}
    fields_sed_bc = {'SED/lir_dust_contribution_bc':('disk', 'total')}
    fields_seds_nodust = {'SED/ab_nodust':('disk','bulge_t','total'),}
    fields_seds_dust = {'SED/ab_dust':('disk','bulge_t','total'),}

  #  fields = {'galaxies': ('mstars_disk', 'mstars_bulge','sfr_disk', 'sfr_burst','type','mgas_metals_bulge', 'mgas_metals_disk')}

    fields = {'galaxies': ('mstars_disk', 'mstars_bulge','sfr_disk', 'sfr_burst','type','mgas_metals_bulge', 'mgas_metals_disk','mgas_bulge','mgas_disk')}



    #Bands information:
#(0): "hst/ACS_update_sep07/wfc_f775w_t81", "hst/wfc3/IR/f160w",
    #(2): "F200W_JWST", "FUV_GALEX", "NUV_GALEX", "u_SDSS", "g_SDSS", "r_SDSS",
    #(8): "i_SDSS", "z_SDSS", "Y_VISTA", "J_VISTA", "H_VISTA", "K_VISTA",
    #(14): "W1_WISE", "I1_Spitzer", "I2_Spitzer", "W2_WISE", "I3_Spitzer",
    #(19): "I4_Spitzer", "W3_WISE", "W4_WISE", "P70_Herschel", "P100_Herschel",
    #(24): "P160_Herschel", "S250_Herschel", "S350_Herschel", "S450_JCMT",
    #(28): "S500_Herschel", "S850_JCMT", "FUV_Nathan", "Band9_ALMA",
    #(32): "Band8_ALMA", "Band7_ALMA", "Band6_ALMA", "Band4_ALMA",
    #(36): "Band3_ALMA", "BandL_VLA", "BandS_VLA"
    LFs_dust     = np.zeros(shape = (len(zlist), len(mbins)))
    print("This is zlist")
    print(zlist)
    
    seds_lir_lst = []
    
    hdf5_lir_lst = []
    
    seds_lir_bc_lst = []
    
    seds_bands_lst = []
    
    seds_nodust_lst = []
    
    lir_total_W_lst = []

    Teff_lst = []
    for index, snapshot in enumerate(redshift_table[zlist]):
        print("Will read snapshot %s" % (str(snapshot)))
        print("This is modle_dir")
        print(model_dir)

        hdf5_data = common.read_data(model_dir, snapshot, fields, subvols)

        seds_lir = common.read_photometry_data_variable_tau_screen(model_dir, snapshot, fields_sed, subvols, file_hdf5_sed)
        seds_lir_bc = common.read_photometry_data_variable_tau_screen(model_dir, snapshot, fields_sed_bc, subvols, file_hdf5_sed)


        seds_nodust = common.read_photometry_data_variable_tau_screen(model_dir, snapshot, fields_seds_nodust, subvols, file_hdf5_sed)
        
        seds_bands = common.read_photometry_data_variable_tau_screen(model_dir, snapshot, fields_seds_dust, subvols, file_hdf5_sed)

        (volh, h0, band_14, band_30, lir_total_W, Teff) = prepare_data(hdf5_data, seds_lir, seds_bands, seds_lir_bc, index, LFs_dust, obsdir)
        
        seds_nodust_lst.append(seds_nodust)
        
        seds_lir_lst.append(seds_lir)
        
        seds_lir_bc_lst.append(seds_lir_bc)
        
        seds_bands_lst.append(seds_bands)

        hdf5_lir_lst.append(hdf5_data)
        
        lir_total_W_lst.append(lir_total_W)
    
        Teff_lst.append(Teff)
    
    df,ab_lst=dataframe(hdf5_lir_lst,zlist,seds_bands_lst,seds_lir_lst,lir_total_W_lst,h0,Teff_lst)
  #  schr_vs_med(df)
    qir_z_plt(df)
  #  hist(df)
 #   lum_m(df,ab_lst)
 #   qir_vs_qir(df)

#    qir_with_m_plt(df)
  #  qir_z_plt(df)

   # q_m_m_z(df)
   # sfr_rad_lum(df)
   # rad_lum_func_plt(df,h0,volh)
    #rad_lum_func_plt_2(df,h0,volh)
 #   qir_metals_mass_bins(df)
 #   qir_v_dust_ff_sync(df)
#    fir_lir_mass(df)

 #   gas_metal_vs_stellar_mass(df)
#    lo_faro_plots(df)
#    GAMA_plots(df)

    LFs_dust = take_log(LFs_dust, volh, h0)
    plot_lir_lf(plt, outdir, obsdir, LFs_dust, file_name)
    

    
if __name__ == '__main__':
    main(*common.parse_args())
   
    #compile at home: python3 lir_highz_study-dale_z.py -m Shark -s medi-SURS -S ~/SHARK_Out/ -z ~/SHARK_Out/medi-SURS/redshift_list  -v 0 -o ~/SHARK_Out/output/
#compile  on magnus: 

#
# ICRAR - International Centre for Radio Astronomy Research
# (c) UWA - The University of Western Australia, 2018
# Copyright by UWA (in the framework of the ICRAR)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""Size plots"""

import functools
