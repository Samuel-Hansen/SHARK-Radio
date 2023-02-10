
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
from matplotlib.pyplot import cm
from collections import OrderedDict

from matplotlib.pyplot import cm
import matplotlib as mpl

import statistics as stat

#zlist = [0, 0.909822023685613,2.00391410007239, 3.0191633709527, 3.95972701662501, 5.02220991014863]
zlist = [0]
#zlist = [0, 1.0, 2.00391410007239, 3.0191633709527, 3.95972701662501, 5.02220991014863, 5.96592270612165, 7.05756323172746, 8.0235605165086, 8.94312532315157, 9.95650268434316]
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

def prepare_data(hdf5_data, seds, seds_bands, seds_bc, index, LFs_dust, obsdir):

    (h0, volh, mdisk, mbulge, sfr_disk, sfr_burst, typ,mgas_metals_bulge, mgas_metals_disk,mgas_bulge,mgas_disk,rgas_disk,rstar_disk) = hdf5_data
    bin_it = functools.partial(us.wmedians, xbins=xmf)

    lir_disk = seds[0]
    lir_bulge = seds[1]
    lir_total = seds[2] #total IR luminosity in units of Lsun

    lir_cont_bc = seds_bc[1]

    lir_total = np.array(lir_total, dtype=np.float64)
    Tbc = 50.0
    Tdiff = 25.0

    
    Teff = Tbc * lir_cont_bc[0] + Tdiff * (1 - lir_cont_bc[0]) #check if fraction

    #luminosity weight dust temperature

    lir_total_W = lir_total[0] * Lsun  #units of W
    seds_disk = seds_bands[0]
    seds_bulge = seds_bands[1]
    seds_total = seds_bands[2] 
     
    band_14 = seds_total[12,:] #the absolute magnitude of the 1.4 GHz band
    band_30 = seds_total[13,:] #the absolute magnitude of the 3.0 GHz band
    print("seds_total")
    print(print(seds_total))
    print("This is lenght of seds total")
    print(len(seds_total))

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
  
    return(volh, h0, band_14, band_30, lir_total_W,Teff)

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
        
        stellar_mass_lst = []
        stellar_mass_lst = [a+b for a,b in zip(hdf5_lir_lst[z][2],hdf5_lir_lst[z][3])]
        stellar_mass_lst = np.array(stellar_mass_lst)
        ind0 = np.where(stellar_mass_lst != 0)
        stellar_mass_lst = stellar_mass_lst[ind0]
        flux_lst = []
        
        for i in seds_bands_lst[z][2][9:16]: #iterates over different redshift (z), total liuminosity (index 2 of this list) and the 1.4GHz frequency (11 index)
            flux = 10**((i+48.6)/(-2.5)) * (4*np.pi*(d * 100)**2) / 1e7 #flux in W/Hz
            flux_lst.append(flux)
        
        qir_lst = [np.log10(a/(3.75*10**12))-np.log10(b) for a,b in zip(lir_total_W_lst[z], flux_lst[3])] #qir
     #   print("This is qir_lst")
     #   print(qir_lst[:4])
     #   print("THis is the length of qir_lst",len(qir_lst))
        qir_lst = np.array(qir_lst,dtype=np.float64)
        flux_lst = np.array(flux_lst,dtype=np.float64)
        
        lir_total = np.array(lir_total_W_lst[z], dtype=np.float64)
        ind109 = np.where(lir_total > 10**7 * Lsun)
        lir_total = lir_total[ind109]
        qir_lst = qir_lst[ind109]
        flux_lst = flux_lst[3][ind109]
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
        
        
        
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661)
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
    
    print("This is median qir_lst for Dale")
    print(qir_med_lst)
    print("This is median qir_Lst for Bressan")
    print(qir_bress)
    
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
 #   plt.xlabel("z")
 #   plt.ylabel("qir")
 #   cbar.set_label("Log Stellar Mass (Solar masses)")
 #   plt.show()

def dataframe(hdf5_lir_lst,zlist,seds_bands_lst,seds_lir_lst,lir_total_W_lst,h0,Teff_lst):
    h0 = float(h0)

    dist_mpc = 10 #distance in  pc
    d = 3.08567758128*10**17 #distance in m
    lstlst = []
    q = 0
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661)

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
        mgas_disk_array = np.array(hdf5_lir_lst[z][10])
        r_gas_disk_array = np.array(hdf5_lir_lst[z][11]) #half radius of the gas disk [cMpc/h]
        rstar_disk_array = np.array(hdf5_lir_lst[z][12]) # half-mass radius of the stellar disk
         
        
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
        r_gas_disk_array = r_gas_disk_array[ind0]
        rstar_disk_array = rstar_disk_array[ind0]
        
        lir_total_w = lir_total_W_lst[z]
        Teff = Teff_lst[z]
        
        redshift = zlist[z]
        
        print("This is seds_bands_lst[z][2]")
        print(seds_bands_lst[z][2])
        print("This is the length")
        print(len(seds_bands_lst[z][2]))
        flux_lst = []
        for i in seds_bands_lst[z][2][9:16]: #iterates over different redshift (z), total liuminosity (index 2 of this list) and the 1.4GHz frequency (11 index)

            flux = 10**((i+48.6)/(-2.5)) * (4*np.pi*(d * 100)**2) / 1e7 #flux in W/Hz
            flux_lst.append(flux)
            
            

        dale_qir_lst = [np.log10(a/(3.75*10**12))-np.log10(b) for a,b in zip(lir_total_W_lst[z], flux_lst[3])] #qir
        for j in range(len(mstars_disk_array)):
            lst = []
            mstars_disk = mstars_disk_array[j]
            mstars_bulge = mstars_bulge_array[j]
            sfr_disk = sfr_disk_array[j]
            sfr_burst = sfr_burst_array[j]
            mstars_tot = (mstars_disk + mstars_bulge)/h0
            sfr = (sfr_disk + sfr_burst) / 1e9 / h0
            sb_frac = sfr_burst/sfr #star burst fraction
            typ = type_array[j]
            gas_metallicity = ((bgm_array[j] + dgm_array[j])/(mgas_disk_array[j] + mgas_bulge_array[j]))/0.018 #gives gas metallicity
            lir_w = lir_total_w[j]
            teff = Teff[j]
            mgas = (mgas_disk_array[j] + mgas_bulge_array[j]) / h0 #gas mass
            r_gas_disk = r_gas_disk_array[j]*c_speed/h0
            rstar_disk = rstar_disk_array[j]*c_speed/h0

            qir_bress = qir_lst_bress[z][j]
            bress_rad_lum = bress_rad_lum_lst[z][j] #radio luminosity of bressan model in watts
            qir_dale = dale_qir_lst[j]
            fir_flux = lir_total_W_lst[z][j]
            rad_flux = flux_lst[3][j]
            freefree = freefree_lst[z][j] #freefree luminosity
            sync = sync_lst[z][j] #syncrhtotron luminosity
            
            m_ab_z0 = (-2.5)*10**(bress_rad_lum/(4*np.pi*(d*100)**2)) - 48.60
            
            
            
            
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
            lst.append(mgas)
            lst.append(freefree)
            lst.append(sync)
            lst.append(teff)
            lst.append(sb_frac)
            lst.append(r_gas_disk)
            lst.append(rstar_disk)
            lst.append(m_ab_z0)
            lstlst.append(lst)
    
    df = pd.DataFrame(lstlst, columns = ['z','mstars_disk','mstars_bulge','mstars_tot','sfr_disk','sfr_bulge','sfr','qir_bress','qir_dale','fir_flux','dale_rad_lum','type','gas_metal','lir_w','bress_rad_lum','mgas','freefree','sync','Teff','sb_frac','r_gas_disk','rstar_disk','m_ab_z0'])
  #  print("Hooray!")
    df['sfr/q'] = 'q'
    df['sf_test'] = 1
  #  print("This is df")
  #  print(df)
    sf_lst = []
    for z in zlist:
        sfr_z = df.loc[((df['z'] == z)&(df['mstars_tot'] > 10**(9))&(df['mstars_tot'] < 10**(10))&(df['type'] == 0)),'sfr']
        m_z = df.loc[((df['z'] == z)&(df['mstars_tot'] > 10**(9))&(df['mstars_tot'] < 10**(10))&(df['type'] == 0)),'mstars_tot']
        
        sfr_z = np.log10(sfr_z)
        m_z = np.log10(m_z)
        
        a,b = np.polyfit(m_z,sfr_z,1)
        df['sf_test'] = abs(np.log10(df['sfr']) - (a * np.log10(df['mstars_tot']) + b))
        df.loc[((df['z'] == z)&(df['sf_test'] < 0.3)), 'sfr/q'] = 'sf'
        
        
        
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

       # plt.plot(10**(x),10**(y),color = 'red',label = 'line fit')
       # plt.fill_between(x,y1,y2,color = 'red')
       # plt.scatter(sf_gal_m, sf_gal_sfr, label = 'Star forming')
       # plt.scatter(q_gal_m, q_gal_sfr, label = 'Quenched')
       # plt.yscale('log')
       # plt.xscale('log')
       # plt.xlabel("Stellar mass")
       # plt.ylabel("SFR")
       # plt.xlim(10**8,10**12)
       # plt.ylim(10**(-3),10**(3))
       # plt.legend()
       # plt.show()
        sf_lst.append((a,b))
    
   # print("This is the number of star forming galaxies")
   # print(len(df.loc[df['sfr/q'] == 'sf','sfr/q']))
  #  print(df)
    

    
    return df, sf_lst

    
def rad_lum_func_plt(df,seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst):
    h0 = 0.6751
    volh = 144703.125
    vol = volh/h0**3
    mlow = 0
    mupp = 40
    dm = 0.25
    mbins = np.arange(mlow, mupp, dm)
    xmf = mbins + dm/2.0 #setting up the bins
    
    zlst = 0
    
    zlist = [0, 0.909822023685613,2.00391410007239, 3.0191633709527, 3.95972701662501, 5.02220991014863]
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661)
    
    
    print("This is bress rad lum lst")
    print(bress_rad_lum_lst)
    
    df2 = pd.read_csv("/Users/samuelhansen/Documents/Masters/Masters_Thesis/bonata_data_z0.csv")
    
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

    
    
    
    

    
    lst = [0.1,0.3,0.5,0.7,0.9,2,4,6,8,10]
    
    fig, ax = plt.subplots(2,2)
    
    
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661)
    bress_rad_lum = bress_rad_lum_lst

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm    

    
    color = iter(cm.rainbow(np.linspace(0, 1, 10)))
    for i in lst:
            qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2)*i,0.0638298,0.0163,0.815661)
            blum = np.log10(bress_rad_lum_lst) #logging the dataset

            bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins
    
            bhist = bhist[0]/vol/dm
        
        

            #text = "$\tau$ = " + str(round(i,2))
            c = next(color)
            ax[0,0].plot(xmf, np.log10(bhist), c=c, label=i)
    
    
    
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661)
    

     
    ax[0,0].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="o",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020')
    ax[0,0].errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019')
    ax[0,0].errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020')
    ax[0,0].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020')
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661)
    bress_rad_lum = bress_rad_lum_lst

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(xmf,np.log10(bhist),False,False)
    ax[0,0].plot(mid_rad_lst,med_fir_lst,c = 'black')
    ax[0,0].fill_between(mid_rad_lst,low_fir_lst,high_fir_lst,alpha=0.5,color='black')
    ax[0,0].set_title('$L_{sync}$')
    
    
    y1 = -7.2
    y2 = 0.2
    x1 = 18.8
    x2 = 25.2
    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin=2.16426*10**(-2)*0.1,vmax = 2.16426*10**(-2)*10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
   # ax[1,1].title("Radio Luminosity Function")
   # ax[1,1].xlabel("Log10(L_1.4Ghz) (W/Hz)")
    ax[0,0].set_ylabel("$Log_{10}$(Φ) ($Mpc^{-3} dex^{-1}$)")
    ax[0,0].axes.get_xaxis().set_visible(False)
    ax[0,0].set_xlim(x1,x2)
    ax[0,0].set_ylim(y1,y2)
    leg = ax[0,0].legend(loc='upper center', bbox_to_anchor=(1.05, -1.222), ncol=7)
    leg.get_frame().set_linewidth(0.0)
    #ax[1,1].legend()
    
  #  lst = np.linspace((1/n)*1.944,n*1.944,11)

    color = iter(cm.rainbow(np.linspace(0, 1, 10)))
    for i in lst:
        qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06*i,0.0163,0.815661)
        blum = np.log10(bress_rad_lum_lst) #logging the dataset

        bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins
    
        bhist = bhist[0]/vol/dm
        
        

            #text = "$\tau$ = " + str(round(i,2))
        c = next(color)
        ax[0,1].plot(xmf, np.log10(bhist), c=c, label='_nolegend_')
    
    
    
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661)
    

     
    ax[0,1].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="o",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020')
    ax[0,1].errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019')
    ax[0,1].errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020')
    ax[0,1].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020')
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661)
    
    bress_rad_lum = bress_rad_lum_lst

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm    


    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(xmf,np.log10(bhist),False,False)
    ax[0,1].plot(mid_rad_lst,med_fir_lst,c = 'black')
    ax[0,1].fill_between(mid_rad_lst,low_fir_lst,high_fir_lst,alpha=0.5,color='black')
    ax[0,1].set_title('$C_{SNR}$')
    
    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin = 0.06*0.1,vmax = 0.06*10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
  #  cb1 = fig.colorbar(sm, ax=[ax[0, 1]])
  #  cb1.set_label('$C_{SNR}$')
   # ax[1,1].title("Radio Luminosity Function")
   # ax[1,1].xlabel("Log10(L_1.4Ghz) (W/Hz)")
    #ax[1,1].ylabel("Log10(Φ) (Mpc^-3 dex^-1)")
    ax[0,1].axes.get_xaxis().set_visible(False)
    ax[0,1].axes.get_yaxis().set_visible(False)
    ax[0,1].set_xlim(x1,x2)
    ax[0,1].set_ylim(y1,y2)
    
    
  #  lst = np.linspace(0,1,10)
    
    color = iter(cm.rainbow(np.linspace(0, 1, 10)))
    for i in lst:
        qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163*i,0.815661)
    
        blum = np.log10(bress_rad_lum_lst) #logging the dataset

        bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins
    
        bhist = bhist[0]/vol/dm
        
        

            #text = "$\tau$ = " + str(round(i,2))
        c = next(color)
        ax[1,0].plot(xmf, np.log10(bhist), c=c, label='_nolegend_')
    
    
    
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661)
    
    

     
    ax[1,0].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="o",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020')
    ax[1,0].errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019')
    ax[1,0].errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020')
    ax[1,0].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020')
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661)
    
    bress_rad_lum = bress_rad_lum_lst

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm    


    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(xmf,np.log10(bhist),False,False)
    ax[1,0].plot(mid_rad_lst,med_fir_lst,c = 'black')
    ax[1,0].fill_between(mid_rad_lst,low_fir_lst,high_fir_lst,alpha=0.5,color='black')
    ax[1,0].set_title('$nu_{CCSN}$')
    
    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin = 0.1*0.0163,vmax = 10*0.0163)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
  #  cb1 = fig.colorbar(sm, ax=[ax[1, 0]])
  #  cb1.set_label('$\nu_{CCSN}$')
   # ax[1,0].title("Radio Luminosity Function")
    ax[1,0].set_xlabel("$Log_{10}$($L_{1.4Ghz}$) (W/Hz)")
    ax[1,0].set_ylabel("Log$_{10}$(Φ) ($Mpc^{-3} dex^{-1})$")
    ax[1,0].set_xlim(x1,x2)
    ax[1,0].set_ylim(y1,y2)
    
   # lst = np.linspace(0,1,10)
    
    #ENT
    color = iter(cm.rainbow(np.linspace(0, 1, 10)))
    for i in lst:
        qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661*i)
    
    
        blum = np.log10(bress_rad_lum_lst) #logging the dataset

        bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins
    
        bhist = bhist[0]/vol/dm
        
        

            #text = "$\tau$ = " + str(round(i,2))
        c = next(color)
        ax[1,1].plot(xmf, np.log10(bhist), c=c, label=i)
            
    
    
    
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661)
    
    

     
    ax[1,1].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="o",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020')
    ax[1,1].errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019')
    ax[1,1].errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020')
    ax[1,1].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020')
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661)
    
    bress_rad_lum = bress_rad_lum_lst

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm    


    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(xmf,np.log10(bhist),False,False)
    ax[1,1].plot(mid_rad_lst,med_fir_lst,c = 'black')
    ax[1,1].fill_between(mid_rad_lst,low_fir_lst,high_fir_lst,alpha=0.5,color='black')
    ax[1,1].set_title('$alpha_{EI}$')
    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin = 0.1*0.815661,vmax = 10*0.815661)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
  #  cb1 = fig.colorbar(sm, ax=[ax[1, 1]])
   # cb1.set_label('alpha_{EI}')
  #  plt.title("Radio Luminosity Function")
    ax[1,1].set_xlabel("Log$_{10}$($L_{1.4Ghz}$) (W/Hz)")
  #  ax[0,1].axes.get_xaxis().set_visible(False)
    ax[1,1].axes.get_yaxis().set_visible(False)
   # plt.ylabel("Log10(Φ) (Mpc^-3 dex^-1)")
    ax[1,1].set_xlim(x1,x2)
    ax[1,1].set_ylim(y1,y2)
  #  plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=7)

    plt.subplots_adjust(
                    wspace=0.1,
                    hspace=0.1)
    fig.set_size_inches(9, 9)
    plt.savefig("constants_test.pdf",format='pdf')
    plt.show()

    
    
    ###Redshift Plots###

    vol = volh/h0**3
    mlow = 0
    mupp = 40
    dm = 0.25
    mbins = np.arange(mlow, mupp, dm)
    xmf = mbins + dm/2.0 #setting up the bins
   
    fig, ax = plt.subplots(2,3)

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
    
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661)
    
    bress_rad_lum = bress_rad_lum_lst

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm    


    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(xmf,np.log10(bhist),False,False)


    ax[0,0].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="o",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020')
    ax[0,0].errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019')
    ax[0,0].errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020')
    ax[0,0].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020')
    ax[0,0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0,0].set_title("z = 0")
#    ax[0,0].xlabel("Log10(L_1.4Ghz) (W/Hz)")
    ax[0,0].set_ylabel("$Log_{10}$($Phi$) ($Mpc^{-3} dex^{-1}$)")
    ax[0,0].set_xlim(19,25)
    ax[0,0].set_ylim(-7,0)

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
    
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661)
    
    bress_rad_lum = bress_rad_lum_lst

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm    


    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(xmf,np.log10(bhist),False,False)


    ax[0,1].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="o",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020')
    ax[0,1].errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019')
    ax[0,1].errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020')
    ax[0,1].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020')
    ax[0,1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0,1].set_title("z = 1.0")
#    ax[0,0].xlabel("Log10(L_1.4Ghz) (W/Hz)")
    #ax[0,0].set_ylabel("$Log_{10}$($Phi$) ($Mpc^{-3} dex^{-1}$)")
    ax[0,1].set_xlim(19,25)
    ax[0,1].set_ylim(-7,0)
    
    df2 = pd.read_csv("bonata_data_z2.csv")
    
    bonatoy = df2.loc[(df2['ref'] == 'Bonato2020'),'log_phi']
    bonatox = df2.loc[(df2['ref'] == 'Bonato2020'),'log_L_1.4']
    bonatoerr_up = df2.loc[(df2['ref'] == 'Bonato2020'),'err_sup_phi']
    bonatoerr_down = df2.loc[(df2['ref'] == 'Bonato2020'),'err_inf_phi']
    
    novaky = df2.loc[(df2['ref'] == 'Novak2017'),'log_phi']
    novakx = df2.loc[(df2['ref'] == 'Novak2017'),'log_L_1.4']
    novakerr_up = df2.loc[(df2['ref'] == 'Novak2017'),'err_sup_phi']
    novakerr_down = df2.loc[(df2['ref'] == 'Novak2017'),'err_inf_phi']
    
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661)
    
    bress_rad_lum = bress_rad_lum_lst

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm    


    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(xmf,np.log10(bhist),False,False)


    ax[0,2].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="o",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020')
    ax[0,2].errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019')
    ax[0,2].errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020')
    ax[0,2].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020')
    ax[0,2].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,2].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,2].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0,2].set_title("z = 2.0")
    ax[0,2].set_xlabel("$Log_{10}$($L_{1.4Ghz}$) (W/Hz)")
    #ax[0,0].set_ylabel("$Log_{10}$($Phi$) ($Mpc^{-3} dex^{-1}$)")
    ax[0,2].set_xlim(19,25)
    ax[0,2].set_ylim(-7,0)
    
    df2 = pd.read_csv("bonata_data_z3.csv")
    
    novaky = df2.loc[(df2['ref'] == 'Novak2017'),'log_phi']
    novakx = df2.loc[(df2['ref'] == 'Novak2017'),'log_L_1.4']
    novakerr_up = df2.loc[(df2['ref'] == 'Novak2017'),'err_sup_phi']
    novakerr_down = df2.loc[(df2['ref'] == 'Novak2017'),'err_inf_phi']
    
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661)
    
    bress_rad_lum = bress_rad_lum_lst

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm    


    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(xmf,np.log10(bhist),False,False)


    ax[1,0].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="o",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020')
    ax[1,0].errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019')
    ax[1,0].errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020')
    ax[1,0].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020')
    ax[1,0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,0].set_title("z = 3.0")
    ax[1,0].set_xlabel("$Log_{10}$($L_{1.4Ghz}$) (W/Hz)")
    #ax[0,0].set_ylabel("$Log_{10}$($Phi$) ($Mpc^{-3} dex^{-1}$)")
    ax[1,0].set_xlim(19,25)
    ax[1,0].set_ylim(-7,0)
    
    df2 = pd.read_csv("bonata_data_z4.csv")
    
    novaky = df2.loc[(df2['ref'] == 'Novak2017'),'log_phi']
    novakx = df2.loc[(df2['ref'] == 'Novak2017'),'log_L_1.4']
    novakerr_up = df2.loc[(df2['ref'] == 'Novak2017'),'err_sup_phi']
    novakerr_down = df2.loc[(df2['ref'] == 'Novak2017'),'err_inf_phi']
    
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661)
    
    bress_rad_lum = bress_rad_lum_lst

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm    


    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(xmf,np.log10(bhist),False,False)


    ax[1,1].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="o",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020')
    ax[1,1].errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019')
    ax[1,1].errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020')
    ax[1,1].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020')
    ax[1,1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,1].set_title("z = 4.0")
    ax[1,1].set_xlabel("$Log_{10}$($L_{1.4Ghz}$) (W/Hz)")
    #ax[0,0].set_ylabel("$Log_{10}$($Phi$) ($Mpc^{-3} dex^{-1}$)")
    ax[1,1].set_xlim(19,25)
    ax[1,1].set_ylim(-7,0)
    
    df2 = pd.read_csv("bonata_data_z5.csv")
    
    novaky = df2.loc[(df2['ref'] == 'Novak2017'),'log_phi']
    novakx = df2.loc[(df2['ref'] == 'Novak2017'),'log_L_1.4']
    novakerr_up = df2.loc[(df2['ref'] == 'Novak2017'),'err_sup_phi']
    novakerr_down = df2.loc[(df2['ref'] == 'Novak2017'),'err_inf_phi']
    
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst = bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,[0],2.16426*10**(-2),0.06,0.0163,0.815661)
    
    bress_rad_lum = bress_rad_lum_lst

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm    


    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(xmf,np.log10(bhist),False,False)


    ax[1,2].errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="o",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020')
    ax[1,2].errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019')
    ax[1,2].errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020')
    ax[1,2].errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020')
    ax[1,2].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,2].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,2].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,2].set_title("z = 5.0")
    ax[1,2].set_xlabel("$Log_{10}$($L_{1.4Ghz}$) (W/Hz)")
    #ax[0,0].set_ylabel("$Log_{10}$($Phi$) ($Mpc^{-3} dex^{-1}$)")
    ax[1,2].set_xlim(19,25)
    ax[1,2].set_ylim(-7,0)
    plt.legend()
   # fig.text(0.5, 0.04, "Log10(L_1.4Ghz) (W/Hz)", ha='center')
   # fig.text(0.04, 0.5, "Log10($\Phi$) (Mpc^-3 dex^-1)", va='center', rotation='vertical')
    fig.suptitle('Radio Luminosity function for different redshifts')
    plt.show()
    
def GAMA_plots(df):
    
    gd = pd.read_csv('Documents/Masters/Masters_Thesis/GAMA_data.csv') #GAMA Data

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
    
    fig, ax = plt.subplots(2,2)
    
    m_err = (gd['StellarMass_50']-gd['StellarMass_16'],gd['StellarMass_84']-gd['StellarMass_50'])
    sfr_err = (gd['SFR_50']-gd['SFR_16'],gd['SFR_84']-gd['SFR_50'])
    qir_bress = df.loc[((df['z'] < 0.1)&(df['fir_flux']>498972038.2)&(df['qir_bress']>0)&(df['qir_bress']<3)&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)&(df['sfr']>0.01)&(df['sfr/q'] =='sf')),'qir_bress']
    rad_bress = df.loc[((df['z'] < 0.1)&(df['fir_flux']>498972038.2)&(df['qir_bress']>0)&(df['qir_bress']<3)&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)&(df['sfr']>0.01)&(df['sfr/q'] =='sf')),'bress_rad_lum']
    fir_bress = df.loc[((df['z'] < 0.1)&(df['fir_flux']>498972038.2)&(df['qir_bress']>0)&(df['qir_bress']<3)&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)&(df['sfr']>0.01)&(df['sfr/q'] =='sf')),'fir_flux']
    m_shark = df.loc[((df['z'] < 0.1)&(df['fir_flux']>498972038.2)&(df['qir_bress']>0)&(df['qir_bress']<3)&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)&(df['sfr']>0.01)&(df['sfr/q'] =='sf')),'mstars_tot']
    sfr_shark = df.loc[((df['z'] < 0.1)&(df['fir_flux']>498972038.2)&(df['qir_bress']>0)&(df['qir_bress']<3)&(df['mstars_tot']>969913055.97323)&(df['mstars_tot']<529656145234.989)&(df['sfr']>0.01)&(df['sfr/q'] =='sf')),'sfr']
    
  #  m_shark = df.loc[((df['z'] < 0.5)&(df['bress_rad_lum']>5.60906e+19)&(df['fir_flux']>498972038.2)),'mstars_tot']
  #  sfr_shark = df.loc[((df['z'] < 0.5)&(df['bress_rad_lum']>5.60906e+19)&(df['fir_flux']>498972038.2)),'sfr']
    

    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(rad_bress,fir_bress,True,True)
    
    ax[0,0].hexbin(rad_bress,fir_bress,mincnt = 1, xscale='log',yscale='log',cmap='rainbow', gridsize = 30)
    ax[0,0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0,0].errorbar(radlum,firlum, xerr = radlum_err,yerr = firlum_err,fmt="o",markerfacecolor='white', markeredgecolor='black',label = 'GAMA Data',ecolor = 'black')
    ax[0,0].set_xscale('log')
    ax[0,0].set_yscale('log')
    ax[0,0].set_title("LIR vs. Lrad")
    ax[0,0].set_xlabel("Log10(Lrad (1.4GHz)) W/Hz")
    ax[0,0].set_ylabel("Log10(LIR) W")
   # cb = plt.colorbar(c)
   # cb.set_label('Number count')
   # plt.show()
    
    mid_m_lst,med_qir_lst,low_qir_lst,high_qir_lst = median_line(m_shark,qir_bress,True,False)
    

    
    ax[0,1].hexbin(m_shark,qir_bress,mincnt = 1, xscale='log',cmap='rainbow', gridsize = 30) 
    ax[0,1].plot(mid_m_lst,med_qir_lst,'black')
    ax[0,1].plot(mid_m_lst,low_qir_lst,'black',linestyle='dashed')   
    ax[0,1].plot(mid_m_lst,high_qir_lst,'black',linestyle='dashed')
    ax[0,1].errorbar(m,qir, xerr = m_err,yerr = qir_err,fmt="o",label = 'GAMA Data',markerfacecolor='white', markeredgecolor='black',ecolor = 'black')
    ax[0,1].set_xscale('log')
    ax[0,1].set_title("qir vs. Stellar mass")
    ax[0,1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[0,1].set_ylabel("qir")
    
    mid_sfr_lst,med_qir_lst,low_qir_lst,high_qir_lst = median_line(sfr_shark,qir_bress,True,False)
    
    ax[1,0].hexbin(sfr_shark,qir_bress,mincnt = 1, xscale='log',cmap='rainbow', gridsize = 30)
    
    ax[1,0].plot(mid_sfr_lst,med_qir_lst,'black')
    ax[1,0].plot(mid_sfr_lst,low_qir_lst,'black',linestyle='dashed')   
    ax[1,0].plot(mid_sfr_lst,high_qir_lst,'black',linestyle='dashed')
    
    ax[1,0].errorbar(sfr,qir, xerr =sfr_err,yerr = qir_err,fmt="o",label = 'GAMA Data',markerfacecolor='white', markeredgecolor='black',elinewidth = 0.3,ecolor = 'black',alpha=0.5)
    ax[1,0].set_xscale('log')
    ax[1,0].set_title("GAMA Data Comparison - qir vs. SFR")
    ax[1,0].set_xlabel("Log10(SFR/$M_\odot$/yr)")
    ax[1,0].set_ylabel("qir")
        
    mid_m_lst,med_sfr_lst,low_sfr_lst,high_sfr_lst = median_line(m_shark,sfr_shark,True,True)
    
    c = ax[1,1].hexbin(m_shark,sfr_shark,mincnt = 1, xscale='log',yscale = 'log',cmap='rainbow', gridsize = 30)
    
    ax[1,1].plot(mid_m_lst,med_sfr_lst,'black')
    ax[1,1].plot(mid_m_lst,low_sfr_lst,'black',linestyle='dashed')   
    ax[1,1].plot(mid_m_lst,high_sfr_lst,'black',linestyle='dashed')
    

    ax[1,1].errorbar(m,sfr, xerr =sfr_err,yerr = qir_err,fmt="o",label = 'GAMA Data',markerfacecolor='white', markeredgecolor='black',ecolor = 'black')
    ax[1,1].set_title("GAMA Data Comparison - SFR vs. Stellar Mass")
    ax[1,1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[1,1].set_ylabel("Log10(SFR/$M_\odot$/yr)")
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('qir')
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
    

def bressan_model(seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst,h0,zlist,LNT,E_ratio,nu_ccsn,alpha_EI):
    
    
    
    Lsunwatts = 3.846e26
    h = 6.6261e-27 #cm2 g s-1
    d10pc = 3.086e+19
    dfac = 4 * PI * d10pc**2
    qir_bress = []
    bress_rad_lum = []
    bress_lir_total = []
    qir_lst = []
    freefree_lst = []
    sync_lst = []
    ion_phot_lst = []
    
    IncludeSelfAbsorption = False
    
    for z in range(len(zlist)):
        sfr_disk = np.array(hdf5_lir_lst[z][4])
        sfr_disk_2 = []
        sfr_bulge = np.array(hdf5_lir_lst[z][5])
        sfr_bulge_2 = []
        sfr_tot = []
        sfr_tot = sfr_bulge + sfr_disk
        mdisk = np.array(hdf5_lir_lst[z][2])
        mbulge = np.array(hdf5_lir_lst[z][3])
        r_gas_disk_array = np.array(hdf5_lir_lst[z][11]) #half radius of the gas disk [cMpc/h]
        
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
        lir_total = lir_total[ind108]
        ms = ms[ind108]
        sfr_tot= sfr_tot[ind108]
        r_gas_disk_array = r_gas_disk_array[ind0]
        r_gas_disk_array = r_gas_disk_array[ind108]
        
        
        
        print("This is length of sfr")
        print(len(sfr_tot))

        n = 0
        
        def tau_factor(m, wave, rgas_disk,nu):
            wave_m = wave * 1e-10 #wavelength in m
            wave_cm = wave_m  * 1e2 #wavelength in cm
            
            T = 1e4 #Temperature in K
            
            freq = (c_speed/ wave) / 1e9 #Frequency in GHz
            
            ne = 10.0**((m + 48.6) / (-2.5)) * dfac * freq * wave_cm #number of electrons
            
            l = (rgas_disk/h/c_speed)*1e6 #distance in pc
            
            tau = 8.2 * 10**(-2) * T**(-1.35) * nu**(-2.1) * (ne**2 * l)
            
            return tau

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

        def synchrotron_lum(SFR, nu,LNT,E_ratio,nu_ccsn,alpha_EI):
            
            
            #LNT = 6.1*10**(-2)*(1.49/0.408)**(-0.8) ##2.16 x10^(-2) W/Hz/10^(21)
            ENT = LNT/nu_ccsn ##1.31 x 10^(23) W/Hz/10^(21)
           # nu_ccsn = 0.0163
            
            
            
            ESNR = ENT * E_ratio ##0.0784 x 10^(23)W/Hz/10^(21)
            EEI = ENT*(1-E_ratio) ##1.23 x 10^(23) W/Hz/10^(21)
           # alpha = 0.815661

            comp1 = ESNR * (nu / 1.49)**(-0.5) + (EEI * (nu / 1.49)**(-alpha_EI))
            nuSNCC = SFR * 0.011148
            lum = comp1 * 1e30
            lum = lum * nuSNCC
            
            if(IncludeSelfAbsorption == True):
                for i in (range(100)):
                    print("SELF ABSORPTION IS INCLUDED")

                
                comp1 = ESNR * (nu / 1.49)**(-0.5) + ENT * (nu / 1.49)**(alpha) * np.e**(-tau)
            
            
            return lum


        total_mags_nod = np.array(seds_bands_lst[z][2], dtype=np.float64)
        ion_mag = total_mags_nod[1,:]
     #   ion_mag = ion_mag[ind109]
        ion_mag = ion_mag[ind108]
        q_ionis = ionising_photons(ion_mag, 912.0) #in s^-1
 
        selection_freq = (8.4, 5.0, 3.0, 1.4, 0.61, 0.325, 0.15)
        lum_radio = np.zeros(shape = (len(selection_freq), len(q_ionis)), dtype=np.float64)
        freefree = np.zeros(shape = (len(selection_freq), len(q_ionis)), dtype=np.float64)
        sync = np.zeros(shape = (len(selection_freq), len(q_ionis)), dtype=np.float64)
        for i, nu in enumerate(selection_freq):
            lum_radio[i,:] = freefree_lum(q_ionis[:], nu) + synchrotron_lum(sfr_tot[:], nu,LNT,E_ratio,nu_ccsn,alpha_EI)

            freefree[i,:] = freefree_lum(q_ionis[:], nu)/1e7
            
            sync[i,:] = synchrotron_lum(sfr_tot[:], nu,LNT,E_ratio,nu_ccsn,alpha_EI)/1e7
            

        
        
        qIR_bressan = np.log10(lir_total/3.75e12) - np.log10(lum_radio[3,:]/1e7)
        

        inf_lst = np.where(qIR_bressan == np.inf)
        notinf_lst = np.where(qIR_bressan != np.inf)
        qir_lst.append(qIR_bressan)
        qir_bress.append(np.median(qIR_bressan[notinf_lst]))
        
        bress_rad_lum.append(lum_radio[3][notinf_lst]/1e7)

        bress_lir_total.append(np.median(lir_total[notinf_lst]))
        
        

        
        freefree_lst.append(freefree[3])
        sync_lst.append(sync[3])
           
            
        ion_phot_lst.append(q_ionis)
    
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
       # print('#redshift: %s' % str(zlist[b]))

     #   for a,b in zip(x,y[0]):
         #   print (a,b)
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
    fields_seds_dust_ap = {'SED/ap_dust':('disk','bulge_t','total'),} 

    fields = {'galaxies': ('mstars_disk', 'mstars_bulge','sfr_disk', 'sfr_burst','type','mgas_metals_bulge', 'mgas_metals_disk','mgas_bulge','mgas_disk','rgas_disk','rstar_disk')}
    
    
    
    #(0): "z_SDSS", "Band_ionising_photons", "FUV_Nathan", "Band9_ALMA",
    #(4): "Band8_ALMA", "Band7_ALMA", "Band6_ALMA", "Band4_ALMA", "Band3_ALMA",
    #(9): "BandX_VLA", "BandC_VLA", "BandS_VLA", "BandL_VLA", "Band_610MHz",
    #(14): "Band_325MHz", "Band_150MHz"

    
    #
    
    
    
    #Bands information:
    #(0): "hst/ACS_update_sep07/wfc_f775w_t81:x", "hst/wfc3/IR/f160w",
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
        hdf5_data = common.read_data(model_dir, snapshot, fields, subvols)

        seds_lir = common.read_photometry_data_variable_tau_screen(model_dir, snapshot, fields_sed, subvols, file_hdf5_sed)
        seds_lir_bc = common.read_photometry_data_variable_tau_screen(model_dir, snapshot, fields_sed_bc, subvols, file_hdf5_sed)


        seds_nodust = common.read_photometry_data_variable_tau_screen(model_dir, snapshot, fields_seds_nodust, subvols, file_hdf5_sed)
        
        seds_bands = common.read_photometry_data_variable_tau_screen(model_dir, snapshot, fields_seds_dust, subvols, file_hdf5_sed)

        (volh, h0, band_14, band_30, lir_total_W,Teff) = prepare_data(hdf5_data, seds_lir, seds_bands, seds_lir_bc, index, LFs_dust, obsdir)
        
        seds_nodust_lst.append(seds_nodust)
        
        seds_lir_lst.append(seds_lir)
        
        seds_lir_bc_lst.append(seds_lir_bc)
        
        seds_bands_lst.append(seds_bands)

        hdf5_lir_lst.append(hdf5_data)
        
        lir_total_W_lst.append(lir_total_W)
        
        Teff_lst.append(Teff)
        
        
    


  
    df,sf_lst = dataframe(hdf5_lir_lst,zlist,seds_bands_lst,seds_lir_lst,lir_total_W_lst,h0,Teff_lst)

    rad_lum_func_plt(df,seds_bands_lst,seds_lir_lst,hdf5_lir_lst,lir_total_W_lst)
#    GAMA_plots(df)
#    mstars_disk = hdf5_data[2]
#    mstars_bulge = hdf5_data[3]
 #   mstars_tot = []
  #  for i, j in zip(mstars_disk, mstars_bulge):
  #      k = i + j
  #      if k > 0: 
  #          mstars_tot.append(k)
#    radio_convert(h0, zlist,seds_lir_lst,hdf5_lir_lst,seds_lir_bc_lst,seds_bands_lst, lir_total_W_lst)
    def take_log(x,v,h):
        x = x / (v / h**3.0)
        ind = np.where(x > 0)
        x[ind] = np.log10(x[ind])
        return x

    LFs_dust = take_log(LFs_dust, volh, h0)
    plot_lir_lf(plt, outdir, obsdir, LFs_dust, file_name)
    

    
if __name__ == '__main__':
    main(*common.parse_args())
   
    #compile at home: python3 lir_highz_study-dale_z.py -m Shark -s medi-SURS -S ~/SHARK_Out/ -z ~/SHARK_Out/medi-SURS/redshift_list  -v 0 -o ~/SHARK_Out/output/
    #compile on magnus: 
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