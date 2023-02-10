
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
import matplotlib as mpl
from scipy.spatial import distance_matrix
from collections import OrderedDict


import statistics as stat
zlist=[0, 0.194738848008908]

#zlist = [0, 0.194738848008908, 0.909822023685613, 2.00391410007239, 3.0191633709527, 3.95972701662501, 5.02220991014863, 5.96592270612165, 7.05756323172746, 8.0235605165086,8.94312532315157, 9.95650268434316]
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

    (h0, volh, mdisk, mbulge, sfr_disk, sfr_burst, typ,mgas_metals_bulge, mgas_metals_disk,mgas_bulge,mgas_disk,rgas_disk,rstar_disk,matom_disk,mgas_metals_disk,mmol_disk,position_x,position_y, position_z) = hdf5_data
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
        
        
        
    qir_bress, bress_rad_lum, bress_lir_total = bressan_model(seds_nodust_lst,seds_bands_lst,hdf5_lir_lst,lir_total_W_lst,h0,zlist)
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

def dataframe(hdf5_lir_lst,zlist,seds_bands_lst,seds_lir_lst,lir_total_W_lst,seds_nodust_lst,h0,Teff_lst):
    h0 = float(h0)

    dist_mpc = 10 #distance in  pc
    d = 3.08567758128*10**17 #distance in m
    lstlst = []
    q = 0
    qir_lst_bress, qir_bress, bress_rad_lum_lst, bress_lir_total,freefree_lst, sync_lst, q_ionis = bressan_model(seds_nodust_lst,seds_bands_lst,hdf5_lir_lst,lir_total_W_lst,h0,zlist)

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
        matom_disk_array = np.array(hdf5_lir_lst[z][13]) #atomic gas mass in disk
        mgas_metals_disk_array = np.array(hdf5_lir_lst[z][14]) #mass of metals in gas in disk
        mmol_disk_array = np.array(hdf5_lir_lst[z][15]) #molecular gas mass in disk
        pos_x = np.array(hdf5_lir_lst[z][16]) #position x
        pos_y = np.array(hdf5_lir_lst[z][17]) #position y
        pos_z = np.array(hdf5_lir_lst[z][18]) #position z
        
        ind0 = np.where(stellar_mass_array > 10**8)
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
        matom_disk_array = matom_disk_array[ind0]
        mgas_metals_disk_array = mgas_metals_disk_array[ind0]
        mmol_disk_array = mmol_disk_array[ind0]
        pos_x = pos_x[ind0]
        pos_y = pos_y[ind0]
        pos_z = pos_z[ind0]
        
        print("This is posx")
        print(pos_x)
        
        lir_total_w = lir_total_W_lst[z]
        Teff = Teff_lst[z]
        
        redshift = zlist[z]
        
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
            r_gas_disk = r_gas_disk_array[j]/h0
            rstar_disk = rstar_disk_array[j]/h0
            mgas_disk = mgas_disk_array[j]/h0
            matom_disk = matom_disk_array[j]/h0
            mgas_metals_disk = mgas_metals_disk_array[j]/h0
            mmol_disk = mmol_disk_array[j]/h0
            pos_x_temp = pos_x[j]
            pos_y_temp = pos_y[j]
            pos_z_temp = pos_z[j]
            
            gas_surf_dens = mgas_disk/(2*r_gas_disk**2*np.pi) #gas surface density
            
            

            qir_bress = qir_lst_bress[z][j]
            bress_rad_lum = bress_rad_lum_lst[z][j] #radio luminosity of bressan model in watts
            qir_dale = dale_qir_lst[j]
            fir_flux = lir_total_W_lst[z][j]
            rad_flux = flux_lst[3][j]
            freefree = freefree_lst[z][j] #freefree luminosity
            sync = sync_lst[z][j] #syncrhtotron luminosity
            
            ionising_photons = q_ionis[z][j]
            
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
            lst.append(ionising_photons)
            lst.append(mgas_disk)
            lst.append(matom_disk)
            lst.append(mgas_metals_disk)
            lst.append(mmol_disk)
            lst.append(gas_surf_dens)
            lst.append(pos_x_temp)
            lst.append(pos_y_temp)
            lst.append(pos_z_temp)
            lstlst.append(lst)
    
    df = pd.DataFrame(lstlst, columns = ['z','mstars_disk','mstars_bulge','mstars_tot','sfr_disk','sfr_bulge','sfr','qir_bress','qir_dale','fir_flux','dale_rad_lum','type','gas_metal','lir_w','bress_rad_lum','mgas','freefree','sync','Teff','sb_frac','r_gas_disk','rstar_disk','m_ab_z0','ionising_photons','mgas_disk','matom_disk','mgas_metals_disk','mmol_disk','gas_surf_dens','pos_x','pos_y','pos_z'])
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
        try:
            a,b = np.polyfit(m_z,sfr_z,1)
        except:
            print("using a and b from last round")

        df['sf_test'] = abs(np.log10(df['sfr']) - (a * np.log10(df['mstars_tot']) + b))
        df.loc[((df['z'] == z)&(df['sf_test'] < 0.3)), 'sfr/q'] = 'sf'
        
        
        
        sf_gal_sfr = df.loc[((df['z'] == z)&(df['sfr/q'] == 'sf')),'sfr']
        sf_gal_m = df.loc[((df['z'] == z)&(df['sfr/q'] == 'sf')),'mstars_tot']
        q_gal_sfr = df.loc[((df['z'] == z)&(df['sfr/q'] == 'q')),'sfr']
        q_gal_m = df.loc[((df['z'] == z)&(df['sfr/q'] == 'q')),'mstars_tot'] 
        strz = str(z)
        tit = "z = " + strz[:6]
        
        
        
       # plt.title(tit)
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
    print("This is sfr/sync")
    for i in range(99):
        print(df['sfr']/df['sync'])
    
    
    
    return df, sf_lst
    
    
def GAMA_plots(df):
    
    gd = pd.read_csv('GAMA_data.csv') #GAMA Data

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
    
    vmin = 4
    vmax = 30
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(rad_bress,fir_bress,True,True)
    
    ax[0,0].hexbin(rad_bress,fir_bress,mincnt = 1, xscale='log',yscale='log',cmap='rainbow', gridsize = 30,vmin= vmin, vmax = vmax)
    ax[0,0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0,0].errorbar(radlum,firlum, xerr = radlum_err,yerr = firlum_err,fmt="o",markerfacecolor='white', markeredgecolor='black',label = 'GAMA Data',ecolor='black',elinewidth = 0.3)
    ax[0,0].set_xscale('log')
    ax[0,0].set_yscale('log')
    ax[0,0].set_title('L$_{IR}$ vs. L$_{rad}$')
    ax[0,0].set_xlabel('log$_{10}$(L$_{rad/1.4GHz}$)/ W/Hz')
    ax[0,0].set_ylabel('log$_{10}$(L$_{IR}$) W')
   # cb = plt.colorbar(c)
   # cb.set_label('Number count')
   # plt.show()
    
    mid_m_lst,med_qir_lst,low_qir_lst,high_qir_lst = median_line(m_shark,qir_bress,True,False)
    

    
    ax[0,1].hexbin(m_shark,qir_bress,mincnt = 1, xscale='log',cmap='rainbow', gridsize = 30,vmin= vmin, vmax = vmax) 
    ax[0,1].plot(mid_m_lst,med_qir_lst,'black')
    ax[0,1].plot(mid_m_lst,low_qir_lst,'black',linestyle='dashed')   
    ax[0,1].plot(mid_m_lst,high_qir_lst,'black',linestyle='dashed')
    ax[0,1].errorbar(m,qir, xerr = m_err,yerr = qir_err,fmt="o",label = 'GAMA Data',markerfacecolor='white', markeredgecolor='black',ecolor = 'black',elinewidth = 0.3)
    ax[0,1].set_xscale('log')
    ax[0,1].set_title("qir vs. Stellar mass")
    ax[0,1].set_xlabel("log$_{10}$(Stellar Mass/$M_\odot$)")
    ax[0,1].set_ylabel("qir")
    
    mid_sfr_lst,med_qir_lst,low_qir_lst,high_qir_lst = median_line(sfr_shark,qir_bress,True,False)
    
    ax[1,1].hexbin(sfr_shark,qir_bress,mincnt = 4, xscale='log',cmap='rainbow', gridsize = 30,vmin= vmin, vmax = vmax)
    
    ax[1,1].plot(mid_sfr_lst,med_qir_lst,'black')
    ax[1,1].plot(mid_sfr_lst,low_qir_lst,'black',linestyle='dashed')   
    ax[1,1].plot(mid_sfr_lst,high_qir_lst,'black',linestyle='dashed')
    
    ax[1,1].errorbar(sfr,qir, xerr =sfr_err,yerr = qir_err,fmt="o",label = 'GAMA Data',markerfacecolor='white', markeredgecolor='black',ecolor = 'black',elinewidth = 0.3)
    ax[1,1].set_xscale('log')
    ax[1,1].set_title("qir vs. SFR")
    ax[1,1].set_xlabel("log$_{10}$(SFR/$M_\odot$/yr)")
    ax[1,1].set_ylabel("qir")
        
    mid_m_lst,med_sfr_lst,low_sfr_lst,high_sfr_lst = median_line(m_shark,sfr_shark,True,True)
    
    c = ax[1,0].hexbin(m_shark,sfr_shark,mincnt = 1, xscale='log',yscale = 'log',cmap='rainbow', gridsize = 30,vmin= vmin, vmax = vmax)
    
    ax[1,0].plot(mid_m_lst,med_sfr_lst,'black')
    ax[1,0].plot(mid_m_lst,low_sfr_lst,'black',linestyle='dashed')   
    ax[1,0].plot(mid_m_lst,high_sfr_lst,'black',linestyle='dashed')
    

    ax[1,0].errorbar(m,sfr, xerr =sfr_err,yerr = qir_err,fmt="o",label = 'GAMA Data',markerfacecolor='white', markeredgecolor='black',ecolor = 'black',elinewidth = 0.3)
    ax[1,0].set_title("SFR vs. Stellar Mass")
    ax[1,0].set_xlabel("log$_{10}$(Stellar Mass/$M_\odot$)")
    ax[1,0].set_ylabel("log$_{10}$(SFR/$M_\odot$/yr)")
  #  cb = fig.colorbar(c, ax=ax.ravel().tolist())
  #  cb.set_label('Number Count')
    fig.set_size_inches(9, 9)
    plt.tight_layout()
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Number Count')
    plt.savefig("GAMA_plot.pdf",format='pdf')
    plt.show()
    
    
def lo_faro_plots(df):
    

    Lsun = 3.828 * 10**26 #W
    df_lf = pd.read_csv('Documents/Masters/Masters_Thesis/french_paper_2_pg16.csv')
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

    
    rad_bress_1 = df.loc[((df['z'] < 1.05)&(df['z']>0.8)&(df['fir_flux']>7.88568e+37)&(df['z'] == 0.909822023685613)),'bress_rad_lum']
    fir_bress_1 = df.loc[((df['z'] < 1.05)&(df['z']>0.8)&(df['fir_flux']>7.88568e+37)&(df['z'] == 0.909822023685613)),'fir_flux']
    
    rad_bress_2 = df.loc[((df['z'] < 2.2)&(df['z']>1.5)&(df['fir_flux']>7.88568e+37)&(df['z'] == 2.00391410007239)),'bress_rad_lum']
    fir_bress_2 = df.loc[((df['z'] < 2.2)&(df['z']>1.5)&(df['fir_flux']>7.88568e+37)&(df['z'] == 2.00391410007239)),'fir_flux']
    
    rad_bress_1 = df.loc[((df['fir_flux']>3e37)& (df['mstars_tot']>2e8)&(df['z'] == 0.909822023685613)),'bress_rad_lum']
    fir_bress_1 = df.loc[((df['fir_flux']>3e37)& (df['mstars_tot']>2e8)&(df['z'] == 0.909822023685613)),'fir_flux']
    
    rad_bress_2 = df.loc[((df['fir_flux']>3e37)& (df['mstars_tot']>2e8)&(df['z'] == 2.00391410007239)),'bress_rad_lum']
    fir_bress_2 = df.loc[((df['fir_flux']>3e37)& (df['mstars_tot']>2e8)&(df['z'] == 2.00391410007239)),'fir_flux']
    
    rad_bress_1 = np.log10(rad_bress_1)
    
    rad_bress_2 = np.log10(rad_bress_2)
    fir_bress_2 = np.log10(fir_bress_2)
    fir_bress_1 = np.log10(fir_bress_1)
    

    
    Lrad2_uncert = np.log10(df_lf.loc[(df_lf['z'] > 1.5),'L1.4_err'])

    
    
    
#['z','mstars_disk','mstars_bulge','mstars_tot','sfr_disk','sfr_bulge','sfr','qir_bress','qir_dale','fir_flux','dale_rad_lum','type','gas_metal','lir_w','bress_rad_lum','mgas','freefree','sync','Teff','sb_frac','r_gas_disk'])


    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(rad_bress_1,fir_bress_1,True,True)
    


    fig, ax = plt.subplots(2,2)
    fig.set_size_inches(15, 11)
    c = ax[0,0].hexbin(rad_bress_1,fir_bress_1,mincnt = 1, cmap='rainbow', gridsize = 30,vmin = 0, vmax = 1000)
    ax[0,0].scatter(Lrad1,LIR1,marker="o",label = 'GAMA Data',color='white', edgecolor='black')
    ax[0,0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0,0].set_xlim(20,26)
    ax[0,0].set_ylim(36,40)
    ax[0,0].set_xlabel("Log10(L 1.4 GHz) W/Hz")
    ax[0,0].set_ylabel("Log10(LFIR) W/Hz")
    ax[0,0].set_title("z ~ 1")
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(rad_bress_2,fir_bress_2,True,True)
    ax[0,1].hexbin(rad_bress_2,fir_bress_2,mincnt = 1, cmap='Greys', gridsize = 30,vmin = 0, vmax = 1000)
    ax[0,1].scatter(Lrad2,LIR2,marker="o",label = 'GAMA Data',color='white', edgecolor='black')
    ax[0,1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0,1].set_xlabel("Log10(L 1.4 GHz) W/Hz")
    ax[0,1].set_ylabel("Log10(LFIR) W/Hz")
   # ax[0,1].text('LIR vs. Lrad', y=1.0, pad=-14)
    ax[0,1].set_title("z ~ 2")

    
    m_1 = df.loc[((df['fir_flux']>3e37)& (df['mstars_tot']>2e8)&(df['z'] == 0.909822023685613)),'mstars_tot']
    sfr_1 = df.loc[((df['fir_flux']>3e37)& (df['mstars_tot']>2e8)&(df['z'] == 0.909822023685613)),'sfr']
    
    m_2 = df.loc[((df['fir_flux']>3e37)& (df['mstars_tot']>2e8)&(df['z'] == 2.00391410007239)),'mstars_tot']
    sfr_2 = df.loc[((df['fir_flux']>3e37)& (df['mstars_tot']>2e8)&(df['z'] == 2.00391410007239)),'sfr']
    
    m_1 = np.log10(m_1)
    m_2 = np.log10(m_2)
    
    sfr_1 = np.log10(sfr_1)
    sfr_2 = np.log10(sfr_2)
    
    qir1 = df.loc[((df['fir_flux']>3e37)& (df['mstars_tot']>2e8)&(df['z'] == 0.909822023685613)),'qir_bress']
    qir2 = df.loc[((df['fir_flux']>3e37)& (df['mstars_tot']>2e8)&(df['z'] == 2.00391410007239)),'qir_bress']
    
    
    m_lf_1 = np.log10(df_lf.loc[(df_lf['z'] < 1.5),'M_star'])
    m_lf_2 = np.log10(df_lf.loc[(df_lf['z'] > 1.5),'M_star'])
    
    sfr_lf_1 = np.log10(df_lf.loc[(df_lf['z'] < 1.5),'SFR10'])
    sfr_lf_2 = np.log10(df_lf.loc[(df_lf['z'] > 1.5),'SFR10'])
    
    qir_lf_1 = df_lf.loc[(df_lf['z'] < 1.5),'qTIR']
    qir_lf_2 = df_lf.loc[(df_lf['z'] > 1.5),'qTIR']
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(m_1,sfr_1,False,False)

    c = ax[1,0].hexbin(sfr_1,m_1,mincnt = 1, cmap='rainbow', gridsize = 30)
    ax[1,0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,0].scatter(m_lf_1,qir_lf_1,marker="o",label = 'GAMA Data',color='white', edgecolor='black')

    ax[1,0].set_xlim(8,12)
    ax[1,0].set_ylim(0,3.5)
    ax[1,0].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[1,0].set_ylabel("qir")
    ax[1,0].set_title("z ~ 1")
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(m_2,sfr_2,False,False)
    
    ax[1,1].hexbin(sfr_2,m_2,mincnt = 1, cmap='Greys', gridsize = 30)
    ax[1,1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,1].scatter(m_lf_2,qir_lf_2,marker="o",label = 'GAMA Data',color='white', edgecolor='black')
    ax[1,1].set_xlim(8,12)
    ax[1,1].set_ylim(0,3.5)
    ax[1,1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[1,1].set_ylabel("Log10(SFR/$M_\odot$/yr)")
    ax[1,1].set_title("z ~ 2")
    
    fig.text(0.4, 0.9, "LIR vs. Lrad",weight='bold')
    fig.text(0.4, 0.475, "SFR vs. Stellar Mass$",weight='bold')
    
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Number Count')
   # fig.text(0.5, 0.04, "Log10(Stellar Mass/$M_\odot$) ", ha='center')
   # fig.text(0.04, 0.5, "qir", va='center', rotation='vertical')
   # fig.suptitle('qir vs. Stellar Mass - Lo Faro et al.')
    plt.show()
    
    m_1 = df.loc[((df['fir_flux']>3e37)& (df['mstars_tot']>2e8)&(df['z'] == 0.909822023685613)),'mstars_tot']
    sfr_1 = df.loc[((df['fir_flux']>3e37)& (df['mstars_tot']>2e8)&(df['z'] == 0.909822023685613)),'sfr']
    
    m_2 = df.loc[((df['fir_flux']>3e37)& (df['mstars_tot']>2e8)&(df['z'] == 2.00391410007239)),'mstars_tot']
    sfr_2 = df.loc[((df['fir_flux']>3e37)& (df['mstars_tot']>2e8)&(df['z'] == 2.00391410007239)),'sfr']
    
    m_1 = np.log10(m_1)
    m_2 = np.log10(m_2)
    
    sfr_1 = np.log10(sfr_1)
    sfr_2 = np.log10(sfr_2)
    
    qir1 = df.loc[((df['fir_flux']>3e37)& (df['mstars_tot']>2e8)&(df['z'] == 0.909822023685613)),'qir_bress']
    qir2 = df.loc[((df['fir_flux']>3e37)& (df['mstars_tot']>2e8)&(df['z'] == 2.00391410007239)),'qir_bress']
    
    fig, ax = plt.subplots(2,2)
    fig.set_size_inches(15, 11)
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(sfr_1,qir1,False,False)
    
    c = ax[0,0].hexbin(sfr_1,qir1,mincnt = 1, cmap='rainbow', gridsize = 30)
    ax[0,0].scatter(sfr_lf_1,qir_lf_1,marker="o",label = 'GAMA Data',color='white', edgecolor='black')
    ax[0,0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
   # ax[0,0].set_xlim(1,3)
   # ax[0,0].set_ylim(0,3.5)
    ax[0,0].set_title("z ~ 1")
    ax[0,0].set_xlabel("Log10(SFR/$M_\odot$/yr)")
    ax[0,0].set_ylabel("qir")
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(sfr_2,qir2,False,False)
    ax[0,1].hexbin(sfr_2,qir2,mincnt = 1, cmap='rainbow', gridsize = 30,vmin = 0, vmax = 1000)
    ax[0,1].scatter(sfr_lf_2,qir_lf_2,marker="o",label = 'GAMA Data',color='white', edgecolor='black')
    ax[0,1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0,1].set_xlabel("Log10(SFR/$M_\odot$/yr)")
    ax[0,1].set_ylabel("qir")
  #  ax[0,1].set_xlim(1,3)
  #  ax[0,1].set_ylim(0,3.5)
    ax[0,1].set_title("z ~ 2")
    
    
    qir1 = df.loc[((df['fir_flux']>3e37)& (df['mstars_tot']>2e8)),'qir_bress']
    qir2 = df.loc[((df['fir_flux']>3e37)& (df['mstars_tot']>2e8)),'qir_bress']
    
    
    m_lf_1 = np.log10(df_lf.loc[(df_lf['z'] < 1.5),'M_star'])
    m_lf_2 = np.log10(df_lf.loc[(df_lf['z'] > 1.5),'M_star'])
    
    sfr_lf_1 = np.log10(df_lf.loc[(df_lf['z'] < 1.5),'SFR10'])
    sfr_lf_2 = np.log10(df_lf.loc[(df_lf['z'] > 1.5),'SFR10'])
    
    qir_lf_1 = df_lf.loc[(df_lf['z'] < 1.5),'qTIR']
    qir_lf_2 = df_lf.loc[(df_lf['z'] > 1.5),'qTIR']
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(m_1,qir1,False,False)

    c = ax[1,0].hexbin(m_1,qir1,mincnt = 1, cmap='rainbow', gridsize = 30,vmin = 0, vmax = 1000)
    ax[1,0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,0].scatter(m_lf_1,qir_lf_1,marker="o",label = 'GAMA Data',color='white', edgecolor='black')

  #  ax[1,0].set_xlim(8,12)
  #  ax[1,0].set_ylim(0,3.5)
    ax[1,0].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[1,0].set_ylabel("qir")
    ax[1,0].set_title("z ~ 1")
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(m_2,qir2,False,False)
    
    ax[1,1].hexbin(m_2,qir2,mincnt = 1, cmap='Greys', gridsize = 30,vmin = 0, vmax = 1000)
    ax[1,1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,1].scatter(m_lf_2,qir_lf_2,marker="o",label = 'GAMA Data',color='white', edgecolor='black')
    #ax[1,1].set_xlim(8,12)
    #ax[1,1].set_ylim(0,3.5)
    ax[1,1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[1,1].set_ylabel("qir")
    ax[1,1].set_title("z ~ 2")
    
    fig.text(0.4, 0.9, "qir vs. SFR", weight='bold')
    fig.text(0.4, 0.475, "qir vs. Stellar Mass", weight='bold')
    
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Number Count')
   # fig.text(0.5, 0.04, "Log10(Stellar Mass/$M_\odot$) ", ha='center')
   # fig.text(0.04, 0.5, "qir", va='center', rotation='vertical')
   # fig.suptitle('qir vs. Stellar Mass - Lo Faro et al.')
    plt.show()
    
  
    
    
    
def qir_plots(df):
    
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'qir_bress']
    
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(mstars,qir,False,False)
    
    
    c = plt.hexbin(mstars,qir,mincnt = 100, cmap='rainbow', gridsize = 30)
    plt.plot(mid_rad_lst,med_fir_lst,'black')
    plt.plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    plt.plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    plt.xlabel("Log10(Stellar Mass/$M_\odot$)")
    plt.ylabel('qir')
    plt.title('qir vs. Stellar Mass')
    cb = plt.colorbar(c)
    cb.set_label('Number Count')
    
    plt.show()
    
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'qir_bress']    
    teff = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'Teff']
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(mstars,qir,False,False)
    
    
    c = plt.hexbin(mstars,qir,C=teff,mincnt = 100, cmap='rainbow', gridsize = 30)
    plt.plot(mid_rad_lst,med_fir_lst,'black')
    plt.plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    plt.plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    plt.xlabel("Log10(Stellar Mass/$M_\odot$)")
    plt.ylabel('qir')
    plt.title('qir vs. Stellar Mass')
    cb = plt.colorbar(c)
    cb.set_label('Number Count')
    
    plt.show()
    
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'qir_bress']    
    fir = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'fir_flux'])
    rad = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'bress_rad_lum'])
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(mstars,qir,False,False)
    
    fig, ax = plt.subplots(1,2)
    
    c = ax[0].hexbin(mstars,qir,C=fir,mincnt = 100, cmap='rainbow', gridsize = 30)
    ax[0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[0].set_ylabel('qir')
    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin = min(fir),vmax = max(fir))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb1 = fig.colorbar(sm, ax=[ax[0]])
    cb1.set_label('Log10(FIR/W)')
    
    c = ax[1].hexbin(mstars,qir,C=rad,mincnt = 100, cmap='rainbow', gridsize = 30)
    ax[1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[1].set_ylabel('qir')
    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin = min(rad),vmax = max(rad))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb1 = fig.colorbar(sm, ax=[ax[1]])
    cb1.set_label('Log10(Lrad/W/Hz)')
    
    plt.show()
    
    
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'qir_bress']    
    fir = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'fir_flux'])
    rad = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)),'bress_rad_lum'])
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(mstars,qir,False,False)
    
    fig, ax = plt.subplots(1,2)
    
    fir = fir - mstars
    rad = rad - mstars
    
    
    c = ax[0].hexbin(mstars,qir,C=fir,mincnt = 100, cmap='rainbow', gridsize = 30)
    ax[0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[0].set_ylabel('qir')
    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin = min(fir),vmax = max(fir))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb1 = fig.colorbar(sm, ax=[ax[0]])
    cb1.set_label('Log10(FIR/W) - Log10(Stellar Mass/$M_\odot$)')
    
    c = ax[1].hexbin(mstars,qir,C=rad,mincnt = 100, cmap='rainbow', gridsize = 30)
    ax[1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[1].set_ylabel('qir')
    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin = min(rad),vmax = max(rad))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb1 = fig.colorbar(sm, ax=[ax[1]])
    cb1.set_label('Log10(Lrad/W/Hz) -  Log10(Stellar Mass/$M_\odot$)')
    
    plt.show()
    
    
    fig, ax = plt.subplots(1,2)
    
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e10)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)&(df['type'] == 0)),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e10)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)&(df['type'] == 0)),'qir_bress']
    metallicity = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e10)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)&(df['type'] == 0)),'gas_metal'])
    teff = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e10)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)&(df['type'] == 0)),'Teff']
    
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(metallicity,qir,False,False)
    ax[0].hexbin(metallicity,qir,C=teff,mincnt = 1, cmap='rainbow', gridsize = 30)
    ax[0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0].set_xlabel("Log10(Gas Metallicity)")
    ax[0].set_ylabel('qir')
    ax[0].set_title('Centrals')
    
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)&(df['type'] == 1)),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)&(df['type'] == 1)),'qir_bress']
    metallicity = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)&(df['type'] == 1)),'gas_metal'])
    teff = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 3)&(df['qir_bress'] > 0)&(df['type'] == 1)),'Teff']
    
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(metallicity,qir,False,False)
    
    c=ax[1].hexbin(metallicity,qir,C=teff,mincnt = 1, cmap='rainbow', gridsize = 30)
    ax[1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1].set_xlabel("Log10(Gas Metallicity)")
    ax[1].set_ylabel('qir')
    ax[1].set_title('Sattellites')
    cb = fig.colorbar(c, ax=ax.ravel().tolist())
    cb.set_label('Effective Dust Temperature (K)')
    
    plt.show()

    
def central_counter(lst):
    lst_len = len(lst)
    counter = 0
    for i in lst:
        if i == 0:
            counter += 1
    
    return counter/lst_len
    
    
def met_dist_cent(df):
    
    fig, ax = plt.subplots(1,3)
    
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'qir_bress']
    metallicity = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'gas_metal'])
    teff = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'Teff']
    typ = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'type']
    dist_ms = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'sf_test']
    
    ssfr = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'sfr']) - mstars
    
    
    n = 1
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(mstars,qir,False,False)
    c = ax[0].hexbin(mstars,qir,C = metallicity,mincnt = n, cmap='rainbow', gridsize = 30,reduce_C_function=np.median)
    ax[0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[0].set_ylabel('qir')

    cb1 = fig.colorbar(c,ax=[ax[0]])
    cb1.set_label('$Log_{10}$(Gas Metallicity/$Z_{gas}/Z_\odot$)')
    
    c = ax[1].hexbin(mstars,qir,C = ssfr,mincnt = n, cmap='rainbow', gridsize = 30,reduce_C_function=np.median,vmin = -12,vmax = -10)
    ax[1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[1].set_ylabel('qir')

    cb1 = fig.colorbar(c,ax=[ax[1]])
    cb1.set_label('SSFR')
    
    c = ax[2].hexbin(mstars,qir,C = typ,mincnt = n, cmap='rainbow', gridsize = 30,reduce_C_function=central_counter)
    ax[2].plot(mid_rad_lst,med_fir_lst,'black')
    ax[2].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[2].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[2].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[2].set_ylabel('qir')

    cb1 = fig.colorbar(c,ax=[ax[2]])
    cb1.set_label('Central Fraction')
    fig.suptitle("qir vs. stellar mass for all galaxies")
    plt.show()
    
    fig, ax = plt.subplots(1,3)
    
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfr/q'] =='sf')),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfr/q'] =='sf')),'qir_bress']
    metallicity = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfr/q'] =='sf')),'gas_metal'])
    teff = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfr/q'] =='sf')),'Teff']
    typ = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfr/q'] =='sf')),'type']
    dist_ms = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfr/q'] =='sf')),'sf_test']
    
    
    
    
    n = 1
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(mstars,qir,False,False)
    c = ax[0].hexbin(mstars,qir,C = metallicity,mincnt = n, cmap='rainbow', gridsize = 30,reduce_C_function=np.median)
    ax[0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[0].set_ylabel('qir')

    cb1 = fig.colorbar(c,ax=[ax[0]])
    cb1.set_label('$Log_{10}$(Gas Metallicity/$Z_{gas}/Z_\odot$)')
    
    c = ax[1].hexbin(mstars,qir,C = dist_ms,mincnt = n, cmap='rainbow', gridsize = 30,reduce_C_function=np.median)
    ax[1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[1].set_ylabel('qir')

    cb1 = fig.colorbar(c,ax=[ax[1]])
    cb1.set_label('Distance to Main Sequence/$log_{10}$ (M/yr)')
    
    c = ax[2].hexbin(mstars,qir,C = typ,mincnt = n, cmap='rainbow', gridsize = 30,reduce_C_function=central_counter)
    ax[2].plot(mid_rad_lst,med_fir_lst,'black')
    ax[2].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[2].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[2].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[2].set_ylabel('qir')

    cb1 = fig.colorbar(c,ax=[ax[2]])
    cb1.set_label('Central Fraction')
    fig.suptitle("qir vs. stellar mass for star-forming galaxies")
    plt.show()

    
def ionising_phot_sfr(df):
    
    q_h = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'ionising_photons'])
    sfr = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'sfr'])
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(q_h,sfr,False,False)
    n = 1
    plt.hexbin(q_h,sfr,mincnt = n, cmap='rainbow', gridsize = 30,reduce_C_function=np.median)
    plt.plot(mid_rad_lst,med_fir_lst,'black')
    plt.plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    plt.plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    plt.xlabel("Log10(Q(H))")
    plt.ylabel("Log10(SFR)")
    plt.title("SFR vs. Q(H)")
    plt.show()
    
    
def SFR_z(df):
    
    sfr_med = []
    sfr_upper = []
    sfr_lower = []
    zlst = []
    
    for z in zlist:
    
        sfr = np.log10(df.loc[(df['z'] == z),'sfr'])
        
        sfr_med.append(np.median(sfr))
        sfr_upper.append(np.percentile(sfr,84))
        sfr_lower.append(np.percentile(sfr,16))

    plt.plot(zlist,sfr_med,'black')
    plt.plot(zlist,sfr_upper,'black',linestyle='dashed')
    plt.plot(zlist,sfr_lower,'black',linestyle='dashed')
    plt.title("SFR vs. z")
    plt.show()
    
    qir = df.loc[((df['mstars_tot'] > 1e7)&(df['mstars_tot'] < 1e10)),'qir']

    
def sfr_function(df,h0,volh):
    vol = volh/h0**3
    mlow = 8
    mupp = 12
    dm = 0.25
    mbins = np.arange(mlow, mupp, dm)
    xmf = mbins + dm/2.0 #setting up the bins
    
    slow = -3
    supp = 3
    
    for z in zlist:
        if z < 1:

            sbins = np.arange(slow, supp, dm)
            xmf = sbins + dm/2.0 #setting up the bins
            sfr = np.log10(df.loc[((df['z'] == z)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)),'sfr'])
            shist,_ = np.histogram(sfr, bins=np.append(sbins,supp)) #creating a histogram between the logged dataset and the bins
            shist = shist/vol/dm

            
            line, = plt.plot(xmf,shist)
            line.set_label('label = str(round(z,3)')


    sf_df = pd.read_csv('Documents/Masters/Masters_Thesis/SFR_function2.csv')
    source_lst_z1 = ['Sobral et al. (2013, H α)','Patel et al. (2013, IR)','Gruppioni et al. (2013, IR)','Mauch & Sadler (2007, Radio)','Patel et al. (2013, IR)','Marchetti et al. (2016, IR)','Robotham et al. (2011, UV)']
    fmt_lst = ['s','o','d','P','*','h','X']
    q = 0

    for source in source_lst_z1:
        
        z = np.mean(sf_df.loc[(sf_df['source'] == source),'Z'])
        sfr = np.log10(sf_df.loc[(sf_df['source'] == source),'SFR'])
        sfrf = sf_df.loc[(sf_df['source'] == source),'SFRF']*10**(-2)
        sfrf_err_upp = sf_df.loc[(sf_df['source'] == source),'SFR_err_upp']*10**(-2)
        sfrf_err_low = sf_df.loc[(sf_df['source'] == source),'SFR_err_low']*10**(-2)

        fmt = fmt_lst[q]
        
        label = 'z = ' + str(round(z,3)) + ' ' + source
        
        points = plt.errorbar(sfr,sfrf,yerr = [sfrf_err_upp,sfrf_err_low],fmt=fmt,markerfacecolor='white', markeredgecolor='black',ecolor = 'black')
        points.set_label(label)
        q +=1
    plt.title("SFR Function for z < 1")
    plt.xlabel("$Log_{10}$(SFR) ($M_\odot yr^{-1}$)")
    plt.ylabel('Φ ($Mpc^{-3}$)')
    plt.yscale('log')
    plt.legend()
    plt.show()

    
    for z in zlist:
        if z > 1:

            sbins = np.arange(slow, supp, dm)
            xmf = sbins + dm/2.0 #setting up the bins
            sfr = np.log10(df.loc[((df['z'] == z)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)),'sfr'])
            shist,_ = np.histogram(sfr, bins=np.append(sbins,supp)) #creating a histogram between the logged dataset and the bins
            shist = shist/vol/dm

            
            plt.plot(xmf,shist,label = str(round(z,3)))


    sf_df = pd.read_csv('Documents/Masters/Masters_Thesis/SFR_function2.csv')
    source_lst_z1 = ['Sobral et al. (2013, H α)','Patel et al. (2013, IR)','Gruppioni et al. (2013, IR)','Mauch & Sadler (2007, Radio)','Patel et al. (2013, IR)','Marchetti et al. (2016, IR)','Robotham et al. (2011, UV)']
    zlist_2 = [1.3,1.9,2.6,5,6,7,8]
    fmt_lst = ['s','o','d','P','*','h','X']
    q = 0

    for z in zlist_2:
        
        z = np.mean(sf_df.loc[(sf_df['Z'] == z),'Z'])
        sfr = np.log10(sf_df.loc[(sf_df['Z'] == z),'SFR'])
        sfrf = sf_df.loc[(sf_df['Z'] == z),'SFRF']*10**(-2)
        sfrf_err_upp = sf_df.loc[(sf_df['Z'] == z),'SFR_err_upp']*10**(-2)
        sfrf_err_low = sf_df.loc[(sf_df['Z'] == z),'SFR_err_low']*10**(-2)

        fmt = fmt_lst[q]
        if z < 3:
            label = 'z = ' + str(z) + ' - ' + 'Alavi et al. (2016, UV)'
        else:
            label = 'z = ' + str(z) + ' - ' + 'Bouwens et al. (2015, UV)'

        plt.errorbar(sfr,sfrf,yerr = [sfrf_err_upp,sfrf_err_low],fmt=fmt,label = label,markerfacecolor='white', markeredgecolor='black',ecolor = 'black')

        q +=1
    plt.title("SFR Function for z < 1")
    plt.xlabel("$Log_{10}$(SFR) ($M_\odot yr^{-1}$)")
    plt.ylabel('Φ ($Mpc^{-3}$)')
    plt.yscale('log')
    plt.legend()
    plt.show()
    
def all_galaxies_gas_disks(df):
    
    n = 1
    
    fig, ax = plt.subplots(2,4)
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'qir_bress']
    mmol_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'mmol_disk'])
    matom_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'matom_disk'])
    teff = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'Teff']
    r_gas_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'r_gas_disk'])
    mgas_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'mgas_disk'])
    sfr = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'sfr'])
    mgas_metals_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'mgas_metals_disk'])
    gas_surf_dens = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)),'gas_surf_dens'])
   
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(mstars,qir,False,False)
    c = ax[0,0].hexbin(mstars,qir,C=mmol_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[0,0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
   # ax[0,0].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[0,0].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[0,0]])
    cb1.set_label('Log10(Molecular Gas Mass in Disk/$M_\odot$)')
    
    c = ax[0,1].hexbin(mstars,qir,C=matom_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[0,1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0,1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    #ax[0,1].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[0,1]])
    cb1.set_label('Log10(Atomic Gas Mass in Disk/$M_\odot$)')
    
    c = ax[0,2].hexbin(mstars,qir,C=mgas_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[0,2].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,2].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,2].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
  #  ax[0,2].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
  #  ax[0,2].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[0,2]])
    cb1.set_label('Log10(Total Gas Mass in Disk/$M_\odot$)')
    
    c = ax[0,3].hexbin(mstars,qir,C=mgas_metals_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[0,3].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,3].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,3].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
  #  ax[0,3].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
  #  ax[0,3].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[0,3]])
    cb1.set_label('Log10(Total Metals in Disk Gas/$M_\odot$)')
    
    c = ax[1,0].hexbin(mstars,qir,C=teff,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[1,0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,0].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[1,0].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[1,0]])
    cb1.set_label('Effective Dust Temperature/K')
    
    c = ax[1,1].hexbin(mstars,qir,C=r_gas_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[1,1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
   # ax[1,1].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[1,1]])
    cb1.set_label('Log10(Half-mass radius of the gas disk/Mpc)')
    
    c = ax[1,2].hexbin(mstars,qir,C=sfr,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[1,2].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,2].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,2].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,2].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
   # ax[1,2].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[1,2]])
    cb1.set_label("Log10(SFR/$M_\odot$/yr)")
    
    c = ax[1,3].hexbin(mstars,qir,C=gas_surf_dens,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[1,3].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,3].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,3].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,3].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    #ax[1,3].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[1,3]])
    cb1.set_label("Log10(Gas Surface Density/$M_\odot$/$Mpc^2$)")
    plt.suptitle("qir vs. total stellar mass - all galaxies")
    fig.set_size_inches(18.5, 10.5)
    fig.savefig("all_qir_m_disk_8_9.png")
    plt.show()
        

        
def sf_galaxies_gas_disks(df):
    
    n = 1
    
    fig, ax = plt.subplots(2,4)
    mstars = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfr/q'] == 'sf')),'mstars_tot'])
    qir = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfr/q'] == 'sf')),'qir_bress']
    mmol_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfr/q'] == 'sf')),'mmol_disk'])
    matom_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfr/q'] == 'sf')),'matom_disk'])
    teff = df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e12)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfr/q'] == 'sf')),'Teff']
    r_gas_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfr/q'] == 'sf')),'r_gas_disk'])
    mgas_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfr/q'] == 'sf')),'mgas_disk'])
    sfr = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfr/q'] == 'sf')),'sfr'])
    mgas_metals_disk = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfr/q'] == 'sf')),'mgas_metals_disk'])
    gas_surf_dens = np.log10(df.loc[((df['z'] == 0)&(df['mstars_tot'] > 1e8)&(df['mstars_tot'] < 1e9)&(df['qir_bress'] < 5)&(df['qir_bress'] > -1)&(df['sfr/q'] == 'sf')),'gas_surf_dens'])
   
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(mstars,qir,False,False)
    c = ax[0,0].hexbin(mstars,qir,C=mmol_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[0,0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
   # ax[0,0].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[0,0].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[0,0]])
    cb1.set_label('Log10(Molecular Gas Mass in Disk/$M_\odot$)')
    
    c = ax[0,1].hexbin(mstars,qir,C=matom_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[0,1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[0,1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    #ax[0,1].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[0,1]])
    cb1.set_label('Log10(Atomic Gas Mass in Disk/$M_\odot$)')
    
    c = ax[0,2].hexbin(mstars,qir,C=mgas_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[0,2].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,2].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,2].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
  #  ax[0,2].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
  #  ax[0,2].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[0,2]])
    cb1.set_label('Log10(Total Gas Mass in Disk/$M_\odot$)')
    
    c = ax[0,3].hexbin(mstars,qir,C=mgas_metals_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[0,3].plot(mid_rad_lst,med_fir_lst,'black')
    ax[0,3].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[0,3].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
  #  ax[0,3].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
  #  ax[0,3].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[0,3]])
    cb1.set_label('Log10(Total Metals in Disk Gas/$M_\odot$)')
    
    c = ax[1,0].hexbin(mstars,qir,C=teff,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[1,0].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,0].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,0].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,0].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    ax[1,0].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[1,0]])
    cb1.set_label('Effective Dust Temperature/K')
    
    c = ax[1,1].hexbin(mstars,qir,C=r_gas_disk,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[1,1].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,1].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,1].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,1].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
   # ax[1,1].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[1,1]])
    cb1.set_label('Log10(Half-mass radius of the gas disk/Mpc)')
    
    c = ax[1,2].hexbin(mstars,qir,C=sfr,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[1,2].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,2].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,2].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,2].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
   # ax[1,2].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[1,2]])
    cb1.set_label("Log10(SFR/$M_\odot$/yr)")
    
    c = ax[1,3].hexbin(mstars,qir,C=gas_surf_dens,mincnt = n, cmap='rainbow', gridsize = 30)
    ax[1,3].plot(mid_rad_lst,med_fir_lst,'black')
    ax[1,3].plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    ax[1,3].plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    ax[1,3].set_xlabel("Log10(Stellar Mass/$M_\odot$)")
    #ax[1,3].set_ylabel('qir')
    
    cb1 = fig.colorbar(c,ax=[ax[1,3]])
    cb1.set_label("Log10(Gas Surface Density/$M_\odot$/$Mpc^2$)")
    plt.suptitle("qir vs. total stellar mass - all galaxies")
    fig.set_size_inches(18.5, 10.5)
    plt.savefig("sf_qir_m_disk_8_9.png")
    plt.show()

def rad_lum_func_plt_z0(df,h0,volh):
    
    
    
    vol = volh/h0**3
    mlow = 0
    mupp = 40
    dm = 0.25
    mbins = np.arange(mlow, mupp, dm)
    xmf = mbins + dm/2.0 #setting up the bins

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
    
    dhist = dhist/vol/dm
    
    
    bress_rad_lum = df.loc[(df['z'] < 0.1),'bress_rad_lum']

    blum = np.log10(bress_rad_lum) #logging the dataset

    bhist= np.histogram(blum, bins=np.append(mbins,mupp)) #creating a histogram between the logged dataset and the bins

    bhist = bhist[0]/vol/dm
    
    mid_rad_lst,med_fir_lst,low_fir_lst,high_fir_lst = median_line(xmf,np.log10(bhist),False,False)
    plt.plot(mid_rad_lst,med_fir_lst)
    plt.fill_between(mid_rad_lst,low_fir_lst,high_fir_lst,alpha=0.5)
    
    #plt.plot(mid_rad_lst,low_fir_lst,'black',linestyle='dashed')   
    #plt.plot(mid_rad_lst,high_fir_lst,'black',linestyle='dashed')
    

    plt.errorbar(bonatox,bonatoy,yerr = [bonatoerr_down,bonatoerr_up],fmt="o",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Bonato et al. 2020')
    plt.errorbar(butlerx,butlery,yerr = [butlererr_down, butlererr_up],fmt="s",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Butler et al. 2019')
    plt.errorbar(ocranx,ocrany,yerr = [ocranerr_down, ocranerr_up],fmt="P",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Ocran et al. 2020')
    plt.errorbar(novakx,novaky,yerr = [novakerr_down,novakerr_up],fmt="^",markerfacecolor='white', markeredgecolor='black',ecolor='black',label = 'Novak et al. 2020')
    plt.xlabel("log$_{10}$(L$_{1.4GHz}$) (W/Hz)",size=13)
    plt.ylabel("log$_{10}$(Φ) (Mpc$^{-3}$ dex$^{-1}$)",size=13)
    plt.xlim(19,25)
    plt.ylim(-7,0)
    plt.savefig("rad_lum_func_z0.pdf",format='pdf')
    plt.legend()
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
    ax[0].set_xlabel("Log$_{10}$($L_{1.4Ghz}$) (W/Hz)")
    ax[0].set_ylabel("$Log_{10}$(Φ) ($Mpc^{-3} dex^{-1}$)")
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
    ax[1].set_xlabel("Log$_{10}$($L_{1.4Ghz}$) (W/Hz)")
    ax[1].set_ylabel("$Log_{10}$(Φ) ($Mpc^{-3} dex^{-1}$)")
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
    
    bress_rad_lum = df.loc[((df['z'] < 2.2)&(df['z']>1.8)),'bress_rad_lum']

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
    ax[2].set_xlabel("Log$_{10}$($L_{1.4Ghz}$) (W/Hz)")
    ax[2].set_ylabel("$Log_{10}$(Φ) ($Mpc^{-3} dex^{-1}$)")
    ax[2].set_xlim(19,25)
    ax[2].set_ylim(-7,0)

    plt.legend()
    plt.show()
    
    
    
def qir_hist(df):
    
    qir_cent = df.loc[((df['mstars_tot'] > 1e8)&(df['type'] == 0)&(df['sfr/q']=='sf')&(df['qir_bress']>1)&(df['qir_bress']<3)),'qir_bress']
    qir_sat = df.loc[((df['mstars_tot'] > 1e8)&(df['type'] == 1)&(df['sfr/q']=='sf')&(df['qir_bress']>1)&(df['qir_bress']<3)),'qir_bress']
    qir_orph = df.loc[((df['mstars_tot'] > 1e8)&(df['type'] == 2)&(df['sfr/q']=='sf')&(df['qir_bress']>1)&(df['qir_bress']<3)),'qir_bress']

    qir_cent_q = df.loc[((df['mstars_tot'] > 1e8)&(df['type'] == 0)&(df['sfr/q']=='q')&(df['qir_bress']>1)&(df['qir_bress']<3)),'qir_bress']
    qir_sat_q = df.loc[((df['mstars_tot'] > 1e8)&(df['type'] == 1)&(df['sfr/q']=='q')&(df['qir_bress']>1)&(df['qir_bress']<3)),'qir_bress']
    qir_orph_q = df.loc[((df['mstars_tot'] > 1e8)&(df['type'] == 2)&(df['sfr/q']=='q')&(df['qir_bress']>1)&(df['qir_bress']<3)),'qir_bress']
    
    
    
    
    bins = 20
    
    y, binEdges = np.histogram(qir_cent, bins=bins,density = True)
    #plt.hist(qir_cent, bins=100, edgecolor='black')
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    plt.plot(bincenters, y, '-', c='black',label = 'Centrals')
    
    y, binEdges = np.histogram(qir_sat, bins=bins,density = True)
    #plt.hist(qir_cent, bins=100, edgecolor='black')
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    plt.plot(bincenters, y, '-', c='Blue',label = 'Satellites')    
 

    y, binEdges = np.histogram(qir_orph, bins=bins,density = True)
    #plt.hist(qir_cent, bins=100, edgecolor='black')
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    plt.plot(bincenters, y, '-', c='Red',label = 'Orphans')
    
    y, binEdges = np.histogram(qir_cent_q, bins=bins,density = True)
    #plt.hist(qir_cent, bins=100, edgecolor='black')
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    plt.plot(bincenters, y, ':', c='black',label = 'Centrals - Quenched')
    
    y, binEdges = np.histogram(qir_sat_q, bins=bins,density = True)
    #plt.hist(qir_cent, bins=100, edgecolor='black')
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    plt.plot(bincenters, y, ':', c='Blue',label = 'Satellites - Quenched')    
 

    y, binEdges = np.histogram(qir_orph_q, bins=bins,density = True)
    #plt.hist(qir_cent, bins=100, edgecolor='black')
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    plt.plot(bincenters, y, ':', c='Red',label = 'Orphans')    
    
    plt.legend()
    plt.show()
    
    
def xyz_plts(df):
    
    qir_bress = df.loc[((df['pos_z']>100)&(df['pos_z']<110)),'qir_bress']
    pos_x = df.loc[((df['pos_z']>100)&(df['pos_z']<110)),'pos_x']
    pos_y = df.loc[((df['pos_z']>100)&(df['pos_z']<110)),'pos_y']
   # pos_z = df.loc[df['mstars_tot']>10,'pos_z']
    
    qir_cent = df.loc[(df['type'] == 0),'qir_bress']
    qir_sat = df.loc[(df['type'] == 1),'qir_bress']
    qir_orph = df.loc[(df['type'] == 2),'qir_bress']
    
    plt.hist(qir_cent)
    plt.title("Centrals")
    plt.show()
    
    plt.hist(qir_sat)
    plt.title("Satellites")
    plt.show()
    
    plt.hist(qir_orph)
    plt.title("Orphans")
    plt.show()
    
    fig, ax = plt.subplots(2,1)
    c = ax[0].hexbin(pos_x,pos_y,C=qir_bress,mincnt = 4,cmap='rainbow', gridsize = 30)
    cb1 = fig.colorbar(c, ax=[ax[0]])
    cb1.set_label('qir')
    
    c = ax[1].hexbin(pos_x,pos_y,mincnt = 4,cmap='rainbow', gridsize = 30)
    cb1 = fig.colorbar(c, ax=[ax[1]])
    cb1.set_label('number count')
    
    
    m = 10000
    n = m + 10000
    plt.show()
    
    xyz_mat = df[['pos_x','pos_y','pos_z']]
    print(xyz_mat)
    dist_mat = distance_matrix(xyz_mat[:n],xyz_mat[:n])
    print(dist_mat)
    
    lst = []
    
    for i in dist_mat:
        i.sort()
        neigh_5 = i[5]
        if neigh_5 < 20:
            lst.append(neigh_5)
        else:
            lst.append(np.nan)
        
    qir = df['qir_bress']
    lst = np.log10(lst)
    dist_ms = df['sf_test']
    plt.hexbin(lst,qir[:n],C= dist_ms[:n],mincnt = 10,cmap='rainbow', gridsize = 30)
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

        def synchrotron_lum(SFR, nu):


            LNT = 6.1*10**(-2)*(1.49/0.408)**(-0.8) ##2.16 x10^(-2) W/Hz/10^(21)
            ENT = LNT/0.0163 ##1.31 x 10^(23) W/Hz/10^(21)
            ESNR = ENT * 0.06 ##0.0784 x 10^(23)W/Hz/10^(21)
            EEI = 0.94*ENT ##1.23 x 10^(23) W/Hz/10^(21)
            alpha = 0.815661

            comp1 = ESNR * (nu / 1.49)**(-0.5) + (EEI * (nu / 1.49)**(-alpha))
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
            lum_radio[i,:] = freefree_lum(q_ionis[:], nu) + synchrotron_lum(sfr_tot[:], nu)

            freefree[i,:] = freefree_lum(q_ionis[:], nu)/1e7
            
            sync[i,:] = synchrotron_lum(sfr_tot[:], nu)/1e7
            

        
        
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
    
    return qir_lst, qir_bress, bress_rad_lum, bress_lir_total, freefree_lst, sync_lst, ion_phot_lst
    
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

    fields = {'galaxies': ('mstars_disk','mstars_bulge','sfr_disk','sfr_burst','type','mgas_metals_bulge','mgas_metals_disk','mgas_bulge','mgas_disk','rgas_disk','rstar_disk','id_galaxy','matom_disk','mgas_metals_disk','mmol_disk','position_x','position_y','position_z')}
    
    
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
        
        
    


  
    df,sf_lst = dataframe(hdf5_lir_lst,zlist,seds_bands_lst,seds_lir_lst,lir_total_W_lst,seds_nodust_lst,h0,Teff_lst)
    print(df)
  #  rad_size(df)
 #   density(df)
  #  qir_z_plt(df)
   # schr_vs_med(df)
    #fir_lir_mass(df)
    
    #qir_with_m_plt(df)
    
    #q_m_m_z(df)
    
    #qir_vs_qir(df)
    
    #sfr_rad_lum(df)
    #rad_lum_func_plt_z0(df,h0,volh)
  #  rad_lum_func_plt_2(df,h0,volh)

   # hist(df)
   # qir_with_m_plt(df)
    
    #lum_m(df,sf_lst)
  #  qir_metals_mass_bins(df)
  #  metal_hist(df)
    
  #  gas_metal_vs_stellar_mass(df)
    
    #derived_SFR(df)
    
   # qir_v_dust_ff_sync(df)       
   # lo_faro_plots(df)
    
    #GAMA_plots(df)
  #  GAMA_flux_limit(df)
   # met_dist_cent(df)
  #  sfr_function(df,h0,volh)
   # qir_plots(df)
   # all_galaxies_gas_disks(df)
   # SFR_z(df)
   # ionising_phot_sfr(df)

  #  all_galaxies_gas_disks(df)

#    sf_galaxies_gas_disks(df)
    xyz_plts(df)
  #  qir_hist(df)
    #rad_lum_func_plt_z0(df,h0,volh)
    mstars_disk = hdf5_data[2]
    mstars_bulge = hdf5_data[3]
    mstars_tot = []
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