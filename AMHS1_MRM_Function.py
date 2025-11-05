import numpy as np
import matplotlib.pyplot as plt
from math import *

def sech(x):
   return np.divide(1,np.cosh(x))

class RFall:
   #all properties related to B1, B0, pulse sequence, and other RF things
   def __init__(self):
       #B1
       self.Hgyro = 42.58e6 #gyromagnetic ratio [Hz/T] 
       self.B1x = []
       self.B1y = []
       self.B1x_rad = 0. 
       self.B1y_rad = 0. 

       #time variables
       self.TR = 15e-3 #TR [s]
       self.periods = 1
       self.dur = self.periods*self.TR #RF pulse duration [s]
       self.dt = 1.0e-6 #time step in [s]
       self.t = np.linspace(0,self.TR,int(self.TR/self.dt)) #single TR [s]
       self.t_tot = np.linspace(0,self.dur,int(self.dur/self.dt)) #total duration [s]
       self.Tp = 5.99e-3 #Pulse time [s]

   def set_B1mag(self,B1):
       self.B1_mag = B1

   def AMHS1(self,cycle,correction,refoc,coils, plot_pulse,w1x,center,R):
       Tp = self.Tp #pulse time in [s]
       B1yfreq = np.tile(self.B1_mag*self.Hgyro,(len(self.t),1)) #B1y frequencies in gradient in [Hz], tiled
       
       #Variable initialization
       N = int(np.round(Tp/self.dt)) #total steps in pulse
       ntimes = N/2 
       theta = np.zeros(N) 
       amp = np.zeros(N)
       phs = np.zeros(N)
       dw = np.zeros(N)
       phsout = np.zeros(N)
       TR_idx = int(np.round(self.TR/self.dt)) + 1 #indices for whole TR
       nrange = np.arange(1,N+1) #indices for just Tp
       
       #Calculation of HS1 pulse envelope F_1(t) shape function
       factor = 0.5 * np.pi * R
       cutoff = 0.01 #truncation factor
       temp = 2.0/cutoff
       beta = -np.log((temp+np.sqrt(np.power(temp,2)-4.0))/2.0)
       base = -1.0 + (0.5/ntimes)
       theta = beta * (base + ((nrange-1.0)/ntimes))
       
       amp = 1.0/(np.cosh(theta)) #F_1(t)
       phs = (factor/beta) * np.log(1.0/(np.cosh(beta))/amp) #phi_HS1(t)

       #AMHS1 correction - F_cor(t)
       ampog = amp #save original amp for comparison
       A = R/(2*(Tp))
       t_scaled = (2*self.t[:N]/Tp)-1 
       FM = -A*np.tanh(beta*t_scaled) + center #FM(t)
       phs_norm = (np.divide(phs-np.min(phs),(np.max(phs)-np.min(phs)))) #normalized phase
       FM_norm = (np.divide(FM-np.min(FM),(np.max(FM)-np.min(FM)))) #normalized FM
       scale = np.multiply(FM_norm/np.max(FM_norm),1) #scaling by FM
       scale = 2*(scale-0.5) 
       FM_norm2 = (np.divide(FM,(np.max(FM)-np.min(FM))))/2 #
       scale2 = np.multiply(np.multiply(FM_norm2/np.max(FM_norm2),1),phs_norm) #scaling by phi
       
       #Whether or not to apply correction
       if correction==1:
           amp = (np.multiply(amp,(scale2)))    
           amp = amp/np.max(amp)
       
       #plot corrected vs uncorrected
       if plot_pulse==1:
           plt.title('Unscaled Vs Scaled')
           plt.ylabel('Normalized Amplitude')
           plt.xlabel('Time (s)')
           plt.plot(self.t[0:len(ampog)]*1e3,ampog,color="#FF5555",linewidth=2)
           plt.plot(self.t[0:len(ampog)]*1e3,amp,color="#017991",linewidth=2)
           plt.show()
   
       slcoff_tiled = center+0.0*B1yfreq[:N,:].T #matching dimensions for slice offset + slice offset in [Hz]
       nrange_tiled = np.tile(nrange,(len(self.B1_mag),1)) #tiling n_range for use of gradient
       phs_tiled = np.add(phs,np.multiply(nrange_tiled,(2*np.pi*Tp*slcoff_tiled/N))) #recalc phases with offset + tile

       #if 2nd step of phase cycle 
       if cycle==1:
           phs_tiled = -phs_tiled
       
       B1yTR = B1yfreq[:TR_idx,:].T #B1y to first TR
       
       #find index closest to center
       diff_array = abs(B1yTR[:,-1]-center)
       center_idx = diff_array.argmin()
       amp_tiled = np.tile(amp,(len(self.B1_mag),1)) #tile F1(t)
       
       #tiling B1x
       B1x_tiled_Tp = np.multiply(w1x,np.multiply(amp_tiled,np.sin(phs_tiled))) #B1x tiled but only over Tp
       B1x_tiled = np.zeros((len(self.B1_mag),len(self.t))) #B1x tiled but over full time range       
       B1x_tiled[:,int(N/2):int(N/2)+N] = B1x_tiled_Tp #insert B1x into center of pulse
       
       #tiling B1y
       #using this B1y_tiled primarily for phase at the center of the slice
       B1y_tiled = np.zeros((len(self.B1_mag),len(self.t))) #B1y tiled but over full time range
       B1y_tiled = np.tile(B1yTR[center_idx,:],(len(self.B1_mag),1))
       B1y_tiled[:,0:int(N/2)] = refoc*-np.flip(B1y_tiled[:,0:int(N/2)]) #flip lobes for refocusing
       B1y_tiled[:,int(3*N/2):int(4*N/2)] = refoc*-np.flip(B1y_tiled[:,int(3*N/2):int(4*N/2)])
       B1y_tiled[:,int(4*N/2)::] = 0 #zero anything after the pulse
       
       #this is the actual B1y amplitude 
       B1yTR2 = B1yTR #save og 
       B1yTR2[:,0:int(N/2)] = refoc*-B1yTR[:,0:int(N/2)]
       B1yTR2[:,int(3*N/2):int(4*N/2)] = refoc*-B1yTR[:,int(3*N/2):int(4*N/2)]
       B1yTR2[:,int(4*N/2)::] = 0
       
       #flipping of lobes for phase cycling
       if cycle==1:
        B1yTR2 = -B1yTR2
        
       #B1x and B1y for two coil case
       B1x = B1x_tiled
       B1y = B1yTR2
       
       #phs and amp for one coil case
       phsout = np.arctan2(B1y_tiled,B1x_tiled)#
       amp = np.abs(B1yTR2)+0*np.sqrt(np.power(B1x_tiled,2) + np.power(B1y_tiled,2))  
   
       #B1x and B1y for one coil case 
       rl = np.multiply(amp,np.cos(phsout))
       im = np.multiply(amp,np.sin(phsout))
    
       if plot_pulse==1:
        mid = -1
        plt.figure(constrained_layout=True)
        plt.subplot(2,2,1)
        plt.plot(B1x[mid,:],label='B1x')
        plt.plot(B1y[mid,:],label='B1y')
        plt.legend()
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.title("B1x+B1y")
        
        plt.subplot(2,2,2)
        plt.plot(phsout[mid,:],label='B1x')
        plt.xlabel("Index")
        plt.ylabel("Phase (radians)")
        plt.title("PHS")

        plt.subplot(2,2,3)
        plt.plot(amp[mid,:],label='B1x')
        plt.xlabel("Index")
        plt.ylabel("Amplitude (Hz)")  
        plt.title("AMP")

        plt.subplot(2,2,4)
        plt.plot(rl[mid,:],label='real')
        plt.plot(im[mid,:],label='imag')
        plt.legend()
        plt.xlabel("Index")
        plt.ylabel("Amplitude (Hz)")
        plt.title("Full Pulse")
        plt.show()
        
       #clean up variables and convert  
       if coils == 2: #two coils    
        self.B1x = B1x
        self.B1y = B1y
        
       if coils == 1: #one coil     
        self.B1x = rl 
        self.B1y = im
       
       #repeat over number of periods specified       
       self.B1x = np.tile(self.B1x,(1,self.periods))
       self.B1y = np.tile(self.B1y,(1,self.periods))
       
       #convert to rad/s
       self.B1x_rad = 2*np.pi*self.B1x 
       self.B1y_rad = 2*np.pi*self.B1y 

       #plot final output - self.B1x, self.B1y, these can be used in bloch simulations or can take the last index (-1) for use on a spectrometer
       if plot_pulse==1:
        plt.figure(constrained_layout=True)
        plt.plot(self.t_tot,(self.B1x[mid,:]),label='real',linewidth=2)
        plt.plot(self.t_tot,(self.B1y[mid,:]),label='imag',linewidth=2)
        plt.title("B1 pulse rotating frame")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Hz)")
        plt.tight_layout()
        plt.show()
 
 
#Initialization-------------------------------------------------------------------------------------------------
RF = RFall()
#B1y gradient in [T]
B1y_first = 0e3/42.58e6 #[T]
B1y_last = 5e3/42.58e6 #[T]
B1y_steps = 1000 #number of steps in gradient
B1y_all = np.linspace(B1y_first,B1y_last,num=B1y_steps,endpoint=True) #Maximum B1 values 

#Run Pulse------------------------------------------------------------------------------------------------------
#initialize pulse sequence and dependent constants
refoc = 1 #refocus? y=1, n=0
phs_cycle = 0 #phase cycle step in 2 step phase cycle
scale = 0 #scaling? 
plot_pulse = 1 #plot outputs? 0=no, 1=yes
coils = 2 #2 = two coils, 1 = one coil
R = 10.4 #TBW product, R value
w1x = 500 #w1xmax in [Hz]
center = 1000 #w1y(r_center) in [Hz]             
             
RF.set_B1mag(B1y_all)
RF.AMHS1(phs_cycle,scale,refoc,coils,plot_pulse,w1x,center,R) 