# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:51:20 2023


@author: andre
"""

#%% Pacotes utilizados nas simulações
import matplotlib.pyplot as pl # para os gráficos
import numpy as np
import scipy.signal as signal # chirp

np.random.seed(2022)

#%% Propriedades do gráfico
pl.rcParams['font.family'] = ['serif']
pl.rcParams['font.serif'] = ['Times New Roman']
pl.rcParams['figure.autolayout'] = True
pl.rcParams['text.usetex'] = True

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

pl.rc('font', size=SMALL_SIZE)          # controls default text sizes
pl.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
pl.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
pl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
pl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
pl.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
pl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#%% Parâmetros físicos do sistema
k1 = 5.49e+3 # rigidez [N/m]
k2 = 3.24e+4
k3 = 4.68e+7
ms = 0.26 # massa [kg] 
c= 1.36 # coeficiente de amortecimento [Ns/m]
wn = np.sqrt(k1/ms) # freq natural linearizada [rad/s]
zeta = c/(2*wn*ms)
wd = wn*np.sqrt(1 - zeta**2) # freq natural amortecida [rad/s]
wnhz=wn/(2*np.pi)

#%% Condições iniciais
x0 = 0 #deslocamento inicial
v0 = 0 #velocidade inicial

#%% Parâmetros amostrais (estatístico)
Np = 1024 # número de pontos
Fs = 512 # pontos por segundo
t  = np.linspace(0, Np/Fs, Np, endpoint=False)
# Intervalo de frequência analisado (em torno de wn)
wmin = 15 # [Hz]
wmax = 35 # [hz]

#%% Integrador rungekutta para resolver EDOs
def Linear_oscillator(m,c,k1,k2,k3,F,t,disp0,vel0,alpha): #NONLinear_oscilator
    # Parâmetros de entrada da função:
    # m: massa do sistema
    # c: coeficiente de amortecimento do sistema
    # k: rigidez do sistema (Adicionar k2 e k3)
    # t: vetor tempo
    # disp0: deslocamento inicial
    # vel0: velocidade inicial
    # F: vetor força aplicada
    disp = np.zeros((len(t))) # define um vetor vazio para o deslocamento a cada instante de tempo, esses zeros serão preenchidos com os valores obtidos pelo integrador
    vel = np.zeros((len(t))) # idem, agora para a velocidade
    acc = np.zeros((len(t))) #idem, agora para a aceleração
    disp[0,] = disp0 # define que o primeiro valor no vetor de deslocamento é o deslocamento inicial que definimos
    vel[0,] = vel0 #idem para a velocidade
    k=k1
    # Para a rigidez não linear: adicionar componentes de k2 e k3 na próxima linha
    acc[0,] = (1/m)*(F[0,] -c*vel[0,] -k*disp[0,] -k2*(disp[0,]**2) -k3*(disp[0,]**3))
    dt = t[2]-t[1] 
    for i in range(1,len(t)):
        if disp[i-1]>0:
            k=alpha*k1
        else:
            k=k1
        
        k11 = dt*vel[i-1,]
        k12 = (1/m)*dt*(F[i-1,] - c*vel[i-1,] - k*disp[i-1,] -k2*(disp[i-1,]**2) -k3*(disp[i-1,]**3))
        
        k21 = dt*(vel[i-1,]+k12/2) 
        k22 = (1/m)*dt*((F[i-1,]+F[i,])/2 - c*(vel[i-1,]+k12/2) - k*(disp[i-1,]+k11/2)-k2*(disp[i-1,]+k11/2)**2 -k3*(disp[i-1,]+k11/2)**3)  
        
        k31 = dt*(vel[i-1,]+k22/2) 
        k32 = (1/m)*dt*((F[i-1,]+F[i,])/2 - c*(vel[i-1,]+k22/2) - k*(disp[i-1,]+k21/2) -k2*(disp[i-1,]+k21/2)**2 -k3*(disp[i-1,]+k21/2)**3)  
        
        k41 = dt*(vel[i-1,]+k32) 
        k42 = (1/m)*dt*(F[i,] - c*(vel[i-1,]+k32) - k*(disp[i-1,]+k31) -k2*(disp[i-1,]+k31)**2 -k3*(disp[i-1,]+k31)**3)
        
        disp[i,] = disp[i-1,]+(1/6)*(k11+2*k21+2*k31+k41) 
        vel[i,] = vel[i-1,]+(1/6)*(k12+2*k22+2*k32+k42) 
        
        acc[i,] = (1/m)*(F[i,] - c*vel[i,] - k*disp[i,] -k2*(disp[i,]**2) -k3*(disp[i,]**3)) 
    return(acc,vel,disp) # Parâmetros de saída

#%%
def normalizar (u,y):
    # normalização:
    # entrada = u
    # saída   = y
    std_sig_input = np.std(u)
    std_sig_output = np.std(y)

    mean_sig_input = np.mean(u)
    mean_sig_output = np.mean(y)
    return (std_sig_input,std_sig_output,mean_sig_input,mean_sig_output)
#%%
def ad_ruido (signal_output,t):
    std_sig_output = np.std(signal_output) #STandard Deviation = desvio padrão
    ruido = np.random.normal(0, 0.05*std_sig_output, size=len(t))
    signal_output = signal_output + ruido
    return(signal_output)
    
#%% Exemplo de resposta do Oscilador Linear
alpha=1
k2 = 0#3.24e+4
k3 = 0#4.68e+7

F_chirp_min = 1*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
acc_chirp_min,vel_chirp_min,disp_chirp_min = Linear_oscillator(ms,c,k1,k2,k3,F_chirp_min,t,x0,v0,alpha)

# Entrada de varredura senoidal
pl.figure()
pl.grid(b=None, which='major', axis='both')
pl.plot(t,F_chirp_min,'b', label='Duffing + Trinca')
pl.xlabel('Tempo [s]')
pl.ylabel('Amplitude [N]')
pl.xlim(0,2)
#pl.legend()
#pl.title('Entrada de varredura senoidal')
pl.show()

# Saída de deslocamento do sistema
pl.figure()
pl.grid(b=None, which='major', axis='both')
pl.plot(t,1000*disp_chirp_min,'b', label='Duffing + Trinca')
pl.xlabel('Tempo [s]')
pl.ylabel('Deslocamento [mm]')
pl.xlim(0,2)
#pl.legend()
#pl.title('Entrada de varredura senoidal')
pl.show()


#%% Resposta ao impulso do sistema no domínio da frequência

F_impulse = np.zeros((len(t),)) # cria um vetor de zeros
F_impulse[1] = 1 # aloca a amplitude da força na posição inicial (na verdade na segunda amostra)
acc_imp_L,vel_imp_L,disp_imp_L = Linear_oscillator(ms,c,k1,k2,k3,F_impulse,t,x0,v0,alpha)

# Saída no domínio da frequência
f_imp, Pxx_den_imp = signal.welch(disp_imp_L, fs=Fs, window='boxcar', nperseg=Np, noverlap=0, nfft=len(t), detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')

pl.figure()
pl.grid(which='major', axis='both')
pl.semilogy(f_imp, Pxx_den_imp,'b')
pl.xlabel('Frequência [Hz]')
pl.xlim(0,100)
pl.ylabel('Densidade Espectral [V$^2$/Hz]')
# pl.title('Resposta ao impulso do sistema no domínio da frequência')
pl.show()

#%% Espectograma do Oscilador Linear

pl.figure()
pl.specgram(disp_chirp_min, Fs=Fs, cmap='jet',mode='psd')
pl.ylim(0,250)
pl.colorbar(label='[dB]')
pl.ylabel('Frequência [Hz]')
pl.xlabel('Tempo [s]')
pl.show()


#%% Espectograma do Oscilador Linear
# ALTERNATIVA: NORMALIZAR SINAIS
u1 = F_chirp_min
y1 = disp_chirp_min

# Normalização
std_sig_input,std_sig_output,mean_sig_input,mean_sig_output= normalizar(u1,y1)
u1 = (u1 - mean_sig_input)/std_sig_input
y1 = (y1 - mean_sig_output)/std_sig_output

# pl.figure()
# pl.specgram(y1, Fs=Fs, cmap='jet',mode='psd', vmin=-180, vmax=0)
# pl.ylim(0,250)
# pl.colorbar(label='$[dB]$')
# pl.ylabel('$Frequ \hat encia [Hz]$')
# pl.xlabel('$Tempo [s]$')
# pl.show()

#%% Exemplo de resposta do Oscilador de Duffing
alpha=1
k2 = 3.24e+4
k3 = 4.68e+7

F_chirp_min = 0.1*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
acc_chirp_min,vel_chirp_min,disp_chirp_min_Duff = Linear_oscillator(ms,c,k1,k2,k3,F_chirp_min,t,x0,v0,alpha)
acc_chirp_min,vel_chirp_min,disp_chirp_min_Lin = Linear_oscillator(ms,c,k1,0,0,F_chirp_min,t,x0,v0,alpha)


Dif_DuffL = disp_chirp_min_Duff - disp_chirp_min_Lin


# Entrada de varredura senoidal
pl.figure()
pl.grid(b=None, which='major', axis='both')
pl.plot(t,1000*disp_chirp_min_Duff,'b', label='Duffing')
pl.plot(t,1000*disp_chirp_min_Lin,'--r', alpha=0.8,  label='Linear')
pl.plot(t,1000*Dif_DuffL,'--k',alpha=0.8, label='Diferença')
pl.xlabel('Tempo [s]')
pl.ylabel('Deslocamento [mm]')
pl.xlim(0,2)
#pl.legend()
#pl.title('Entrada de varredura senoidal')
pl.show()

F_chirp_max = 0.8*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
acc_chirp_max,vel_chirp_max,disp_chirp_max_Duff = Linear_oscillator(ms,c,k1,k2,k3,F_chirp_max,t,x0,v0,alpha)
acc_chirp_max,vel_chirp_max,disp_chirp_max_Lin = Linear_oscillator(ms,c,k1,0,0,F_chirp_max,t,x0,v0,alpha)

Dif_DuffH = disp_chirp_max_Duff - disp_chirp_max_Lin

# Saída de deslocamento do sistema
pl.figure()
pl.grid(b=None, which='major', axis='both')
pl.plot(t,1000*disp_chirp_max_Duff,'b', label='Duffing')
pl.plot(t,1000*disp_chirp_max_Lin, '--r', label='Linear')
pl.plot(t,1000*Dif_DuffH,'k',alpha=0.8, label='Diferença')
pl.xlabel('Tempo [s]')
pl.ylabel('Deslocamento [mm]')
pl.xlim(0,2)
#pl.legend()
#pl.title('Entrada de varredura senoidal')
pl.show()

#%% Resposta do oscilador de Duffing no domínio da frequência

F_chirp = 0.02*np.sin(wn*t) # amplitude * função seno (valor do angulo, quanto maior esse valor aqui dentro, menor será a frequência)
acc_imp_NLlow,vel_imp_NLlow,disp_imp_NLlow = Linear_oscillator(ms,c,k1,0,0,F_chirp,t,x0,v0,1)
acc_imp_NLhigh,vel_imp_NLhigh,disp_imp_NLhigh = Linear_oscillator(ms,c,k1,k2,k3,F_chirp,t,x0,v0,alpha)


# Saída no domínio da frequência
f_imp_NLlow, Pxx_den_NLlow = signal.welch(disp_imp_NLlow, fs=Fs, window='hann', nperseg=Np, noverlap=0, nfft=len(t), detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')
f_imp_NLhigh, Pxx_den_NLhigh = signal.welch(disp_imp_NLhigh, fs=Fs, window='hann', nperseg=Np, noverlap=0, nfft=len(t), detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')

pl.figure()
pl.grid(which='major', axis='both')
pl.semilogy(f_imp_NLhigh, Pxx_den_NLhigh,'b', label='Duffing com Trinca')
pl.semilogy(f_imp_NLlow, Pxx_den_NLlow,'--r', label='Linear')
pl.xlabel('Frequência [Hz]')
pl.ylabel('Densidade Espectral [V$^2$/Hz]')
pl.xlim(0,250)
# pl.legend()
# pl.title('Densidade impulso no domínio da frequência')
pl.show()


F_chirp = 0.8*np.sin(wn*t) # amplitude * função seno (valor do angulo, quanto maior esse valor aqui dentro, menor será a frequência)
acc_imp_NLlow,vel_imp_NLlow,disp_imp_NLlow = Linear_oscillator(ms,c,k1,0,0,F_chirp,t,x0,v0,1)
acc_imp_NLhigh,vel_imp_NLhigh,disp_imp_NLhigh = Linear_oscillator(ms,c,k1,k2,k3,F_chirp,t,x0,v0,alpha)


# Saída no domínio da frequência
f_imp_NLlow, Pxx_den_NLlow = signal.welch(disp_imp_NLlow, fs=Fs, window='hann', nperseg=Np, noverlap=0, nfft=len(t), detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')
f_imp_NLhigh, Pxx_den_NLhigh = signal.welch(disp_imp_NLhigh, fs=Fs, window='hann', nperseg=Np, noverlap=0, nfft=len(t), detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')

pl.figure()
pl.grid(which='major', axis='both')
pl.semilogy(f_imp_NLhigh, Pxx_den_NLhigh,'b', label='Duffing com Trinca')
pl.semilogy(f_imp_NLlow, Pxx_den_NLlow,'--r', label='Linear')
pl.xlabel('Frequência [Hz]')
pl.ylabel('Densidade Espectral [V$^2$/Hz]')
pl.xlim(0,250)
# pl.legend()
# pl.title('Densidade impulso no domínio da frequência')
pl.show()

#%% Espectograma do Duffing

F_chirp_min = 0.1*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
acc_chirp_min,vel_chirp_min,disp_chirp_min_Duff = Linear_oscillator(ms,c,k1,k2,k3,F_chirp_min,t,x0,v0,alpha)

pl.figure()
pl.specgram(disp_chirp_min_Duff, Fs=Fs, cmap='jet',mode='psd')
pl.ylim(0,250)
pl.colorbar(label='[dB]')
pl.ylabel('Frequência [Hz]')
pl.xlabel('Tempo [s]')
pl.show()

F_chirp_max = 1*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
acc_chirp_max,vel_chirp_max,disp_chirp_max_Duff = Linear_oscillator(ms,c,k1,k2,k3,F_chirp_max,t,x0,v0,alpha)

pl.figure()
pl.specgram(disp_chirp_max_Duff, Fs=Fs, cmap='jet',mode='psd')
pl.ylim(0,250)
pl.colorbar(label='[dB]')
pl.ylabel('Frequência [Hz]')
pl.xlabel('Tempo [s]')
pl.show()

#%% Exemplo de resposta do Oscilador Linear com Trinca
alpha=0.9
k2 = 0#3.24e+4
k3 = 0#4.68e+7

F_impulse = np.zeros((len(t),)) # cria um vetor de zeros
F_impulse[1] = 1 # aloca a amplitude da força na posição inicial (na verdade na segunda amostra)
acc_chirp_min,vel_chirp_min,disp_imp_Crack = Linear_oscillator(ms,c,k1,k2,k3,F_impulse,t,x0,v0,alpha)
acc_chirp_min,vel_chirp_min,disp_imp_Lin = Linear_oscillator(ms,c,k1,0,0,F_impulse,t,x0,v0,1)

# Entrada de impulso unitário
pl.figure()
pl.grid(b=None, which='major', axis='both')
pl.plot(t,1000*disp_imp_Crack,'b', label='Duffing')
pl.plot(t,1000*disp_imp_Lin,'--r',  label='Linear')
pl.xlabel('Tempo [s]')
pl.ylabel('Deslocamento [mm]')
pl.xlim(0,1)
#pl.legend()
#pl.title('Entrada de varredura senoidal')
pl.show()

#%% Resposta do oscilador com trinca no domínio da frequência

F_chirp = 1*np.sin(wn*t) # amplitude * função seno (valor do angulo, quanto maior esse valor aqui dentro, menor será a frequência)
acc_imp_NLlow,vel_imp_NLlow,disp_imp_NLlow = Linear_oscillator(ms,c,k1,0,0,F_chirp,t,x0,v0,1)
acc_imp_NLhigh,vel_imp_NLhigh,disp_imp_NLhigh = Linear_oscillator(ms,c,k1,k2,k3,F_chirp,t,x0,v0,alpha)


# Saída no domínio da frequência
f_imp_NLlow, Pxx_den_NLlow = signal.welch(disp_imp_NLlow, fs=Fs, window='hann', nperseg=Np, noverlap=0, nfft=len(t), detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')
f_imp_NLhigh, Pxx_den_NLhigh = signal.welch(disp_imp_NLhigh, fs=Fs, window='hann', nperseg=Np, noverlap=0, nfft=len(t), detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')

pl.figure()
pl.grid(which='major', axis='both')
pl.semilogy(f_imp_NLhigh, Pxx_den_NLhigh,'b', label='Duffing com Trinca')
pl.semilogy(f_imp_NLlow, Pxx_den_NLlow,'--r', label='Linear')
pl.xlabel('Frequência [Hz]')
pl.ylabel('Densidade Espectral [V$^2$/Hz]')
pl.xlim(0,250)
# pl.legend()
# pl.title('Densidade impulso no domínio da frequência')
pl.show()


F_impulse = np.zeros((len(t),)) # cria um vetor de zeros
F_impulse[1] = 1 # aloca a amplitude da força na posição inicial (na verdade na segunda amostra)
acc_imp_NLlow,vel_imp_NLlow,disp_imp_NLlow = Linear_oscillator(ms,c,k1,0,0,F_impulse,t,x0,v0,1)
acc_imp_NLhigh,vel_imp_NLhigh,disp_imp_NLhigh = Linear_oscillator(ms,c,k1,k2,k3,F_impulse,t,x0,v0,alpha)


# Saída no domínio da frequência
f_imp_NLlow, Pxx_den_NLlow = signal.welch(disp_imp_NLlow, fs=Fs, window='boxcar', nperseg=Np, noverlap=0, nfft=len(t), detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')
f_imp_NLhigh, Pxx_den_NLhigh = signal.welch(disp_imp_NLhigh, fs=Fs, window='boxcar', nperseg=Np, noverlap=0, nfft=len(t), detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')

pl.figure()
pl.grid(which='major', axis='both')
pl.semilogy(f_imp_NLhigh, Pxx_den_NLhigh,'b', label='Duffing com Trinca')
pl.semilogy(f_imp_NLlow, Pxx_den_NLlow,'--r', label='Linear')
pl.xlabel('Frequência [Hz]')
pl.ylabel('Densidade Espectral [V$^2$/Hz]')
pl.ylim(10e-14,10e-10)
pl.xlim(5,35)
# pl.legend()
# pl.title('Densidade impulso no domínio da frequência')
pl.show()

#%% Espectograma com trinca

F_chirp_min = 1*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
acc_chirp_min,vel_chirp_min,disp_chirp_min_Duff = Linear_oscillator(ms,c,k1,k2,k3,F_chirp_min,t,x0,v0,alpha)

pl.figure()
pl.specgram(disp_chirp_min_Duff, Fs=Fs, cmap='jet',mode='psd')
pl.ylim(0,100)
pl.colorbar(label='[dB]')
pl.ylabel('Frequência [Hz]')
pl.xlabel('Tempo [s]')
pl.show()

#%% Exemplo de resposta do Oscilador de Duffig com Trinca
alpha=0.9
k2 = 3.24e+4
k3 = 4.68e+7

F_chirp_min = 0.1*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
acc_chirp_min,vel_chirp_min,disp_chirp_min_Duff = Linear_oscillator(ms,c,k1,k2,k3,F_chirp_min,t,x0,v0,alpha)
acc_chirp_min,vel_chirp_min,disp_chirp_min_Lin = Linear_oscillator(ms,c,k1,0,0,F_chirp_min,t,x0,v0,1)


Dif_DuffL = disp_chirp_min_Duff - disp_chirp_min_Lin


# Entrada de varredura senoidal
pl.figure()
pl.grid(b=None, which='major', axis='both')
pl.plot(t,1000*disp_chirp_min_Duff,'b', label='Duffing')
pl.plot(t,1000*disp_chirp_min_Lin,'--r',  label='Linear')
pl.plot(t,1000*Dif_DuffL,'k',alpha=0.8, label='Diferença')
pl.xlabel('Tempo [s]')
pl.ylabel('Deslocamento [mm]')
pl.xlim(0,2)
#pl.legend()
#pl.title('Entrada de varredura senoidal')
pl.show()

alpha=0.9
k2 = 3.24e+4
k3 = 4.68e+7

F_chirp_max = 1*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
acc_chirp_max,vel_chirp_max,disp_chirp_max_Duff = Linear_oscillator(ms,c,k1,k2,k3,F_chirp_max,t,x0,v0,alpha)
acc_chirp_max,vel_chirp_max,disp_chirp_max_Lin = Linear_oscillator(ms,c,k1,0,0,F_chirp_max,t,x0,v0,1)

Dif_DuffH = disp_chirp_max_Duff - disp_chirp_max_Lin

# Saída de deslocamento do sistema
pl.figure()
pl.grid(b=None, which='major', axis='both')
pl.plot(t,1000*disp_chirp_max_Duff,'b', label='Duffing')
pl.plot(t,1000*disp_chirp_max_Lin, '--r', label='Linear')
pl.plot(t,1000*Dif_DuffH,'k',alpha=0.8, label='Diferença')
pl.xlabel('Tempo [s]')
pl.ylabel('Deslocamento [mm]')
pl.xlim(0,2)
#pl.legend()
#pl.title('Entrada de varredura senoidal')
pl.show()

#%% Resposta do oscilador de Duffing com Trinca no domínio da frequência
alpha=0.99
F_chirp = 0.02*np.sin(wn*t) # amplitude * função seno (valor do angulo, quanto maior esse valor aqui dentro, menor será a frequência)
acc_imp_NLlow,vel_imp_NLlow,disp_imp_NLlow = Linear_oscillator(ms,c,k1,0,0,F_chirp,t,x0,v0,1)
acc_imp_NLhigh,vel_imp_NLhigh,disp_imp_NLhigh = Linear_oscillator(ms,c,k1,k2,k3,F_chirp,t,x0,v0,alpha)


# Saída no domínio da frequência
f_imp_NLlow, Pxx_den_NLlow = signal.welch(disp_imp_NLlow, fs=Fs, window='hann', nperseg=Np, noverlap=0, nfft=len(t), detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')
f_imp_NLhigh, Pxx_den_NLhigh = signal.welch(disp_imp_NLhigh, fs=Fs, window='hann', nperseg=Np, noverlap=0, nfft=len(t), detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')

pl.figure()
pl.grid(which='major', axis='both')
pl.semilogy(f_imp_NLhigh, Pxx_den_NLhigh,'b', label='Duffing com Trinca')
pl.semilogy(f_imp_NLlow, Pxx_den_NLlow,'--r', label='Linear')
pl.xlabel('Frequência [Hz]')
pl.ylabel('Densidade Espectral [V$^2$/Hz]')
pl.xlim(0,100)
# pl.legend()
# pl.title('Densidade impulso no domínio da frequência')
pl.show()

alpha=0.99
F_chirp = 0.8*np.sin(wn*t) # amplitude * função seno (valor do angulo, quanto maior esse valor aqui dentro, menor será a frequência)
acc_imp_NLlow,vel_imp_NLlow,disp_imp_NLlow = Linear_oscillator(ms,c,k1,0,0,F_chirp,t,x0,v0,1)
acc_imp_NLhigh,vel_imp_NLhigh,disp_imp_NLhigh = Linear_oscillator(ms,c,k1,k2,k3,F_chirp,t,x0,v0,alpha)


# Saída no domínio da frequência
f_imp_NLlow, Pxx_den_NLlow = signal.welch(disp_imp_NLlow, fs=Fs, window='hann', nperseg=Np, noverlap=0, nfft=len(t), detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')
f_imp_NLhigh, Pxx_den_NLhigh = signal.welch(disp_imp_NLhigh, fs=Fs, window='hann', nperseg=Np, noverlap=0, nfft=len(t), detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')

pl.figure()
pl.grid(which='major', axis='both')
pl.semilogy(f_imp_NLhigh, Pxx_den_NLhigh,'b', label='Duffing com Trinca')
pl.semilogy(f_imp_NLlow, Pxx_den_NLlow,'--r', label='Linear')
pl.xlabel('Frequência [Hz]')
pl.ylabel('Densidade Espectral [V$^2$/Hz]')
pl.xlim(0,100)
# pl.legend()
# pl.title('Densidade impulso no domínio da frequência')
pl.show()

alpha=0.9
F_chirp = 0.02*np.sin(wn*t) # amplitude * função seno (valor do angulo, quanto maior esse valor aqui dentro, menor será a frequência)
acc_imp_NLlow,vel_imp_NLlow,disp_imp_NLlow = Linear_oscillator(ms,c,k1,0,0,F_chirp,t,x0,v0,1)
acc_imp_NLhigh,vel_imp_NLhigh,disp_imp_NLhigh = Linear_oscillator(ms,c,k1,k2,k3,F_chirp,t,x0,v0,alpha)


# Saída no domínio da frequência
f_imp_NLlow, Pxx_den_NLlow = signal.welch(disp_imp_NLlow, fs=Fs, window='hann', nperseg=Np, noverlap=0, nfft=len(t), detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')
f_imp_NLhigh, Pxx_den_NLhigh = signal.welch(disp_imp_NLhigh, fs=Fs, window='hann', nperseg=Np, noverlap=0, nfft=len(t), detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')

pl.figure()
pl.grid(which='major', axis='both')
pl.semilogy(f_imp_NLhigh, Pxx_den_NLhigh,'b', label='Duffing com Trinca')
pl.semilogy(f_imp_NLlow, Pxx_den_NLlow,'--r', label='Linear')
pl.xlabel('Frequência [Hz]')
pl.ylabel('Densidade Espectral [V$^2$/Hz]')
pl.xlim(0,100)
# pl.legend()
# pl.title('Densidade impulso no domínio da frequência')
pl.show()

alpha=0.9
F_chirp = 0.8*np.sin(wn*t) # amplitude * função seno (valor do angulo, quanto maior esse valor aqui dentro, menor será a frequência)
acc_imp_NLlow,vel_imp_NLlow,disp_imp_NLlow = Linear_oscillator(ms,c,k1,0,0,F_chirp,t,x0,v0,1)
acc_imp_NLhigh,vel_imp_NLhigh,disp_imp_NLhigh = Linear_oscillator(ms,c,k1,k2,k3,F_chirp,t,x0,v0,alpha)


# Saída no domínio da frequência
f_imp_NLlow, Pxx_den_NLlow = signal.welch(disp_imp_NLlow, fs=Fs, window='hann', nperseg=Np, noverlap=0, nfft=len(t), detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')
f_imp_NLhigh, Pxx_den_NLhigh = signal.welch(disp_imp_NLhigh, fs=Fs, window='hann', nperseg=Np, noverlap=0, nfft=len(t), detrend=False, return_onesided=True, scaling='density', axis=-1, average='mean')

pl.figure()
pl.grid(which='major', axis='both')
pl.semilogy(f_imp_NLhigh, Pxx_den_NLhigh,'b', label='Duffing com Trinca')
pl.semilogy(f_imp_NLlow, Pxx_den_NLlow,'--r', label='Linear')
pl.xlabel('Frequência [Hz]')
pl.ylabel('Densidade Espectral [V$^2$/Hz]')
pl.xlim(0,100)
# pl.legend()
# pl.title('Densidade impulso no domínio da frequência')
pl.show()


#%% Espectograma do Duffing com Trinca

alpha=0.99
F_chirp_min = 0.1*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
acc_chirp_min,vel_chirp_min,disp_chirp_min_Duff = Linear_oscillator(ms,c,k1,k2,k3,F_chirp_min,t,x0,v0,alpha)

pl.figure()
pl.specgram(disp_chirp_min_Duff, Fs=Fs, cmap='jet',mode='psd')
pl.ylim(0,250)
pl.colorbar(label='[dB]')
pl.ylabel('Frequência [Hz]')
pl.xlabel('Tempo [s]')
pl.show()

alpha=0.99
F_chirp_max = 1*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
acc_imp_NLhigh,vel_imp_NLhigh,disp_chirp_max_Duff = Linear_oscillator(ms,c,k1,k2,k3,F_chirp_max,t,x0,v0,alpha)

pl.figure()
pl.specgram(disp_chirp_max_Duff, Fs=Fs, cmap='jet',mode='psd')
pl.ylim(0,250)
pl.colorbar(label='[dB]')
pl.ylabel('Frequência [Hz]')
pl.xlabel('Tempo [s]')
pl.show()


alpha=0.9
F_chirp_min = 0.1*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
acc_chirp_min,vel_chirp_min,disp_chirp_min_Duff = Linear_oscillator(ms,c,k1,k2,k3,F_chirp_min,t,x0,v0,alpha)

pl.figure()
pl.specgram(disp_chirp_min_Duff, Fs=Fs, cmap='jet',mode='psd')
pl.ylim(0,250)
pl.colorbar(label='[dB]')
pl.ylabel('Frequência [Hz]')
pl.xlabel('Tempo [s]')
pl.show()

alpha=0.9
F_chirp_max = 1*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
acc_imp_NLhigh,vel_imp_NLhigh,disp_chirp_max_Duff = Linear_oscillator(ms,c,k1,k2,k3,F_chirp_max,t,x0,v0,alpha)

pl.figure()
pl.specgram(disp_chirp_max_Duff, Fs=Fs, cmap='jet',mode='psd')
pl.ylim(0,250)
pl.colorbar(label='[dB]')
pl.ylabel('Frequência [Hz]')
pl.xlabel('Tempo [s]')
pl.show()
