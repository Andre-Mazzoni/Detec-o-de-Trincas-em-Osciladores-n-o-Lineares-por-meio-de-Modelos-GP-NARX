# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:58:10 2022

@author: microsoft
"""

#%% Pacotes utilizados nas simulações
import matplotlib.pyplot as pl # para os gráficos
import numpy as np
import scipy.signal as signal # chirp
import math as math
import GPy

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
def Linear_oscillator(m,c,k,k2,k3,F,t,disp0,vel0): #NONLinear_oscilator
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
    # Para a rigidez não linear: adicionar componentes de k2 e k3 na próxima linha
    acc[0,] = (1/m)*(F[0,] -c*vel[0,] -k*disp[0,] -k2*(disp[0,]**2) -k3*(disp[0,]**3))
    dt = t[2]-t[1] 
    for i in range(1,len(t)):
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
    
#%% Entrada chirp
F_chirp_min = 0.1*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0) 
acc_chirp_min,vel_chirp_min,disp_chirp_min = Linear_oscillator(ms,c,k1,k2,k3,F_chirp_min,t,x0,v0)

F_chirp_max = 1*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0) 
acc_chirp_max,vel_chirp_max,disp_chirp_max = Linear_oscillator(ms,c,k1,k2,k3,F_chirp_max,t,x0,v0)
#%% Entrada aleatória
F_rand = 5*np.random.normal(0,1,size=(Np,)) 
filt = signal.butter(4, [wmin, wmax], btype='band', fs=Fs, output='sos')
F_rand = signal.sosfiltfilt(filt,F_rand) 
acc_rand,vel_rand,disp_rand = Linear_oscillator(ms,c,k1,k2,k3,F_rand,t,x0,v0)
#%%

# Número de regressores
n_y=11
n_u=3
n_maior = max(n_y, n_u)

#%%
# Entrada e saída de treinamento
u1 = F_chirp_min
y1 = disp_chirp_min

# Normalização
std_sig_input,std_sig_output,mean_sig_input,mean_sig_output= normalizar(u1,y1)

# Adição do ruído
y1 = y1 + np.random.normal(0, 0.05*std_sig_output, size=len(t))
                                                                                # Normalizar só com o menor sinal deu os melhores resultados
u1 = (u1 - mean_sig_input)/std_sig_input
y1 = (y1 - mean_sig_output)/std_sig_output
#%% Gaussian Process

# Matriz e vetor para treino
x_train1 = np.zeros((int(Np - n_maior),n_u+n_y))
y_p1 = np.zeros((int(Np)- n_maior,1))

count = 0
for m in range(n_maior, Np):
    x_train1[count,0:n_u] = u1[m - n_u + 1:m + 1,]
    x_train1[count,n_u:n_y + n_u] = y1[m - n_y:m,]
    y_p1[count,0] = y1[m,]
    count = count + 1
    
#%%
# Entrada e saída de treinamento
u2 = F_chirp_max
y2 = disp_chirp_max

# Adição do ruído

# Adição do ruído
y2 = y2 + np.random.normal(0, 0.05*std_sig_output, size=len(t))

# Normalização
# std_sig_input,std_sig_output,mean_sig_input,mean_sig_output= normalizar(u2,y2)
u2 = (u2 - mean_sig_input)/std_sig_input
y2 = (y2 - mean_sig_output)/std_sig_output

#%% Gaussian Process

# Matriz e vetor para treino
x_train2 = np.zeros((int(Np - n_maior),n_u+n_y))
y_p2 = np.zeros((int(Np)- n_maior,1))

count = 0
for m in range(n_maior, Np):
    x_train2[count,0:n_u] = u2[m - n_u + 1:m + 1,]
    x_train2[count,n_u:n_y + n_u] = y2[m - n_y:m,]
    y_p2[count,0] = y2[m,]
    count = count + 1
      
#%%
x_train = np.zeros(2*(int(Np - n_maior),n_u+n_y))
y_p = np.zeros(2*(int(Np)- n_maior,1))

x_train = np.array([*x_train1, *x_train2])
y_p = np.array([*y_p1, *y_p2]) 

#%% Kernel e modelo RBF
kernel_RBF =  GPy.kern.RBF(input_dim=n_y+n_u,ARD=False)
gp_model_RBF = GPy.models.GPRegression(x_train,y_p,kernel = kernel_RBF)  
gp_model_RBF.optimize('lbfgsb', max_iters = 1e20,messages=False)
gp_model_RBF.optimize_restarts(num_restarts = 5, parallel=False)
# print (gp_model_RBF)

#%% Amplitudes Random RMS

samples=250
div=20  #40
amost=samples*div

Fit_rms_rand = np.zeros(amost)
Fit_med_rms = np.zeros(div)
rms_rand     = np.zeros(amost)
rms_rand_out = np.zeros(amost)
rms_med_rand = np.zeros(div)
amp     = np.zeros(amost)

std_med_rms = np.zeros(div)
Fit_max_rms = np.zeros(div)
Fit_min_rms = np.zeros(div)

# med_samp=0
cont=1
fit_fora=0

for A in range(1,amost+1):

    a=(cont*0.17)
    
    # print (A)
    # print (a)
    # print ('\n')
    
    amp[A-1]=a
    #Entrada e saída
    F_chirp = a*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
    rms_chirp = np.sqrt(np.mean(F_chirp**2))
    
    print ('\n')
    print ('Amostra: '+ str(A))
    print ('RMS da entrada: '+ str(rms_chirp))
    
    rms_rand[A-1]=rms_chirp
    #Entrada e saída
    F_rand = 1*np.random.normal(0,1,size=(Np,)) 
    filt = signal.butter(4, [wmin, wmax], btype='band', fs=Fs, output='sos')
    F_rand = signal.sosfiltfilt(filt,F_rand) 
    F_rand = rms_chirp*(F_rand/(np.sqrt(np.mean(F_rand**2))))
    acc_rand,vel_rand,disp_rand = Linear_oscillator(ms,c,k1,k2,k3,F_rand,t,x0,v0)
        
    u = F_rand     # Chirp dá alguns erros overflow e invalid value
    y = disp_rand
    rms_out=np.sqrt(np.mean(disp_rand**2))
    print ('RMS da saída: '+ str(rms_out))
    rms_rand_out[A-1]=rms_out
    #Adição do ruído
    y = y + np.random.normal(0, 0.05*std_sig_output, size=len(t))

    #Normalização
    u = (u - mean_sig_input)/std_sig_input
    y = (y - mean_sig_output)/std_sig_output
    
    # RBF
    n_t = Np
    x_test_RBF = np.zeros((n_t,n_u+n_y)) 
    vari_RBF = np.zeros((n_t))
    y_GP_RBF = np.zeros((n_t))
    y_GP_RBF[0:n_maior] = y[0:n_maior]
    
    for m in range(max(n_u,n_y), n_t):
        x_test_RBF[m,0:n_u] = u[m - n_u + 1:m + 1,]
        x_test_RBF[m,n_u:n_y + n_u] = y_GP_RBF[m - n_y:m]
        y_GP_RBF[m],vari_RBF[m] = gp_model_RBF.predict(x_test_RBF[m,:].reshape(-1,n_u+n_y))
    
    erro_RBF=np.zeros((n_t))
    erro_RBF= y - y_GP_RBF
    
    # Fit RBF
    ErrNum=0
    for i in range(n_maior,n_t):
        ErrNum=ErrNum+(erro_RBF[i,])**2
        ErrDem=0
    for i in range(n_maior,n_t):
        ErrDem=ErrDem+(y_GP_RBF[i,])**2
    # Fit_rms_rand[A-1]=100*(1-math.sqrt(ErrNum/ErrDem))
    # print('Fit: '+str(Fit_rms_rand[A-1,0]))
    
    if (1-math.sqrt(ErrNum/ErrDem)) > 0:
        Fit_rms_rand[A-1]=100*(1-math.sqrt(ErrNum/ErrDem))
        print('Fit: '+str(Fit_rms_rand[A-1]))
    else:
        Fit_rms_rand[A-1]=0
        print('fit negativo')
        fit_fora=fit_fora+1
    
    if A/samples == A//samples:
        print ('\nentrou no for')
        # for p in range (A-samples,A):
        #     med_samp=med_samp+Fit_RBF_rand[p,0]
        #     print(p)
        #     # print()
        # Fit_med_rand[cont-1,0]=med_samp/samples
        Fit_med_rms[cont-1]=np.mean(Fit_rms_rand[A-samples:A])
        std_med_rms[cont-1]=np.std(Fit_rms_rand[A-samples:A])
        print(Fit_med_rms[cont-1])
        # med_samp=0
        rms_med_rand[cont-1]=rms_chirp
        cont=cont+1

cont=1
Fit_max_rms = np.zeros(div)
Fit_min_rms = np.zeros(div)
for A in range(1,amost+1):
    if A/samples == A//samples:
        Fit_max_rms[cont-1]=max(Fit_rms_rand[A-samples:A])
        Fit_min_rms[cont-1]=min(Fit_rms_rand[A-samples:A])
        cont=cont+1

#%polyfit
Fit_rms_rand2 = np.zeros(amost)
rms_rand2     = np.zeros(amost)
rms_rand_out2 = np.zeros(amost)
for z in range (0,amost):
    Fit_rms_rand2[z] = Fit_rms_rand[z]
    rms_rand2[z]     = rms_rand[z]
    rms_rand_out2[z] = rms_rand_out[z]


order = rms_rand_out2.argsort()
Fit_rms_rand2 = Fit_rms_rand2[order]
rms_rand2 = rms_rand2[order]
rms_rand_out2 = rms_rand_out2[order]


#% média dos rms de saida

Fit_med_rms2 = np.zeros(div)
rms_med_rand2 = np.zeros(div)

order = rms_rand2.argsort()
Fit_rms_rand2 = Fit_rms_rand2[order]
rms_rand_out2 = rms_rand_out2[order]
rms_rand2 = rms_rand2[order]

cont=0
for z in range (0,amost):
    if z/samples == z//samples:
        Fit_med_rms2[cont]=np.mean(Fit_rms_rand2[z-samples:z])
        rms_med_rand2[cont]=np.mean(rms_rand2[z-samples:z])
        cont=cont+1

#%%
pl.figure()
pl.grid(b=None, which='major', axis='both')
# pl.plot(rms_rand_out,Fit_rms_rand,'o', label='Fit_RBF')
pl.plot(1000*rms_med_rand2,Fit_med_rms2,'--k',marker='o')
# pl.legend()
# pl.xlabel('Valor Eficaz da Entrada [mN]')
# pl.ylabel('Aproximação [\%]')
pl.xlabel('Média RMS da Entrada [mN]')
pl.ylabel('$fit$ [\%]')
pl.xlim(0.0,2400)
pl.ylim(55,100)
pl.show()

#%% Amplitudes Chirp RMS

samples=50
div=20  #40
amost=samples*div

Fit_rms_chirp = np.zeros(amost)
Fit_med_chirp = np.zeros(div)
rms_chirp     = np.zeros(amost)
rms_chirp_out = np.zeros(amost)
rms_med_chirp = np.zeros(div)
amp     = np.zeros(amost)

Fit_max_chirp = np.zeros(div)
Fit_min_chirp = np.zeros(div)

# med_samp=0
cont=1
fit_fora=0

for A in range(1,amost+1):

    a=0.1+((cont-1)*0.1)
    
    # print (A)
    # print (a)
    # print ('\n')
    
    amp[A-1]=a
    #Entrada e saída
    F_chirp = a*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
    rms_chirpi = np.sqrt(np.mean(F_chirp**2))
    acc_chirp,vel_chirp,disp_chirp = Linear_oscillator(ms,c,k1,k2,k3,F_chirp,t,x0,v0)
    
    print ('\n')
    print ('Amostra: '+ str(A))
    print ('RMS da entrada: '+ str(rms_chirpi))
    
    rms_chirp[A-1]=rms_chirpi
    #Entrada e saída
        
    u = F_chirp     # Chirp dá alguns erros overflow e invalid value
    y = disp_chirp
    rms_out=np.sqrt(np.mean(disp_chirp**2))
    print ('RMS da saída: '+ str(rms_out))
    rms_chirp_out[A-1]=rms_out
    #Adição do ruído
    y = y + np.random.normal(0, 0.05*std_sig_output, size=len(t))

    #Normalização
    u = (u - mean_sig_input)/std_sig_input
    y = (y - mean_sig_output)/std_sig_output
    
    # RBF
    n_t = Np
    x_test_RBF = np.zeros((n_t,n_u+n_y)) 
    vari_RBF = np.zeros((n_t))
    y_GP_RBF = np.zeros((n_t))
    y_GP_RBF[0:n_maior] = y[0:n_maior]
    
    for m in range(max(n_u,n_y), n_t):
        x_test_RBF[m,0:n_u] = u[m - n_u + 1:m + 1,]
        x_test_RBF[m,n_u:n_y + n_u] = y_GP_RBF[m - n_y:m]
        y_GP_RBF[m],vari_RBF[m] = gp_model_RBF.predict(x_test_RBF[m,:].reshape(-1,n_u+n_y))
    
    erro_RBF=np.zeros((n_t))
    erro_RBF= y - y_GP_RBF
    
    # Fit RBF
    ErrNum=0
    for i in range(n_maior,n_t):
        ErrNum=ErrNum+(erro_RBF[i,])**2
        ErrDem=0
    for i in range(n_maior,n_t):
        ErrDem=ErrDem+(y_GP_RBF[i,])**2
    
    if (1-math.sqrt(ErrNum/ErrDem)) > 0:
        Fit_rms_chirp[A-1]=100*(1-math.sqrt(ErrNum/ErrDem))
        print('Fit: '+str(Fit_rms_chirp[A-1]))
    else:
        Fit_rms_chirp[A-1]=0
        print('fit negativo')
        fit_fora=fit_fora+1
    
    if A/samples == A//samples:
        print ('\nentrou no for')
        # for p in range (A-samples,A):
        #     med_samp=med_samp+Fit_RBF_rand[p,0]
        #     print(p)
        #     # print()
        # Fit_med_rand[cont-1,0]=med_samp/samples
        Fit_med_chirp[cont-1]=np.mean(Fit_rms_chirp[A-samples:A])
        print(Fit_med_chirp[cont-1])
        # med_samp=0
        rms_med_chirp[cont-1]=rms_chirpi
        cont=cont+1

cont=1
Fit_max_rms = np.zeros(div)
Fit_min_rms = np.zeros(div)
for A in range(1,amost+1):
    if A/samples == A//samples:
        Fit_max_chirp[cont-1]=max(Fit_rms_chirp[A-samples:A])
        Fit_min_chirp[cont-1]=min(Fit_rms_chirp[A-samples:A])
        cont=cont+1


#%polyfit
Fit_rms_chirp2 = np.zeros(amost)
rms_chirp2     = np.zeros(amost)
rms_chirp_out2 = np.zeros(amost)
for z in range (0,amost):
    Fit_rms_chirp2[z] = Fit_rms_chirp[z]
    rms_chirp2[z]     = rms_chirp[z]
    rms_chirp_out2[z] = rms_chirp_out[z]


order = rms_chirp_out2.argsort()
Fit_rms_chirp2 = Fit_rms_chirp2[order]
rms_chirp2 = rms_chirp2[order]
rms_chirp_out2 = rms_chirp_out2[order]

polyfit=np.polyfit(rms_chirp_out2,Fit_rms_chirp2,6)
polypontos = [np.polyval(polyfit, i) for i in rms_chirp_out2]


#% média dos rms de saida

Fit_med_chirp2 = np.zeros(div)
rms_med_chirp2 = np.zeros(div)

order = rms_chirp2.argsort()
Fit_rms_chirp2 = Fit_rms_chirp2[order]
rms_chirp_out2 = rms_chirp_out2[order]
rms_chirp2 = rms_chirp2[order]

cont=0
for z in range (0,amost):
    if z/samples == z//samples:
        Fit_med_chirp2[cont]=np.mean(Fit_rms_chirp2[z-samples:z])
        rms_med_chirp2[cont]=np.mean(rms_chirp2[z-samples:z])
        cont=cont+1
        
#%%
pl.figure()
pl.grid(b=None, which='major', axis='both')
# pl.plot(rms_rand_out,Fit_rms_rand,'o', label='Fit_RBF')
pl.plot(1000*rms_med_chirp2,Fit_med_chirp2,'--k',marker='o')
# pl.legend()
# pl.xlabel('Valor Eficaz da Entrada [mN]')
# pl.ylabel('Aproximação [\%]')
pl.xlabel('Média RMS da Entrada [mN]')
pl.ylabel('$fit$ [\%]')
pl.xlim(0.0,1400)
pl.ylim(55,100)
pl.show()
