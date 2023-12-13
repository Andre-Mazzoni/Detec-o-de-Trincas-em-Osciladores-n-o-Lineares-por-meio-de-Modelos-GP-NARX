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
# Entrada e saída de treinamento com baixa excitação
F_chirp_min = 0.1*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0) 
acc_chirp_min,vel_chirp_min,disp_chirp_min = Linear_oscillator(ms,c,k1,k2,k3,F_chirp_min,t,x0,v0)


u1 = F_chirp_min
y1 = disp_chirp_min

# Normalização
std_sig_input,std_sig_output,mean_sig_input,mean_sig_output= normalizar(u1,y1)

# Adição do ruído
y1 = y1 + np.random.normal(0, 0.05*std_sig_output, size=len(t))
                                                                                # Normalizar só com o menor sinal deu os melhores resultados
u1 = (u1 - mean_sig_input)/std_sig_input
y1 = (y1 - mean_sig_output)/std_sig_output
    
#%%
# Entrada e saída de treinamento com alta excitação
F_chirp_max = 1*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0) 
acc_chirp_max,vel_chirp_max,disp_chirp_max = Linear_oscillator(ms,c,k1,k2,k3,F_chirp_max,t,x0,v0)

u2 = F_chirp_max
y2 = disp_chirp_max

# Adição do ruído
y2 = y2 + np.random.normal(0, 0.05*std_sig_output, size=len(t))

# Normalização
# std_sig_input,std_sig_output,mean_sig_input,mean_sig_output= normalizar(u2,y2)
u2 = (u2 - mean_sig_input)/std_sig_input
y2 = (y2 - mean_sig_output)/std_sig_output

#%% Entrada aleatória para treinamento
F_rand = 5*np.random.normal(0,1,size=(Np,)) 
filt = signal.butter(4, [wmin, wmax], btype='band', fs=Fs, output='sos') # Amplitude da entrada 3x maior que a da chirp, para obter a saída com amplitude próxima
F_rand = signal.sosfiltfilt(filt,F_rand) 
acc_rand,vel_rand,disp_rand = Linear_oscillator(ms,c,k1,k2,k3,F_rand,t,x0,v0)

uT = F_rand
yT = disp_rand

# Adição do ruído
yT = yT + np.random.normal(0, 0.05*std_sig_output, size=len(t))

# Normalização
# std_sig_input,std_sig_output,mean_sig_input,mean_sig_output= normalizar(uT,y2)
uT = (uT - mean_sig_input)/std_sig_input
yT = (yT - mean_sig_output)/std_sig_output


#%%
passo = 1
rest = 3

#%% RBF com aleatória alta
sup_RBF = np.zeros((14,14))
ny = np.linspace(2, 16, 14, endpoint=True)
nu = np.linspace(2, 16, 14, endpoint=True)

for n_y in range(2,16):
    print("\nn_y")
    print(n_y)
    for n_u in range(2,16):
        n_maior = max(n_y, n_u)
# treinamento com número de regressores
        
# Matriz e vetor para treino amplitude 1
        x_train1 = np.zeros((int(Np - n_maior),n_u+n_y))
        y_p1 = np.zeros((int(Np)- n_maior,1))

        count = 0
        for m in range(n_maior, Np):
            x_train1[count,0:n_u] = u1[m - n_u + 1:m + 1,]
            x_train1[count,n_u:n_y + n_u] = y1[m - n_y:m,]
            y_p1[count,0] = y1[m,]
            count = count + 1
            
# Matriz e vetor para treino amplitude 2
        x_train2 = np.zeros((int(Np - n_maior),n_u+n_y))
        y_p2 = np.zeros((int(Np)- n_maior,1))
        
        count = 0
        for m in range(n_maior, Np):
            x_train2[count,0:n_u] = u2[m - n_u + 1:m + 1,]
            x_train2[count,n_u:n_y + n_u] = y2[m - n_y:m,]
            y_p2[count,0] = y2[m,]
            count = count + 1
      
# Matriz com duas amplitudes
        x_train = np.zeros(2*(int(Np - n_maior),n_u+n_y))
        y_p = np.zeros(2*(int(Np)- n_maior,1))

        x_train = np.array([*x_train1, *x_train2])
        y_p = np.array([*y_p1, *y_p2]) 

        x_train = x_train[0::passo,:]
        y_p = y_p[0::passo]
        
        kernel =  GPy.kern.RBF(input_dim=n_u + n_y, ARD=False)  #Funcionou bem 
        gp_model_lin = GPy.models.GPRegression(x_train,y_p,kernel = kernel)  
        gp_model_lin.optimize('lbfgsb', max_iters = 1e20,messages=False)
        gp_model_lin.optimize_restarts(num_restarts = rest, parallel=False)

        n_t = Np
        x_test1 = np.zeros((n_t,n_u+n_y)) 
        vari = np.zeros((n_t))
        y_GP = np.zeros((n_t))
        y_GP[0:n_maior] = yT[0:n_maior]

        for m in range(max(n_u,n_y), n_t):
            x_test1[m,0:n_u] = uT[m - n_u + 1:m + 1,]
            x_test1[m,n_u:n_y + n_u] = y_GP[m - n_y:m]
            y_GP[m],vari[m] = gp_model_lin.predict(x_test1[m,:].reshape(-1,n_u+n_y))


        y_max=y_GP + 3*np.sqrt(vari)
        y_min=y_GP - 3*np.sqrt(vari)

        erro=np.zeros((n_t))
        erro= yT - y_GP
        
        ErrNum=0
        for i in range(n_maior,n_t):
            ErrNum=ErrNum+(erro[i,])**2
            ErrDem=0
        for i in range(n_maior,n_t):
                ErrDem=ErrDem+(y_GP[i,])**2
        #calcular estimado e erro
        if 100*(1-math.sqrt(ErrNum/ErrDem)) < 0:
            sup_RBF[n_y-2,n_u-2]=0
        else:
            sup_RBF[n_y-2,n_u-2]=100*(1-math.sqrt(ErrNum/ErrDem))
        print(sup_RBF[n_y-2,n_u-2])

#%% RBF com CHIRP de amplitude alta

uT = u2
yT = y2

sup_h_RBF = np.zeros((14,14))
ny = np.linspace(2, 16, 14, endpoint=True)
nu = np.linspace(2, 16, 14, endpoint=True)

for n_y in range(2,16):
    print("\nn_y")
    print(n_y)
    for n_u in range(2,16):
        n_maior = max(n_y, n_u)
# treinamento com número de regressores
        
# Matriz e vetor para treino amplitude 1
        x_train1 = np.zeros((int(Np - n_maior),n_u+n_y))
        y_p1 = np.zeros((int(Np)- n_maior,1))

        count = 0
        for m in range(n_maior, Np):
            x_train1[count,0:n_u] = u1[m - n_u + 1:m + 1,]
            x_train1[count,n_u:n_y + n_u] = y1[m - n_y:m,]
            y_p1[count,0] = y1[m,]
            count = count + 1
            
# Matriz e vetor para treino amplitude 2
        x_train2 = np.zeros((int(Np - n_maior),n_u+n_y))
        y_p2 = np.zeros((int(Np)- n_maior,1))
        
        count = 0
        for m in range(n_maior, Np):
            x_train2[count,0:n_u] = u2[m - n_u + 1:m + 1,]
            x_train2[count,n_u:n_y + n_u] = y2[m - n_y:m,]
            y_p2[count,0] = y2[m,]
            count = count + 1
      
# Matriz com duas amplitudes
        x_train = np.zeros(2*(int(Np - n_maior),n_u+n_y))
        y_p = np.zeros(2*(int(Np)- n_maior,1))

        x_train = np.array([*x_train1, *x_train2])
        y_p = np.array([*y_p1, *y_p2]) 

        x_train = x_train[0::passo,:]
        y_p = y_p[0::passo]
        
        kernel =  GPy.kern.RBF(input_dim=n_u + n_y, ARD=False)  #Funcionou bem 
        gp_model_lin = GPy.models.GPRegression(x_train,y_p,kernel = kernel)  
        gp_model_lin.optimize('lbfgsb', max_iters = 1e20,messages=False)
        gp_model_lin.optimize_restarts(num_restarts = rest, parallel=False)
        
        n_t = Np
        x_test1 = np.zeros((n_t,n_u+n_y)) 
        vari = np.zeros((n_t))
        y_GP = np.zeros((n_t))
        y_GP[0:n_maior] = yT[0:n_maior]

        for m in range(max(n_u,n_y), n_t):
            x_test1[m,0:n_u] = uT[m - n_u + 1:m + 1,]
            x_test1[m,n_u:n_y + n_u] = y_GP[m - n_y:m]
            y_GP[m],vari[m] = gp_model_lin.predict(x_test1[m,:].reshape(-1,n_u+n_y))


        y_max=y_GP + 3*np.sqrt(vari)
        y_min=y_GP - 3*np.sqrt(vari)

        erro=np.zeros((n_t))
        erro= yT - y_GP
        
        ErrNum=0
        for i in range(n_maior,n_t):
            ErrNum=ErrNum+(erro[i,])**2 
            ErrDem=0
        for i in range(n_maior,n_t):
                ErrDem=ErrDem+(y_GP[i,])**2
        #calcular estimado e erro
        if 100*(1-math.sqrt(ErrNum/ErrDem)) < 0:
            sup_h_RBF[n_y-2,n_u-2]=0
        else:
            sup_h_RBF[n_y-2,n_u-2]=100*(1-math.sqrt(ErrNum/ErrDem))
        print(sup_h_RBF[n_y-2,n_u-2])  

#%% RBF com CHIRP de amplitude baixa
sup_l_RBF = np.zeros((14,14))

uT = u1
yT = y1

ny = np.linspace(2, 16, 14, endpoint=True)
nu = np.linspace(2, 16, 14, endpoint=True)

for n_y in range(2,16):
    print("\nn_y")
    print(n_y)
    for n_u in range(2,16):
        n_maior = max(n_y, n_u)
# treinamento com número de regressores
        
# Matriz e vetor para treino amplitude 1
        x_train1 = np.zeros((int(Np - n_maior),n_u+n_y))
        y_p1 = np.zeros((int(Np)- n_maior,1))

        count = 0
        for m in range(n_maior, Np):
            x_train1[count,0:n_u] = u1[m - n_u + 1:m + 1,]
            x_train1[count,n_u:n_y + n_u] = y1[m - n_y:m,]
            y_p1[count,0] = y1[m,]
            count = count + 1
            
# Matriz e vetor para treino amplitude 2
        x_train2 = np.zeros((int(Np - n_maior),n_u+n_y))
        y_p2 = np.zeros((int(Np)- n_maior,1))
        
        count = 0
        for m in range(n_maior, Np):
            x_train2[count,0:n_u] = u2[m - n_u + 1:m + 1,]
            x_train2[count,n_u:n_y + n_u] = y2[m - n_y:m,]
            y_p2[count,0] = y2[m,]
            count = count + 1
      
# Matriz com duas amplitudes
        x_train = np.zeros(2*(int(Np - n_maior),n_u+n_y))
        y_p = np.zeros(2*(int(Np)- n_maior,1))

        x_train = np.array([*x_train1, *x_train2])
        y_p = np.array([*y_p1, *y_p2]) 

        x_train = x_train[0::passo,:]
        y_p = y_p[0::passo]
        
        kernel =  GPy.kern.RBF(input_dim=n_u + n_y, ARD=False)  #Funcionou bem 
        gp_model_lin = GPy.models.GPRegression(x_train,y_p,kernel = kernel)  
        gp_model_lin.optimize('lbfgsb', max_iters = 1e20,messages=False)
        gp_model_lin.optimize_restarts(num_restarts = rest, parallel=False)
        
        n_t = Np
        x_test1 = np.zeros((n_t,n_u+n_y)) 
        vari = np.zeros((n_t))
        y_GP = np.zeros((n_t))
        y_GP[0:n_maior] = yT[0:n_maior] # aqui usando o yT

        for m in range(max(n_u,n_y), n_t):
            x_test1[m,0:n_u] = uT[m - n_u + 1:m + 1,]
            x_test1[m,n_u:n_y + n_u] = y_GP[m - n_y:m]
            y_GP[m],vari[m] = gp_model_lin.predict(x_test1[m,:].reshape(-1,n_u+n_y))

        erro=np.zeros((n_t))
        erro= yT - y_GP
        
        ErrNum=0
        for i in range(n_maior,n_t):
            ErrNum=ErrNum+(erro[i,])**2
            ErrDem=0
        for i in range(n_maior,n_t):
                ErrDem=ErrDem+(y_GP[i,])**2
        #calcular estimado e erro
        if 100*(1-math.sqrt(ErrNum/ErrDem)) < 0:
            sup_l_RBF[n_y-2,n_u-2]=0
        else:
            sup_l_RBF[n_y-2,n_u-2]=100*(1-math.sqrt(ErrNum/ErrDem))
        print(sup_l_RBF[n_y-2,n_u-2])

#%% Fit médio
Fit_med_RBF = np.zeros((16,16))
Fit_med_RBF = (sup_RBF+sup_h_RBF+sup_l_RBF)/3
 

#%%

nu = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
ny = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]

import seaborn as sns; sns.set_theme()
pl.figure()
imgmed = sns.heatmap(Fit_med_RBF, xticklabels=nu, yticklabels=ny, annot=True,cmap="viridis",fmt='.2g',cbar_kws={'label':'$fit$ [\%]'})
imgmed.set(xlabel='$n_U$', ylabel='$n_y$')
# pl.colorbar(ticks=range(100), label='Fit (%)')
pl.show()
