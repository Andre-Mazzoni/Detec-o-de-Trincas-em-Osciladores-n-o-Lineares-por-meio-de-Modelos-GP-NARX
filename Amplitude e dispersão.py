# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:58:10 2022

@author: microsoft
"""

#%% Pacotes utilizados nas simulações
import matplotlib.pyplot as pl # para os gráficos
import numpy as np
import scipy.signal as signal # chirp
import scipy.spatial as spatial
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

rms_chirp_max = np.sqrt(np.mean(u2**2))

# Adição do ruído
y2 = y2 + np.random.normal(0, 0.05*std_sig_output, size=len(t))

# Normalização
# std_sig_input,std_sig_output,mean_sig_input,mean_sig_output= normalizar(u2,y2)
u2 = (u2 - mean_sig_input)/std_sig_input
y2 = (y2 - mean_sig_output)/std_sig_output

#%% Entrada aleatória para treinamento
F_rand = 2.5*np.random.normal(0,1,size=(Np,)) 
filt = signal.butter(4, [wmin, wmax], btype='band', fs=Fs, output='sos') # Amplitude da entrada 3x maior que a da chirp, para obter a saída com amplitude próxima
F_rand = signal.sosfiltfilt(filt,F_rand) 
acc_rand,vel_rand,disp_rand = Linear_oscillator(ms,c,k1,k2,k3,F_rand,t,x0,v0)

uT = F_rand
yT = disp_rand

rms_rand_max = np.sqrt(np.mean(uT**2))


# Adição do ruído
yT = yT + np.random.normal(0, 0.05*std_sig_output, size=len(t))

# Normalização
# std_sig_input,std_sig_output,mean_sig_input,mean_sig_output= normalizar(uT,y2)
uT = (uT - mean_sig_input)/std_sig_input
yT = (yT - mean_sig_output)/std_sig_output

#%%

# Número de regressores
n_y=11
n_u=3
n_maior = max(n_y, n_u)

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

#%% 

y=yT
u=uT

n_t = Np
x_test1 = np.zeros((n_t,n_u+n_y)) 
vari = np.zeros((n_t))
y_GP = np.zeros((n_t))
y_GP[0:n_maior] = y[0:n_maior] #REVER

for m in range(max(n_u,n_y), n_t):
    x_test1[m,0:n_u] = u[m - n_u + 1:m + 1,]
    x_test1[m,n_u:n_y + n_u] = y_GP[m - n_y:m]
    y_GP[m],vari[m] = gp_model_RBF.predict(x_test1[m,:].reshape(-1,n_u+n_y))

erro=np.zeros((n_t))
erro= y - y_GP

std_erro_treino = np.std(erro)/np.std(y_GP)

pl.figure()
pl.grid(b=None, which='major', axis='both')
# pl.fill_between(t,(y_max+mean_sig_output)*std_sig_output, (y_min+mean_sig_output)*std_sig_output, alpha=0.8, label='$Margin$')
pl.plot(t,1000*(y+mean_sig_output)*std_sig_output, 'r', label='$y_{original}$')
pl.plot(t,1000*(y_GP+mean_sig_output)*std_sig_output,'--b', label='$y_{GP}$')
pl.plot(t,1000*(erro+mean_sig_output)*std_sig_output,'--c', label='$error$')
pl.xlabel('$Time$ [s]') 
# pl.ylabel('$y(t)$ [mm]')
# pl.legend()
pl.xlim(0,2)
#pl.title('Treinamento de dados')
pl.show()

pl.figure()
pl.grid(b=None, which='major', axis='both')
# pl.plot(t,1000*(vari),'--c', label='$variance$')
pl.plot(t,(np.sqrt(vari))*std_sig_output,'k', label='$variance$')
pl.xlabel('$t$ [s]') 
pl.ylabel('$Standard$ $Deviation$ [mm]')
# pl.legend()
pl.ylim(0.0052,0.0066)
pl.xlim(0.0,2)
pl.show()

#%% 

y=y2
u=u2

n_t = Np
x_test1 = np.zeros((n_t,n_u+n_y)) 
vari = np.zeros((n_t))
y_GP = np.zeros((n_t))
y_GP[0:n_maior] = y[0:n_maior] #REVER

for m in range(max(n_u,n_y), n_t):
    x_test1[m,0:n_u] = u[m - n_u + 1:m + 1,]
    x_test1[m,n_u:n_y + n_u] = y_GP[m - n_y:m]
    y_GP[m],vari[m] = gp_model_RBF.predict(x_test1[m,:].reshape(-1,n_u+n_y))

erro=np.zeros((n_t))
erro= y - y_GP

std_erro_treino = np.std(erro)/np.std(y_GP)

pl.figure()
pl.grid(b=None, which='major', axis='both')
# pl.fill_between(t,(y_max+mean_sig_output)*std_sig_output, (y_min+mean_sig_output)*std_sig_output, alpha=0.8, label='$Margin$')
pl.plot(t,1000*(y+mean_sig_output)*std_sig_output, 'r', label='$y_{original}$')
pl.plot(t,1000*(y_GP+mean_sig_output)*std_sig_output,'--b', label='$y_{GP}$')
pl.plot(t,1000*(erro+mean_sig_output)*std_sig_output,'--c', label='$error$')
pl.xlabel('$Time$ [s]') 
# pl.ylabel('$y(t)$ [mm]')
# pl.legend()
pl.xlim(0,2)
#pl.title('Treinamento de dados')
pl.show()

pl.figure()
pl.grid(b=None, which='major', axis='both')
# pl.plot(t,1000*(vari),'--c', label='$variance$')
pl.plot(t,(np.sqrt(vari))*std_sig_output,'k', label='$variance$')
pl.xlabel('$Tempo [s]$') 
pl.ylabel('$Vari \fhat ancia [mm]$')
# pl.legend()
pl.ylim(0.0053,0.0056)
pl.xlim(0.0,2)
pl.show()


#%% Fit 
ErrNum=0
for i in range(n_maior,n_t):
    ErrNum=ErrNum+(erro[i,])**2 
    ErrDem=0
for i in range(n_maior,n_t):
    ErrDem=ErrDem+(y_GP[i,])**2
        #calcular estimado e erro
Fit=100*(1-math.sqrt(ErrNum/ErrDem))

#%%
print ("Fit RBF")
print (Fit)

#%%
n_testes=200
n_entrad=2
n_condic=6
n_amostr=n_testes*n_entrad*n_condic

Ehist = np.zeros(n_amostr)
F2 =    np.zeros(n_amostr)
F  =    np.zeros(n_amostr)
std_err=np.zeros(n_amostr)
it =    np.zeros(n_amostr)
yhist = np.zeros((1,len(t)-max(n_u,n_y)))
y_ = np.zeros((1,len(t)-max(n_u,n_y)))
var = np.zeros((len(t)-max(n_u,n_y)))

Ya=np.zeros(n_amostr)

rms_cond_in =np.zeros(n_entrad*n_condic)
rms_cond_out=np.zeros(n_entrad*n_condic)

rms_sig_in = np.zeros(n_amostr)
rms_sig_out =np.zeros(n_amostr)

rms_chirp_in = np.zeros(n_testes*n_condic)
rms_chirp_out= np.zeros(n_testes*n_condic)
rms_rand_in  = np.zeros(n_testes*n_condic)
rms_rand_out = np.zeros(n_testes*n_condic)
Ya_chirp     = np.zeros(n_testes*n_condic)
Ya_rand      = np.zeros(n_testes*n_condic)

chirp=0
rand=0

cont=0
for n in range(0,n_amostr):
    if n < n_testes:
        u  = 0.1*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
        acc_rand,vel_rand,y = Linear_oscillator(ms,c,k1,k2,k3,u,t,x0,v0)
        
        sig=1
        
    elif n >= n_testes and n < 2*n_testes:
        u = 0.3*np.random.normal(0,1,size=(Np,))
        u = signal.sosfiltfilt(filt,u)
        
        # u = (rms_cond_in[0])*(u/np.sqrt(np.mean(u**2)))
        acc_rand,vel_rand,y = Linear_oscillator(ms,c,k1,k2,k3,u,t,x0,v0)
        
        sig=0
        
    elif n >= 2*n_testes and n < 3*n_testes:
        u  = 0.25*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
        acc_rand,vel_rand,y = Linear_oscillator(ms,c,k1,k2,k3,u,t,x0,v0)
        
        sig=1
        
    elif n >= 3*n_testes and n < 4*n_testes:
        u = 0.8*np.random.normal(0,1,size=(Np,))
        u = signal.sosfiltfilt(filt,u)
        
        # u = (rms_cond_in[2])*(u/np.sqrt(np.mean(u**2)))
        acc_rand,vel_rand,y = Linear_oscillator(ms,c,k1,k2,k3,u,t,x0,v0)
        
        sig=0
        
    elif n >= 4*n_testes and n < 5*n_testes:
        u  = 0.5*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
        acc_rand,vel_rand,y = Linear_oscillator(ms,c,k1,k2,k3,u,t,x0,v0)
        
        sig=1
        
    elif n >= 5*n_testes and n < 6*n_testes:
        u = 1.5*np.random.normal(0,1,size=(Np,))
        u = signal.sosfiltfilt(filt,u)
        
        # u = (rms_cond_in[4])*(u/np.sqrt(np.mean(u**2)))
        acc_rand,vel_rand,y = Linear_oscillator(ms,c,k1,k2,k3,u,t,x0,v0)
        
        sig=0
        
    elif n >= 6*n_testes and n < 7*n_testes:
        u  = 1*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
        acc_rand,vel_rand,y = Linear_oscillator(ms,c,k1,k2,k3,u,t,x0,v0)
        
        sig=1
        
    elif n >= 7*n_testes and n < 8*n_testes:
         u = 2*np.random.normal(0,1,size=(Np,))
         u = signal.sosfiltfilt(filt,u)
         
         # u = (rms_cond_in[6])*(u/np.sqrt(np.mean(u**2))) -A
         acc_rand,vel_rand,y = Linear_oscillator(ms,c,k1,k2,k3,u,t,x0,v0)
         
         sig=0
         
    elif n >= 8*n_testes and n < 9*n_testes:
         u  = 1.2*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
         acc_rand,vel_rand,y = Linear_oscillator(ms,c,k1,k2,k3,u,t,x0,v0)
         
         sig=1
    
    elif n >= 9*n_testes and n < 10*n_testes:
         u = 3*np.random.normal(0,1,size=(Np,))
         u = signal.sosfiltfilt(filt,u)
         
         # u = (rms_cond_in[8])*(u/np.sqrt(np.mean(u**2)))
         acc_rand,vel_rand,y = Linear_oscillator(ms,c,k1,k2,k3,u,t,x0,v0)
         
         sig=0
         
    elif n >= 10*n_testes and n < 11*n_testes:
         u  = 1.4*signal.chirp(t, wmin , max(t), wmax, method='linear', phi=0)
         acc_rand,vel_rand,y = Linear_oscillator(ms,c,k1,k2,k3,u,t,x0,v0)
         
         sig=1
             
    else:
        u = 4.2*np.random.normal(0,1,size=(Np,))
        u = signal.sosfiltfilt(filt,u)
        
        # u = (rms_cond_in[10])*(u/np.sqrt(np.mean(u**2)))
        acc_rand,vel_rand,y = Linear_oscillator(ms,c,k1,k2,k3,u,t,x0,v0)
        
        sig=0
        
    rms_sig_in[n] = np.sqrt(np.mean(u**2))
    rms_sig_out[n] = np.sqrt(np.mean(y**2))
    
    y = y + np.random.normal(0, 0.05*std_sig_output, size=len(t))
    u = (u - mean_sig_input)/std_sig_input
    y = (y - mean_sig_output)/std_sig_output
    
    n_t = Np
    x_test_RBF = np.zeros((n_t,n_u+n_y)) 
    vari_RBF = np.zeros((n_t))
    y_GP_RBF = np.zeros((n_t))
    y_GP_RBF[0:n_maior] = y[0:n_maior]

    for d in range(max(n_u,n_y), n_t):
        x_test_RBF[d,0:n_u] = u[d - n_u + 1:d + 1,]
        x_test_RBF[d,n_u:n_y + n_u] = y_GP_RBF[d - n_y:d]
        y_GP_RBF[d],vari_RBF[d] = gp_model_RBF.predict(x_test_RBF[d,:].reshape(-1,n_u+n_y))

    yhist[0,:]=y_GP_RBF[max(n_u,n_y)::]
    y_[0,:]=y[max(n_u,n_y)::]
    var=vari_RBF[max(n_u,n_y)::]

    erro_RBF=np.zeros((n_t))
    erro_RBF= y - y_GP_RBF
    
    std_erro_predito= np.std(erro_RBF)/np.std(y_GP_RBF)
    
    std_err[n,]=std_erro_predito
    F[n,]=(std_erro_predito)/(std_erro_treino)
    F2[n,]=(std_erro_predito**2)/(std_erro_treino**2)
    
    
    ErrNum=0
    for i in range(n_maior,n_t):
        ErrNum=ErrNum+(erro_RBF[i,])**2
        ErrDem=0
    for i in range(n_maior,n_t):
        ErrDem=ErrDem+(y_GP_RBF[i,])**2
            #calcular estimado e erro
    Ehist[n,]=100*(1-math.sqrt(ErrNum/ErrDem))
    
    it[n,] = n

    Ya[n] = spatial.distance.cdist(y_,yhist, 'seuclidean', V=var)/np.std(y)
    
    if sig==1:
        rms_chirp_in[chirp] = rms_sig_in[n]
        rms_chirp_out[chirp]= rms_sig_out[n]
        Ya_chirp[chirp]     = Ya[n]
        chirp=chirp+1
    else:
        rms_rand_in[rand]  = rms_sig_in[n]
        rms_rand_out[rand] = rms_sig_out[n]
        Ya_rand[rand]      = Ya[n]
        rand=rand+1
    
    
    print (n)
    print (Ehist[n,])
    print ("\n")
#%% 
rms_cond_in[0] =np.mean(rms_sig_in[0*n_testes:((1*n_testes)-1)])
rms_cond_in[1] =np.mean(rms_sig_in[1*n_testes:((2*n_testes)-1)])
rms_cond_in[2] =np.mean(rms_sig_in[2*n_testes:((3*n_testes)-1)])
rms_cond_in[3] =np.mean(rms_sig_in[3*n_testes:((4*n_testes)-1)])
rms_cond_in[4] =np.mean(rms_sig_in[4*n_testes:((5*n_testes)-1)])
rms_cond_in[5] =np.mean(rms_sig_in[5*n_testes:((6*n_testes)-1)])
rms_cond_in[6] =np.mean(rms_sig_in[6*n_testes:((7*n_testes)-1)])
rms_cond_in[7] =np.mean(rms_sig_in[7*n_testes:((8*n_testes)-1)])
rms_cond_in[8] =np.mean(rms_sig_in[8*n_testes:((9*n_testes)-1)])
rms_cond_in[9] =np.mean(rms_sig_in[9*n_testes:((10*n_testes)-1)])
rms_cond_in[10]=np.mean(rms_sig_in[10*n_testes:((11*n_testes)-1)])
rms_cond_in[11]=np.mean(rms_sig_in[11*n_testes:((12*n_testes)-1)])

#%%    
# pl.figure()
# pl.grid(b=None, which='major', axis='both')
# pl.plot(it,Ehist, 'o', label='y')
# pl.xlabel('$Amostra$') 
# pl.ylabel('$fit$')
# pl.title('teste GP')
# pl.show()

#%% Ponto médio de cada intervalo

média_rand=np.zeros(n_condic)
média_rand[0]=np.mean(Ya[n_testes:((2*n_testes)-1)])
média_rand[1]=np.mean(Ya[(3*n_testes):((4*n_testes)-1)])
média_rand[2]=np.mean(Ya[(5*n_testes):((6*n_testes)-1)])
média_rand[3]=np.mean(Ya[(7*n_testes):((8*n_testes)-1)])
média_rand[4]=np.mean(Ya[(9*n_testes):((10*n_testes)-1)])
média_rand[5]=np.mean(Ya[(11*n_testes):((12*n_testes)-1)])

pl.figure()
pl.grid(b=None, which='major', axis='both')
# Separar intervalos de dano
for b in range(0,(n_entrad*n_condic)+1):
    pl.axvline(b*n_testes, 0, 100, color='black', linestyle='dashed')

# Intervalos de entrada aleatória
pl.plot(it[0:((1*n_testes)-1)],Ya[n_testes:((2*n_testes)-1)]        , 'x', color='blue')
pl.plot(it[(1*n_testes):((2*n_testes)-1)],Ya[(3*n_testes):((4*n_testes)-1)], 'x', color='red')
pl.plot(it[(2*n_testes):((3*n_testes)-1)],Ya[(5*n_testes):((6*n_testes)-1)], 'x', color='red')
pl.plot(it[(3*n_testes):((4*n_testes)-1)],Ya[(7*n_testes):((8*n_testes)-1)], 'x', color='blue')
pl.plot(it[(4*n_testes):((5*n_testes)-1)],Ya[(9*n_testes):((10*n_testes)-1)], 'x', color='red')
pl.plot(it[(5*n_testes):((6*n_testes)-1)],Ya[(11*n_testes):((12*n_testes)-1)], 'x', color='red')

# Pontos de média
pl.plot((1*n_testes)-(n_testes/2),média_rand[0],  'o', color='black')
pl.plot((2*n_testes)-(n_testes/2),média_rand[1],  'o', color='black')
pl.plot((3*n_testes)-(n_testes/2),média_rand[2],  'o', color='black')
pl.plot((4*n_testes)-(n_testes/2),média_rand[3],  'o', color='black')
pl.plot((5*n_testes)-(n_testes/2),média_rand[4], 'o', color='black')
pl.plot((6*n_testes)-(n_testes/2),média_rand[5], 'o', color='black')

# Suprimir eixo x
ax = pl.gca()
ax.axes.xaxis.set_visible(False)
# ax.set_facecolor('white')

# Legendas eixo x
pl.text((1*n_testes)-(n_testes/2),-4, str(f'{1000*rms_cond_in[1]:.0f}')+ ' [mN]',horizontalalignment='center')
pl.text((2*n_testes)-(n_testes/2),-4, str(f'{1000*rms_cond_in[3]:.0f}')+ ' [mN]',horizontalalignment='center')
pl.text((3*n_testes)-(n_testes/2),-4, str(f'{1000*rms_cond_in[5]:.0f}')+ ' [mN]',horizontalalignment='center')
pl.text((4*n_testes)-(n_testes/2),-4, str(f'{1000*rms_cond_in[7]:.0f}')+ ' [mN]',horizontalalignment='center')
pl.text((5*n_testes)-(n_testes/2),-4, str(f'{1000*rms_cond_in[9]:.0f}')+ ' [mN]',horizontalalignment='center')
pl.text((6*n_testes)-(n_testes/2),-4, str(f'{1000*rms_cond_in[11]:.0f}')+' [mN]',horizontalalignment='center')

pl.xlim(0,6*n_testes)
# pl.xlim(0,70)
pl.ylabel('$D(y_{\mbox{\scriptsize{exp}}},\mu_{y})$')
# pl.xlabel('Valor Eficaz da Entrada [N]')
pl.ylim(0,70)
# pl.title('Distribuição com a entrada Aleatória')
pl.show()

#%% std/std - chirp
média_chirp=np.zeros(n_condic)
média_chirp[0]=np.mean(Ya[0:((n_testes)-1)])
média_chirp[1]=np.mean(Ya[(2*n_testes):((3*n_testes)-1)])
média_chirp[2]=np.mean(Ya[(4*n_testes):((5*n_testes)-1)])
média_chirp[3]=np.mean(Ya[(6*n_testes):((7*n_testes)-1)])
média_chirp[4]=np.mean(Ya[(8*n_testes):((9*n_testes)-1)])
média_chirp[5]=np.mean(Ya[(10*n_testes):((11*n_testes)-1)])

pl.figure()
pl.grid(b=None, which='major', axis='both')
for b in range(0,(n_entrad*n_condic)+1):
    pl.axvline(b*n_testes, 0, 100, color='black', linestyle='dashed')

pl.plot(it[0:((n_testes)-1)],Ya[0:((n_testes)-1)]                          ,    'x', color='blue')
pl.plot(it[(n_testes):((2*n_testes)-1)],Ya[(2*n_testes):((3*n_testes)-1)],    'x', color='red' )
pl.plot(it[(2*n_testes):((3*n_testes)-1)],Ya[(4*n_testes):((5*n_testes)-1)],    'x', color='red' )
pl.plot(it[(3*n_testes):((4*n_testes)-1)],Ya[(6*n_testes):((7*n_testes)-1)],    'x', color='blue')
pl.plot(it[(4*n_testes):((5*n_testes)-1)],Ya[(8*n_testes):((9*n_testes)-1)],    'x', color='red' )
pl.plot(it[(5*n_testes):((6*n_testes)-1)],Ya[(10*n_testes):((11*n_testes)-1)],'x', color='red' )

pl.plot((n_testes)-(n_testes/2),média_chirp[0],   'o', color='black')
pl.plot((2*n_testes)-(n_testes/2),média_chirp[1], 'o', color='black')
pl.plot((3*n_testes)-(n_testes/2),média_chirp[2], 'o', color='black')
pl.plot((4*n_testes)-(n_testes/2),média_chirp[3], 'o', color='black')
pl.plot((5*n_testes)-(n_testes/2),média_chirp[4], 'o', color='black')
pl.plot((6*n_testes)-(n_testes/2),média_chirp[5],'o', color='black')

ax = pl.gca()
ax.axes.xaxis.set_visible(False)
# ax.set_facecolor('white')

pl.text((1*n_testes)-(n_testes/2),-4, str(f'{1000*rms_cond_in[0]:.0f}')+ ' $[mN]$', horizontalalignment='center')
pl.text((2*n_testes)-(n_testes/2),-4, str(f'{1000*rms_cond_in[2]:.0f}')+ ' $[mN]$', horizontalalignment='center')
pl.text((3*n_testes)-(n_testes/2),-4, str(f'{1000*rms_cond_in[4]:.0f}')+ ' $[mN]$', horizontalalignment='center')
pl.text((4*n_testes)-(n_testes/2),-4, str(f'{1000*rms_cond_in[6]:.0f}')+ ' $[mN]$', horizontalalignment='center')
pl.text((5*n_testes)-(n_testes/2),-4, str(f'{1000*rms_cond_in[8]:.0f}')+ ' $[mN]$', horizontalalignment='center')
pl.text((6*n_testes)-(n_testes/2),-4, str(f'{1000*rms_cond_in[10]:.0f}')+' $[mN]$', horizontalalignment='center')

pl.xlim(0,6*n_testes)
# pl.xlim(0,70)
pl.ylabel('$D(y_{\mbox{\scriptsize{exp}}},\mu_{y})$')
pl.ylim(0,70)
# pl.title('Distribuição com a entrada Chirp')
pl.show()
