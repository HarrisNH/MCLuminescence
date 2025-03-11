import numpy as np
import hydra
from paths import PROJECT_ROOT,CONFIG_DIR
from omegaconf import DictConfig, OmegaConf,ListConfig


def experiment_setup_3D(cfg: DictConfig):
    mc = cfg.exp_type
    phys = cfg.physics
    setup = cfg.setup
    temp_0 = mc.T_start+272.15 #convert to kelvin
    if setup.A_self_calc == 1: #calculate A0 from the given values or overwrite with the given value
        A0 = phys.s_tun*np.exp(-phys.E/(phys.k_b*temp_0))
    else: A0 = phys.A

    #get timesteps,distance array, number of simulations and density function
    timesteps = int(mc.duration/mc.dt)
    rs = np.linspace(mc.dr_prime,mc.distance,int(mc.distance/mc.dr_prime))
    n_sims = mc.sims
    f= 3*(rs)**2*np.exp(-(rs)**3)

    #initialize Lum and count array and populate first row with initial number of electrons at each dist
    Lum = np.zeros((timesteps,n_sims))
    count = np.zeros((timesteps, len(rs),n_sims)) #timesteps, distance, sim
    count_start = f*mc.N0*mc.dr_prime 
    count_start_nsims = np.tile(count_start.reshape(-1, 1), (1, n_sims,))
    count[0] = np.round(count_start_nsims).astype(int) 

    #Calculate tunneling rate for distances
    P_rate = (A0*phys.s_tun)/(phys.B*np.exp((phys.rho_prime)**(-1/3)*rs))
    P_rate_nsim = np.tile(P_rate.reshape(-1, 1), (1, n_sims)) 
    
    print(f"Running {n_sims} simultaneous runs")
    return Lum, count, P_rate_nsim


def experiment_setup_4D(cfg: DictConfig):
    #get config values
    mc = cfg.exp_type
    phys = cfg.physics
    setup = cfg.setup
    
    #get timesteps,distance array, number of simulations and density function
    duration = (mc.T_end-mc.T_start)/np.array(mc.T_rate)
    timesteps = (duration/mc.dt).astype(int)
    rs = np.linspace(mc.dr_prime,mc.distance,int(mc.distance/mc.dr_prime)) # array with distances
    n_sims = mc.sims
    f= 3*(rs)**2*np.exp(-(rs)**3)

    #initialize Lum and count array and populate first row with initial number of electrons at each dist
    count = np.zeros((np.max(timesteps)+1, len(rs),n_sims,len(timesteps))) #timesteps, distance, sim,temp_rate
    count_start = f*mc.N0*mc.dr_prime 
    count_start_nsims = np.tile(count_start.reshape(-1, 1,1), (1, n_sims,len(timesteps)))
    count[0] = np.round(count_start_nsims).astype(int) 
    Lum = np.zeros((np.max(timesteps),n_sims,len(timesteps)))

    print(f"Running {n_sims} simultaneous runs")
    return Lum, count, rs

def TL_non_iso(cfg: DictConfig):
    mc = cfg.exp_type
    phys = cfg.physics
    Lum,count,rs = experiment_setup_4D(cfg)
    
    duration = (mc.T_end-mc.T_start)/np.array(mc.T_rate)
    n_temperatures = (duration/mc.dt).astype(int)
    steps_lists = [np.linspace(mc.T_start+273.15,mc.T_end+273.15,n)  for n in n_temperatures]
    steps_matrix = np.full((np.max(n_temperatures),len(n_temperatures)),1e-14) #pad with values making probability zero

    for i,steps in enumerate(steps_lists):
        steps_matrix[:len(steps),i] = steps
    #Precalculate rate of tunneling
    A = phys.s_tun*np.exp(-phys.E/(phys.k_b*steps_matrix))
    frac_s_B = (phys.s_tun)/(phys.B*np.exp((phys.rho_prime)**(-1/3)*rs))
    P_rate = A.reshape(-1,1,len(n_temperatures))*frac_s_B.reshape(1,len(rs),1)
    P_rate_nsim = np.tile(P_rate.reshape(-1, len(rs),1,len(n_temperatures)), (1, 1,mc.sims,1)) #make it compatible with number of sims
    
    #rate times time delta 
    P = P_rate_nsim*mc.dt
    time_step_size = (mc.dt*np.array(mc.T_rate)) #
    updater = 5000 #print update every x steps
    for i,temp in enumerate(steps_matrix):            
            P[i][count[i]==0] = 0
            #P[i][P[i]>1] = 1 #Dangerous to set to 1, but it is a workaround for the problem of having a rate higher than 1
            delta_n = -np.random.binomial(n=count[i].astype(int), p=P[i])
            Lum[i] = -np.sum(delta_n, axis=0)/time_step_size
            count[i+1] = count[i] + delta_n
            if i%updater==0:
                print("Loop: ",i, "of total steps: ",np.max(n_temperatures), f"\n with total lum in last {updater} steps: ",sum(Lum[i-updater:i])) #fix this
    return Lum, count,steps_matrix

def TL_rhoEs(cfg: DictConfig,rho_prime: float,E: float,s_tun: float) -> None:
    mc = cfg.exp_type
    phys = cfg.physics
    Lum,count,rs = experiment_setup_4D(cfg)
    
    duration = (mc.T_end-mc.T_start)/np.array(mc.T_rate)
    n_temperatures = (duration/mc.dt).astype(int)
    steps_lists = [np.linspace(mc.T_start+272.15,mc.T_end+272.15,n)  for n in n_temperatures]
    steps_matrix = np.full((np.max(n_temperatures),len(n_temperatures)),1e-14) #pad with values making probability zero

    for i,steps in enumerate(steps_lists):
        steps_matrix[:len(steps),i] = steps
    #Precalculate rate of tunneling
    A = s_tun*np.exp(-E/(phys.k_b*steps_matrix))
    frac_s_B = (s_tun)/(phys.B*np.exp((rho_prime)**(-1/3)*rs))
    P_rate = A.reshape(-1,1,len(n_temperatures))*frac_s_B.reshape(1,len(rs),1)
    P_rate_nsim = np.tile(P_rate.reshape(-1, len(rs),1,len(n_temperatures)), (1, 1,mc.sims,1)) #make it compatible with number of sims
    
    #rate times time delta 
    P = P_rate_nsim*mc.dt
    time_step_size = (mc.dt*np.array(mc.T_rate)) #
    updater = 5000 #print update every x steps
    for i,temp in enumerate(steps_matrix):            
            P[i][count[i]==0] = 0
            P[i][P[i]>1] = 1 #Dangerous to set to 1, but it is a workaround for the problem of having a rate higher than 1
            delta_n = -np.random.binomial(n=count[i].astype(int), p=P[i])
            Lum[i] = -np.sum(delta_n, axis=0)/time_step_size
            count[i+1] = count[i] + delta_n
            if i%updater==0:
                print("Loop: ",i, "of total steps: ",np.max(n_temperatures), f"\n with total lum in last {updater} steps: ",sum(Lum[i-updater:i])) #fix this
    return Lum, count,steps_matrix

def TL_iso(cfg : DictConfig) -> None:
    mc = cfg.exp_type
    Lum,count,P_rate_nsim = experiment_setup_3D(cfg)
    i=0
    for t in np.arange(0,mc.duration-mc.dt,mc.dt):
        P = P_rate_nsim*mc.dt #rate times time delta
        delta_n = -np.random.binomial(n=count[i].astype(int), p=P)
        Lum[i] = -np.sum(delta_n, axis=0)/mc.dt
        count[i+1] = count[i] + delta_n
        if t%5==0:
            print("Time: ",t, "of total duration: ",mc.duration, "with lum: ",Lum[i-1])
        i+=1
    return Lum, count

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg : DictConfig):
    lums, counts = TL_non_iso(cfg)
    print("done")

if __name__ == "__main__":
    main()
