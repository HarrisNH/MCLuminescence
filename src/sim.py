from omegaconf import DictConfig, OmegaConf
import numpy as np
import hydra
from paths import PROJECT_ROOT,CONFIG_DIR

def experiment_setup(cfg: DictConfig):
    mc = cfg.exp_type
    phys = cfg.physics
    setup = cfg.setup
    temp_0 = mc.T_start+272.15 #convert to kelvin
    if setup.A_self_calc == 1: #calculate A0 from the given values or overwrite with the given value
        A0 = mc.s_tun*np.exp(-phys.E/(phys.k_b*temp_0))
    else: A0 = mc.A
    
    if mc.name == "isothermal":
        timesteps = int(mc.duration/mc.dt)
    elif mc.name == "non_isothermal":
        duration = (mc.T_end-mc.T_start)/np.array(mc.T_rate)
        timesteps = (duration/mc.dt).astype(int)
    rs = np.linspace(mc.dr_prime,mc.distance,int(mc.distance/mc.dr_prime)) # array with distances
    n_sims = mc.sims
    f= 3*(rs)**2*np.exp(-(rs)**3)

    count = np.zeros((timesteps, len(rs),n_sims)) #timesteps, distance, sim

    count_start = f*mc.N0*mc.dr_prime #initial number of electrons at each distance
    count_start_nsims = np.tile(count_start.reshape(-1, 1), (1, n_sims)) #make it compatible with number of sims
    count[0, :,:] = np.round(count_start_nsims).astype(int) #initialize number of electrons for each distance
    Lum = np.zeros((timesteps,n_sims))

    P_rate = (A0*mc.s_tun)/(mc.B*np.exp((mc.rho_prime)**(-1/3)*rs))
    P_rate_nsim = np.tile(P_rate.reshape(-1, 1), (1, n_sims)) #make it compatible with number of sims
    
    print(f"Running {n_sims} simultaneous runs")
    return Lum, count, P_rate_nsim,rs

updater = 5000 #print update every x steps
@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def TL_non_iso(cfg: DictConfig):
    mc = cfg.exp_type
    phys = cfg.physics
    Lum,count,_,rs = experiment_setup(cfg)

    duration = (mc.T_end-mc.T_start)/mc.T_rate
    loop_step = int(duration/mc.dt)
    steps = np.linspace(mc.T_start+272.15,mc.T_end+272.15,loop_step-1) 

    #Precalculate rate of tunneling
    A = mc.s_tun*np.exp(-phys.E/(phys.k_b*steps))
    frac_s_B = (mc.s_tun)/(mc.B*np.exp((mc.rho_prime)**(-1/3)*rs))
    P_rate = A.reshape(-1,1)*frac_s_B
    P_rate_nsim = np.tile(P_rate.reshape(-1, len(rs),1), (1, 1,mc.sims)) #make it compatible with number of sims
    
    #rate times time delta 
    P = P_rate_nsim*mc.dt
    
    for i,temp in enumerate(steps):            
            P[i][count[i]==0] = 0
            delta_n = -np.random.binomial(n=count[i].astype(int), p=P[i])
            Lum[i] = -np.sum(delta_n, axis=0)/mc.dt
            count[i+1] = count[i] + delta_n
            if i%updater==0:
                print("Loop: ",i, "of total steps: ",loop_step, f"\n with total lum in last {updater} steps: ",sum(Lum[i-updater:i])) #fix this
    return Lum, count,steps



def TL_iso(cfg : DictConfig) -> None:
    mc = cfg.exp_type
    Lum,count,P_rate_nsim,_ = experiment_setup(cfg)
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


if __name__ == "__main__":
    Lum, count = TL_non_iso()
    print(Lum,count)