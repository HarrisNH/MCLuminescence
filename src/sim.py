from omegaconf import DictConfig, OmegaConf,ListConfig
import numpy as np
import hydra
from paths import PROJECT_ROOT,CONFIG_DIR


def config_setup(cfg: DictConfig):
    configs = {
        "mc": cfg.exp_type,
        "phys": cfg.physics,
        "setup": cfg.setup
    }
    # Get maximum length of any list in all dictionaries
    max_length = max(
            len(v)
            for sub_dict in configs.values()
            for v in sub_dict.values()
            if isinstance(v, ListConfig)
        )
        # Extend all lists to have the same length
    for sub_name, sub_dict in configs.items():
        for k, v in sub_dict.items():
            if isinstance(v, ListConfig):
                v = list(v)
                if max_length % len(v) != 0:
                    raise ValueError(
                        f"max_length ({max_length}) is not divisible "
                        f"by len({sub_name}.{k}) ({len(v)})"
                    )
                sub_dict[k] = v * (max_length // len(v))

    return configs


def experiment_setup(i,configs):
    mc = configs["mc"]
    phys = configs["phys"]
    print(mc.loop_type)
    if mc.loop_type == "temp" and (mc.T_rate[i]) != 0:
        duration = (mc.T_end[i]-mc.T_start[i])/(mc.T_rate[i])
    elif mc.loop_type in ["time","optic"]:
        duration = mc.duration

    timesteps = int(duration/mc.dt)
    rs = np.linspace(mc.dr_prime,mc.distance,int(mc.distance/mc.dr_prime)) # array with distances
    n_sims = mc.sims
    f= 3*(rs)**2*np.exp(-(rs)**3)
    #initialize Lum and count array and populate first row with initial number of electrons at each dist
    count = np.zeros((timesteps+1, len(rs),n_sims)) #timesteps, distance, sim
    count_start = f*mc.N0*mc.dr_prime 
    count_start_nsims = np.tile(count_start.reshape(-1, 1), (1, n_sims))
    count[0] = np.round(count_start_nsims).astype(int) 
    Lum = np.zeros((timesteps,n_sims))

    if mc.loop_type == "temp":
        kel = 273.15 #celsius to kelvin
            #Precalculate rate of tunneling
        temps = np.linspace(mc.T_start[i]+kel,mc.T_end[i]+kel,int(duration/mc.dt))
        A = phys.s_tun[i]*np.exp(-phys.E[i]/(phys.k_b[i]*temps))
        frac_s_B = (phys.s_tun[i])/(phys.B[i]*np.exp((phys.rho_prime[i])**(-1/3)*rs))
        P_rate = A.reshape(-1,1)*frac_s_B.reshape(1,len(rs))
        P_rate_nsim = np.tile(P_rate.reshape(-1, len(rs),1), (1, 1,mc.sims)) #make it compatible with number of sims
        #rate times time delta 
        P = P_rate_nsim*mc.dt
    elif mc.loop_type == "time":
        if phys.A_self_calc[i] == 1: #calculate A0 from the given values or overwrite with the given value
            A0 = phys.s_tun[i]*np.exp(-phys.E[i]/(phys.k_b[i]*(mc.T_start[i]+kel)))
        else: A0 = phys.A[i]
        P_rate = (A0*phys.s_tun[i])/(phys.B[i]*np.exp((phys.rho_prime[i])**(-1/3)*rs))
        P_rate_nsim = np.tile(P_rate.reshape(-1, 1), (1, n_sims)) 
        P = P_rate_nsim*mc.dt
    elif mc.loop_type == "optic":
        if phys.A_self_calc[i] == 1: #calculate A0 from the given values or overwrite with the given value
            A0 = 1 #this is not correct
        else: A0 = phys.A[i]
        P_rate = (A0*phys.s_tun[i])/(phys.B[i]*np.exp((phys.rho_prime[i])**(-1/3)*rs))
        P_rate_nsim = np.tile(P_rate.reshape(-1, 1), (1, n_sims)) 
        P = P_rate_nsim*mc.dt
    return count,Lum, P


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def general_run(cfg):
    '''
    Inputs: all configurations in a dict with three dicts inside
    Adjustable parameters are in arrays with len=number of runs
    '''
   
    configs = config_setup(cfg)
    mc = configs["mc"]
    phys = configs["phys"]
    setup = configs["setup"]

    Lums = []
    counts = []

    for i in range(len(mc["T_end"])):
        count, Lum,P = experiment_setup(i,configs)
        if mc.loop_type == "temp":
            kel = 273.15 #celsius to kelvin
            updater = int(P.shape[0]/5)
            print(f"Simulating with temp increase of {mc.T_rate[i]} per second and step size of {mc.dt} seconds")
            timestep_size = mc.T_rate[i]*mc.dt
            for j,temp in enumerate(np.linspace(mc.T_start[i]+kel,mc.T_end[i]+kel,int(P.shape[0]))):
                P[j][count[j]==0] = 0
                #P[j][P[j]>1] = 1 #Dangerous to set to 1, but it is a workaround for the problem of having a rate higher than 1
                delta_n = -np.random.binomial(n=count[j].astype(int), p=P[j])
                Lum[j] = -np.sum(delta_n, axis=0)/timestep_size
                count[j+1] = count[j] + delta_n
                if j%updater==0:
                    if j!=0:
                        print("Loop: ",j, "of total steps: ",P.shape[0], f"\n with total lum in last {updater} steps: ",sum(Lum[j-updater:j])) #fix this
            Lums.append(Lum)
            counts.append(count)
        elif mc.loop_type == "time":
            for j in range(len(count)-1):
                P[j][count[j]==0] = 0
                delta_n = -np.random.binomial(n=count[j].astype(int), p=P[j])
                Lum[j] = -np.sum(delta_n, axis=0)/mc.dt
                count[j+1] = count[j] + delta_n
            Lums.append(Lum)
            counts.append(count)
        elif mc.loop_type == "optic":
            for j in range(len(count)-1):
                P[count[j]==0] = 0
                delta_n = -np.random.binomial(n=count[j].astype(int), p=P)
                Lum[j] = -np.sum(delta_n, axis=0)/mc.dt
                count[j+1] = count[j] + delta_n
            Lums.append(Lum)
            counts.append(count)
    return Lums, counts,configs

if __name__ == "__main__":
    general_run()
    print("Simulation completed")