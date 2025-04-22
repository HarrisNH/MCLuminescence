from paths import PROJECT_ROOT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def extract_data_TL(sample:str = "CLBR"):
    """
    Extract data from excel sheet with experimental data
    """
    data = pd.read_csv(f"{PROJECT_ROOT}/data/raw/{sample}_IRSL50_0.25KperGy.csv")
    
    data["Duration"] = data["RampDuration_s"]
    data["T_start"] = data["Temp_start"]
    data["T_end"] = data["Temp_end"]
    data["Fill"] = data["n/N_calc_from_DRCsatlevel"]
    data["e_ratio"] = 0
    data.drop_duplicates(subset=["T_end"],keep='first',inplace=True)
    data[["Duration", "T_start", "T_end", "Fill", "e_ratio"]].to_csv(f"{PROJECT_ROOT}/data/processed/{sample}_IRSL50_0.25KperGy.csv", index=False)




def extract_data_TL_iso(L0):
    data = pd.read_csv(f"{PROJECT_ROOT}/data/raw/CLBR_IR50 ISO.csv")
    df = pd.DataFrame(columns=["exp_no","e_ratio","temp","dose","time","L"])
    for i in np.arange(3,len(data),2):
        temp = data.iloc[i,1]
        dose  = data.iloc[i+1,1]
        time = (data.iloc[i,3:].values.astype(float))*1000
        L =  (data.iloc[i+1,3:].values).astype(float)
        e_ratio = L/L0

        data_temp = pd.DataFrame({"exp_no": i//2-1,"e_ratio": e_ratio, "temp":temp, "dose":dose, "time":time, "L":L})
        df = pd.concat([df,data_temp],ignore_index=True)
    df = df[pd.to_numeric(df['time'],errors='coerce').notna()]
    df.sort_values(by=["exp_no", "time"], ascending=[True, True], inplace=True)
    df.drop_duplicates(subset=["exp_no", "time"], keep="first", inplace=True)
    df.to_csv(f"{PROJECT_ROOT}/data/processed/CLBR_IR50_ISO.csv", index=False)



def plot_iso_data():
    df = pd.read_csv(f"{PROJECT_ROOT}/data/processed/CLBR_IR50_ISO_plot.csv")  # if you need to load it
    
    # Make sure the relevant columns are numeric
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['L']    = pd.to_numeric(df['L'],    errors='coerce')
    df['L'] = df['L']/1.52
    #df = df[df["exp_no"]!=0]
    # Start the figure
    plt.figure(figsize=(8,6))

    # Loop over each temperature group
    for temp, group in df.groupby('temp'):
        plt.plot(group['time'], group['L'], marker='o', linestyle='-',
                label=f'{temp}')

    # Log‐scale the x‐axis
    plt.xscale('log')
    plt.xlim(0,1e4)
    # Labels, title, grid
    plt.xlabel('Time (s)')
    plt.ylabel('Electron trap ratio')
    plt.title('Filled electron trap ratio at different temperatures')

    # Legend with a title
    plt.legend(title='Temperature °C', loc='upper right', fontsize='small', title_fontsize='medium')

    plt.tight_layout()
    plt.savefig(f"{PROJECT_ROOT}/results/plots/CLBR_IR50_ISO_plot.png")
    plt.show()

def plot_TL_data():
    df = pd.read_csv(f"{PROJECT_ROOT}/data/processed/CLBR_IRSL_0.25KperGY_plot.csv")  # if you need to load it
    
    df['duration'] = (300 - df['T_end'])/25*1086

    # conversion functions
    def dur_to_T(x):
        return 300-25/1086*x

    def T_to_dur(x):
        return (300 - x)/25*1086

    fig, ax = plt.subplots(figsize=(8,5))

    # plot Fill vs duration
    ax.plot(df['duration'], df['Fill'], 'o')
    ax.set_xlabel('Duration (s)')
    ax.set_ylabel('Filled electron trap ratio')
    ax.grid(True, which='both', ls='--', lw=0.5)

    # add secondary x‑axis on top
    secax = ax.secondary_xaxis('top', functions=(dur_to_T, T_to_dur))
    secax.set_xlabel('T_end (°C)')
    ax.title.set_text('TL exp. with -0.25K per Gy')
    plt.tight_layout()
    plt.savefig(f"{PROJECT_ROOT}/results/plots/TL_exp_0.25K_per_Gy.png")
    plt.show()

plot_iso_data()
extract_data_TL("CLBR")
plot_TL_data()

extract_data_TL_iso(L0=1.52)
