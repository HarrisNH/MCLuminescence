from paths import PROJECT_ROOT
import pandas as pd
import numpy as np

def extract_data_TL():
    """
    Extract data from excel sheet with experimental data
    """
    data = pd.read_csv(f"{PROJECT_ROOT}/data/raw/CLBR_IRSL50_0.25KperGy.csv")
    
    data["Duration"] = data["RampDuration_s"]
    data["T_start"] = data["Temp_start"]
    data["T_end"] = data["Temp_end"]
    data["Fill"] = data["n/N_calc_from_DRCsatlevel"]
    data[[ "Duration", "T_start", "T_end","Fill",]].to_csv(f"{PROJECT_ROOT}/data/processed/CLBR_IRSL50_0.25KperGy.csv", index=False)


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


extract_data_TL_iso(L0=1.52)
extract_data_TL()