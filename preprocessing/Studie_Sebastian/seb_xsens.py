from preprocessing import xsens_data
import numpy as np
import pandas as pd



def main(folder_path_Xsens):
    xs_data, xs_frames, header = xsens_data.main(folder_path_Xsens)
    stacked = []
    for i in range(len(xs_frames)):
        stacked.append(xs_data[i])
    stacked= np.vstack(stacked)
    df=pd.DataFrame(stacked)
    #df.to_csv(output_name, index=False, header=header)

    return df, xs_frames, header

if __name__ == '__main__':
    main()