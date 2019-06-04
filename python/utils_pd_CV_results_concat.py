import pandas as pd

afiles = ['20161006_044510_all_p_res_SLCV_NB_[2]k.res',
        '20161006_071527_all_p_res_SLCV_NB_[5]k.res',
        '20161006_073550_all_p_res_SLCV_NB_[10]k.res'
]

print afiles

df_all_folds = pd.DataFrame()
for i, sname in enumerate(afiles):

    if i == 0:
        df = pd.read_pickle(sname)
    else:
        dft = pd.read_pickle(sname)
        df = df.append(dft)
        # print df.info()
        # df.reset_index(inplace=True)

print df


df.to_pickle('20161006_044510_all_p_res_SLCV_NB_[2,5,10]k.res')

