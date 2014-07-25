import myio
import pandas as pd
import numpy as np

df = myio.read_csv(True)
Y = np.genfromtxt('pred.dat')
Y[Y < 0] = 0
#Y = np.rint(Y)

df_v = pd.DataFrame(index=df.index, data=Y[:, 0])
df_t = pd.DataFrame(index=df.index, data=Y[:, 1])
df_f = pd.DataFrame(index=df.index, data=Y[:, 2])
df_a = pd.DataFrame(index=df.index, data=Y)

#df_v.to_csv('/home/vod/flaviov/public_html/vis.csv', header=False)
#df_t.to_csv('/home/vod/flaviov/public_html/twi.csv', header=False)
#df_f.to_csv('/home/vod/flaviov/public_html/fac.csv', header=False)
df_a.to_csv('/home/vod/flaviov/public_html/all.csv', header=False)
