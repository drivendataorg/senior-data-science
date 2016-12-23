# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 08:18:00 2016

"""

import pandas as pd

s = pd.read_csv("../sub/xgb_esb_v7.tst.csv")
s1 = pd.read_csv("../sub/xgb_esb_v5.tst.csv")
s2 = pd.read_csv("../sub/xgb_esb_v4.tst.csv")
s3 = pd.read_csv("../sub/xgb_esb_v3.tst.csv")

s13 = pd.read_csv("../sub/nn_esb_v20.tst.csv")
s14 = pd.read_csv("../sub/sgd_esb_v20.tst.csv")
s15 = pd.read_csv("../sub/xgb_esb_v25.tst.csv")


s[s1.columns[3:]] = ((s1[s1.columns[3:]]*0.3+s[s1.columns[3:]]*0.7)*0.7 + 0.05*s2[s1.columns[3:]] + 0.25*s3[s1.columns[3:]])*0.4 + 0.15*s13[s1.columns[3:]] + 0.15*s14[s1.columns[3:]] + 0.3*s15[s1.columns[3:]]
s.to_csv("../sub/ens13.csv", index=False) # cv: 0.157934278151; LB: 0.1339
