import numpy as np
import sys
import os
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from alignedProtein import alignedProtein

seq = ''
scores = []


def cur_script_name():
    argv0_list = sys.argv[0].split("/")
    script_name = argv0_list[len(argv0_list) - 1]  # get script file name self
    # print("current script:", script_name)
    script_name = script_name[0:-3]  # remove ".py"

    return script_name


def get_data():
    featureSeq = {}

    with open("/data1/projectpy/cnnTensorflow/data/CaricaPapaya/CA_group_features_limit_5_union_level_family.fa.csv",
              'r') as infile:
        i = 0
        name = ''
        latestName = ''

        for row in infile:
            if i != 0:
                cols = row.split(",")
                name = cols[0].strip()

                if name == latestName:
                    activeSite = int(cols[11].strip())
                    activeSites.append(activeSite)

                else:
                    if i > 1:
                        #                        if all(item < 1000 for item in activeSites):
                        if len(seq) < 1001:
                            alignedProtein1 = alignedProtein(latestName, level, seq, alignedseq, alignedstart,
                                                             alignedend,
                                                             activeSites)
                            featureSeq[latestName] = alignedProtein1

                    level = cols[1].strip()
                    alignedseq = cols[2].strip()
                    seq = cols[7].strip()
                    alignedstart = int(cols[8].strip())
                    alignedend = int(cols[9].strip())

                    activeSites = []
                    activeSite = int(cols[11].strip())
                    activeSites.append(activeSite)

                latestName = name

            i += 1
            # if i>359:
            #     print(i)

    print("total sequence :", len(featureSeq))

    return featureSeq


if __name__ == "__main__":
    get_data()
