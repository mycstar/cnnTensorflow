import numpy as np
import sys
import os
import Bio.SeqIO as SeqIO
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


def get_data0():
    featureSeq = {}

    with open("/home/myc/projectpy/cnnTensorflowNew/data/CaricaPapaya/CA_group_features_limit_5_union_level_family.fa.csv",
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
def get_data():
    featureSeq = {}

    with open("/home/myc/projectpy/cnnTensorflowNew/data/CaricaPapaya/CA_activatesite_features_CA_testseq3.fa.csv",
              'r') as infile:
        i = 0
        name = ''
        latestName = ''

        for row in infile:
            if i != 0:
                cols = row.split(",")
                name = cols[0].strip()

                if name == latestName:
                    activeSite = int(cols[6].strip())
                    activeSites.append(activeSite)

                else:
                    if i > 1:
                        #                        if all(item < 1000 for item in activeSites):
                        if len(seq) < 1001:
                            alignedProtein1 = alignedProtein(latestName, level, seq, alignedseq, alignedstart,
                                                             alignedend,
                                                             activeSites)
                            featureSeq[latestName] = alignedProtein1

                    level='family'
                    alignedseq = cols[2].strip()
                    seq = cols[2].strip()
                    alignedstart = int(cols[3].strip())
                    alignedend = int(cols[4].strip())

                    activeSites = []
                    activeSite = int(cols[6].strip())
                    activeSites.append(activeSite)

                latestName = name

            i += 1
            # if i>359:
            #     print(i)

    print("total sequence :", len(featureSeq))

    return featureSeq

def convert_seq_list_to_dict(seq_list127):
    res_dict = {}
    for seqRecord in seq_list127:
        res_dict[seqRecord.id] = seqRecord.seq
    return res_dict


def contain_calc():
    seq_list127 = list(SeqIO.parse("/data1/projectpy/cnnTensorflow/data/CaricaPapaya/CA_group_limit_5.fa", "fasta"))
    seq_127_dict = convert_seq_list_to_dict(seq_list127)
    seq_list81 = list(SeqIO.parse("/data1/projectpy/cnnTensorflow/CaricaPapaya/result/CA_group_features_limit_5_union_level_family_feature_seq.csv", "fasta"))
    cover_count = 0
    for seqRecord in seq_list81:
        id = seqRecord.id
        if id in seq_127_dict.keys():
            cover_count +=1

    print("base list seq is:"+str(len(seq_127_dict)))
    print("B list seq is:" + str(len(seq_list81)))
    print("contained seq is:" + str(cover_count))
    print("cover rate :", float(cover_count)/float(len(seq_127_dict)))


if __name__ == "__main__":
    get_data()
