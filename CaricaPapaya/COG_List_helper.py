import Bio.SeqIO as SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC


def check_invalid_category(word):
    cog_clear_dict = {}
    # seq_list = list(SeqIO.parse("/data1/projectpy/cnnTensorflow/data/CaricaPapaya/protease.lib", "fasta"))
    i = 0
    with open("/data1/projectpy/cnnTensorflow/data/CaricaPapaya/COG_List", 'r') as infile:
        for row in infile:
            seqs = []
            org, pro, COG, cat, annotation, pattern = row.split("\t")
            COG = COG.rstrip()
            COG = COG.strip()
            if cat.find(word) == -1:
                cog_clear_dict[COG] = annotation
            else:
                print("%s line %s sequence is null" % (i, COG))
            i += 1

    print("total clear OG_List base family is:", len(cog_clear_dict))

    target_dict = {}
    # seq_list = list(SeqIO.parse("/data1/projectpy/cnnTensorflow/data/CaricaPapaya/protease.lib", "fasta"))
    i = 0
    with open("/data1/projectpy/DeepFam/data/COG-500-1074/90percent/trans.txt", 'r') as infile:
        for row in infile:
            seqs = []
            rep, COG = row.split("\t")
            COG = COG.rstrip()

            if COG in cog_clear_dict:
                target_dict[rep] = COG
            else:
                print("%s line %s sequence is null" % (i, COG))
            i += 1

    print("total clear target  COG family is:", len(target_dict))

def change_label(inputfile,outputfile):
    with open(inputfile, 'r') as infile:
        with open(outputfile, 'w') as output_handle:
            for row in infile:
                seqs = []
                label, seq = row.split("\t")
                seq = seq.rstrip()

                sample = '0 ' + seq
                output_handle.writelines("%s\n" % sample)

if __name__ == '__main__':
    word = 'protease'
    #check_invalid_category(word)
    change_label("/data1/projectpy/DeepFam/data/COG-500-1074/90percent/random1500_b.txt","/data1/projectpy/DeepFam/data/COG-500-1074/90percent/random1500_b_label_0.txt")



