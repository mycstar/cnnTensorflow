import Bio.SeqIO as SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC

class seqObj(object):
    def __init__(self, num, id, code, sequence, start, end, family):
        self.num = num
        self.id = id
        self.code = code
        self.seq = sequence
        self.start = start
        self.end = end
        self.family = family


seq_dict = {}
seq_list = []
first_line = True
# seq_list = list(SeqIO.parse("/data1/projectpy/cnnTensorflow/data/CaricaPapaya/protease.lib", "fasta"))
i = 0
j=0
with open("/data1/projectpy/cnnTensorflow/data/CaricaPapaya/CA_group_limit_5.txt", 'r') as infile:
    for row in infile:
        seqs = []
        num, id, code, sequence, start, end, family, rest1, rest2 = row.split(",")
        if (sequence != 'NULL') and (len(sequence) > 50) and (len(sequence) < 1000):
            #seq_list.append(seqObj(num, id, code, sequence, start, end, family))
            record = SeqRecord(Seq(sequence,        IUPAC.protein),     num,  id, description=id)
            seq_list.append(record)
        else:
            j +=1
            print("%s line %s sequence is not valid" % (i, num))

        i +=1
print("total invalid sequence is:", j)
print("total duplicate sequence is:", len(seq_list))
for seq_record in seq_list:
    if seq_record.id not in seq_dict:
        seq_dict[seq_record.name] = seq_record
    else:
        print("duplicate id:", seq_record.id)

print("total unique sequence is:", len(seq_dict))


with open("/data1/projectpy/cnnTensorflow/data/CaricaPapaya/CA_group_limit_5.fa", 'w') as output_handle:
    SeqIO.write(seq_list, output_handle, "fasta")

with open("/data1/projectpy/cnnTensorflow/data/CaricaPapaya/CA_group_limit_5_data.txt", 'w') as output_handle:
    for seq_record in seq_list:
        sample = '1 ' + seq_record.seq
        output_handle.writelines("%s\n" % sample)
