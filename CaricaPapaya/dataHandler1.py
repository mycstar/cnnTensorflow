import Bio.SeqIO as SeqIO

# for seq_record in SeqIO.parse("/data1/projectpy/DeepFam/seq2logo-2.1/PF00571_full_length_sequences.fasta", "fasta"):
#
#     print(seq_record.id)
#


class seqObj(object):
    def __init__(self, id, desc, seq):
        self.id = id
        self.desc = desc
        self.seq = seq

seq_length = 0
seq_dict = {}
seq_list =[]
first_line = True
seq_list = list(SeqIO.parse("/data1/projectpy/cnnTensorflow/data/CaricaPapaya/protease1.lib", "fasta"))

print("total protease.lib sequence is:", len(seq_list))
for seq_record in seq_list:
    if seq_record.id not in seq_dict:
        seq_dict[seq_record.id] = seq_record
    else:
        print("duplicate id:", seq_record.id)

print("total unique protease.lib sequence is:", len(seq_dict))


target_seq = []
ali_seq_list = list(SeqIO.parse("/data1/projectpy/cnnTensorflow/data/CaricaPapaya/c1a.fa", "fasta"))
print("total c1a.fa sequence is:", len(ali_seq_list))
for seq_record in ali_seq_list:
    if seq_record.id in seq_dict:
        # ali_seq_dict[seq_record.id] = seq_record.seq
        target_seq.append(seq_dict[seq_record.id])

print("c1a_seq.fa sequence is:", len(target_seq))

with open("/data1/projectpy/cnnTensorflow/data/CaricaPapaya/c1a_seq.fa", 'w') as output_handle:
    SeqIO.write(target_seq, output_handle, "fasta")

target_seq = []
ali_seq_list = list(SeqIO.parse("/data1/projectpy/cnnTensorflow/data/CaricaPapaya/c1b.fa", "fasta"))
print("total c1b.fa sequence is:", len(ali_seq_list))
for seq_record in ali_seq_list:
    if seq_record.id in seq_dict:
        # ali_seq_dict[seq_record.id] = seq_record.seq
        target_seq.append(seq_dict[seq_record.id])

print("c1b_seq.fa sequence is:", len(target_seq))

with open("/data1/projectpy/cnnTensorflow/data/CaricaPapaya/c1b_seq.fa", 'w') as output_handle:
    SeqIO.write(target_seq, output_handle, "fasta")


target_seq = []
ali_seq_list = list(SeqIO.parse("/data1/projectpy/cnnTensorflow/data/CaricaPapaya/c1.fa", "fasta"))
print("total c1.fa sequence is:", len(ali_seq_list))
for seq_record in ali_seq_list:
    if seq_record.id in seq_dict:
        # ali_seq_dict[seq_record.id] = seq_record.seq
        target_seq.append(seq_dict[seq_record.id])

print("c1_seq.fa sequence is:", len(target_seq))

with open("/data1/projectpy/cnnTensorflow/data/CaricaPapaya/c1_seq.fa", 'w') as output_handle:
    SeqIO.write(target_seq, output_handle, "fasta")


