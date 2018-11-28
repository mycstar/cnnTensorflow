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
#seq_list = list(SeqIO.parse("/data1/projectpy/cnnTensorflow/data/CaricaPapaya/protease.lib", "fasta"))
i = 0
with open("/data1/projectpy/cnnTensorflow/data/CaricaPapaya/protease1.lib", 'r') as infile:
    for row in infile:
        #print("line:", i)
        if row.startswith(">"):
            if not first_line:
                seq = ""
                for se in seqs:
                    seq += se.rstrip()

                seq_list.append(seqObj(id, desc, seq))
            first_line = False

            seqs = []
            id, desc = row.split(" ", 1)
            #desc = desc.rstrip()
            id = str.replace(id, ">", "")
            #print("id:", id)
        else:
            seqs.append(row)
        i +=1


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
    for seqO in target_seq:
        #print(">%s %s" % (seqO.id, seqO.desc))
        outh = seqO.id+' '+seqO.desc
        print(outh)
        output_handle.write(outh)
        output_handle.write("%s\n" % seqO.seq)

    #SeqIO.write(target_seq, output_handle, "fasta")




