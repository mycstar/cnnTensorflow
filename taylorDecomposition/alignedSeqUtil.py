import Bio.SeqIO as SeqIO
from alignedSeq import alignedSeq


# from Bio import AlignIO
# alignment = AlignIO.read("/data1/projectpy/cnnTensorflow/data/PF00571_seed.txt", "fasta")
# print("Number of rows: %i" % len(alignment))


# for seq_record in SeqIO.parse("/data1/projectpy/DeepFam/seq2logo-2.1/PF00571_full_length_sequences.fasta", "fasta"):
#
# print(seq_record.id)
#
def get_aligned_seqs(fastaFile, name):
    seq_length = 0
    raw_dict = {}
    motifs = []
    seq_list = list(SeqIO.parse("/data1/projectpy/cnnTensorflow/data/PF00571_seed.txt", "fasta"))
    for seq_record in seq_list:
        id = seq_record.id
        name_id, locations = id.split("/")
        if name_id in raw_dict:
            motifs = raw_dict[name_id]
            alignedSeq1 = alignedSeq(name_id, locations, seq_record.seq)
            motifs.append(alignedSeq1)
        else:

            alignedSeq1 = alignedSeq(name_id, locations, seq_record.seq)
            motifs = [alignedSeq1]
            raw_dict[name_id] = motifs

    return raw_dict


def get_seqs(fastaFile, name):
    aligned_seqs = get_aligned_seqs("d","d")
    seqs = aligned_seqs[name]
    return seqs


