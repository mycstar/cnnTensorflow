class alignedProtein(object):
    def __init__(self, name, level, seq, alignedseq, alignedStart, alignedEnd, activeSites):
        self.name = name
        self.level = level
        self.seq = seq
        self.alignedseq = alignedseq
        self.alignedstart, self.alignedend = (alignedStart, alignedEnd)
        self.activeSites = activeSites

    def getSeq(self):

        return self.seq
