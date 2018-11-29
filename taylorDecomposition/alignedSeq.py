class alignedSeq(object):
    def __init__(self, name, location, seq):
        self.name = name
        self.location = location
        self.seq = seq
        self.start, self.end = (int(item) for item in location.split("-"))
