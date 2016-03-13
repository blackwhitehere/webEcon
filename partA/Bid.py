#!/usr/bin/python

class Bid:
    'represents a bid in a Sponsored Search Auction'

    def calc_cpc(self, value, ctr):
        self.cpc = value/ctr

    def __init__(self, line):
        self.cpc = 0.0
        self.value = 0.0  # the value to this bidder of a click
        self.name = ""  # the name of this bidder
        if line is not None:
            self.value = float(line.split('\t')[0])
            self.name = line.split('\t')[1]
        else:
            self.value = 0
            self.name = "empty"

            # def toString(temp):
            #	print ("bid:%6.2f %s"%(value,name))
            #	return ("bid:%6.2f %s"%(value,name))
