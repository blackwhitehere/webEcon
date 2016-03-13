#!/usr/bin/python

class Slot:
    'represents a slot where an ad can be posted'

    def set_price(self, price):
        self.price = price

    def calc_lost_ctr(self, ctr):
        self.lost_ctr = self.clickThruRate-ctr

    def calc_welfare(self):
        self.profit = self.win_bid-self.price

    def __init__(self, ctr):
        self.clickThruRate = 0.0  # the number of clicks expected
        self.price = 0.0  # price to be paid for those clicks
        self.win_bid = 0.0
        self.profit = 0.0  # profit expected from those clicks
        self.bidder = 0.0  # the Bid that wins this slot
        self.lost_ctr = 0.0
        if ctr is not None:
            self.clickThruRate = float(ctr)
        else:
            self.clickThruRate = 0

        # def toString(temp):
        #	print ("slot: %6.2f %8.2f %8.2f   %s" %(self.clickThruRate,self.price,self.profit,self.bidder))
        #	return ("slot: %6.2f %8.2f %8.2f   %s" %(self.clickThruRate,self.price,self.profit,self.bidder))
