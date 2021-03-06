#!/usr/bin/python
import Bid, Slot


class Auction:
    'This class represents an auction of multiple ad slots to multiple advertisers'

    def __init__(self, term, bids1=[]):
        self.query = term
        self.query = ""
        self.bids = []

        for b in bids1:
            j = 0
            while j < len(self.bids) and float(b.value) < float(self.bids[j].value):
                j += 1
            self.bids.insert(j, b)  # sorts bids from highest to lowest

    '''
    This method accepts a Vector of slots and fills it with the results
    of a VCG auction. The competition for those slots is specified in the bids Vector.
    @param slots a Vector of Slots, which (on entry) specifies only the clickThruRates
    and (on exit) also specifies the name of the bidder who won that slot,
    the price said bidder must pay,
    and the expected profit for the bidder.
    '''

    def executeVCG(self, slots):

        for key, slot in enumerate(slots):
            # assign slot winners according to sorted order of bidders
            if key < len(self.bids):
                slot.bidder = self.bids[key].name
                slot.win_bid = self.bids[key].value

            # calculate lost ctr for all slots with successive slot
            if key < len(slots) - 1:
                slot.calc_lost_ctr(slots[key + 1].clickThruRate)

        for key, bid in enumerate(self.bids):
            # calculate cpc for all bidders who got a slot:
            if key < len(slots) - 1:
                bid.calc_cpc(slots[key+1].win_bid, slots[key + 1].clickThruRate)

        # in case there are more slots than bids:
        if len(slots) > len(self.bids):
            # set lost_ctr of last bidder to 0, since he did not harm anybody
            slots[len(self.bids)-1].lost_ctr = 0.0
            # last bidder does not have a bidder cpc that can be used to compute vcg price
            self.bids[len(self.bids)-1].cpc = 0.0

        # in case there are more/equal bidders than slots:
        if len(slots) < len(self.bids):
            # set lost_ctr of last slot to ctr of last slot
            slots[len(slots)-1].lost_ctr = slots[len(slots)-1].clickThruRate
            # last successful bidder has cpc that of a first bidder who did no get a spot
            self.bids[len(slots)-1].calc_cpc(self.bids[len(slots)].value, slots[len(slots)-1].clickThruRate)

        price_pass = [slot.lost_ctr*self.bids[key].cpc for key, slot in enumerate(slots) if key < len(self.bids)]
        for i in range(len(price_pass)):
            slots[i].set_price(price_pass[i])

        for key, slot in reversed(list(enumerate(slots))):
            # accumulate vcg price from bottom to top slots
            if key < len(slots) - 1:
                slot.price += slots[key + 1].price

        for slot in slots:
            if slot.win_bid >= slot.price:
                slot.profit = slot.price
