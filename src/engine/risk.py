
class RiskManager:
    def __init__(self, sl_mult=8.0, tp_mult=60.0, be_at_R=0.65, tsl=None, tick_size=0.01):
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult
        self.be_at_R = be_at_R
        self.tsl = tsl or {"atr_mult":2.5, "floor_from_median_mult":1.1}
        self.tick = tick_size

    def simulate_trade(self, side, entry_price, atr_entry, series_close, series_atr, median_atr):
        R = atr_entry * self.sl_mult
        if side == "long":
            sl = entry_price - R
            tp = entry_price + self.tp_mult * atr_entry
        else:
            sl = entry_price + R
            tp = entry_price - self.tp_mult * atr_entry
        tsl_gap_min = max(self.tsl["atr_mult"]*series_atr[0], self.tsl["floor_from_median_mult"]*median_atr)
        be_moved = False
        for i in range(1, len(series_close)):
            p = series_close[i]; a = series_atr[i]
            # BE
            if not be_moved:
                up_move = (p - entry_price) if side=="long" else (entry_price - p)
                if up_move >= self.be_at_R * R:
                    sl = entry_price
                    be_moved = True
            # TSL
            gap = max(self.tsl["atr_mult"]*a, tsl_gap_min)
            if side=="long":
                sl = max(sl, p - gap)
                if p >= tp: return self.tp_mult / self.sl_mult, i
                if p <= sl: return (sl - entry_price) / R, i
            else:
                sl = min(sl, p + gap)
                if p <= tp: return self.tp_mult / self.sl_mult, i
                if p >= sl: return (entry_price - sl) / R, i
        # timeout exit
        if side=="long":
            return (series_close[-1]-entry_price)/R, len(series_close)-1
        else:
            return (entry_price-series_close[-1])/R, len(series_close)-1
