
def conv_odds(row):
    absrow = abs(row)
    if row >= 0:
        odds = (absrow / 100) + 1
    else:
        odds = (100 / absrow) + 1
    return odds

class Odds:
    def __init__(self, decimal: float):
        self.decimal = decimal

    @staticmethod
    def from_american(value: int) -> 'Odds':
        return Odds(conv_odds(value))

    @staticmethod
    def from_fractional(numerator: int, denominator: int) -> 'Odds':
        pass

    @staticmethod
    def from_win_percentage(win_percentage) -> 'Odds':
        return Odds(1 / win_percentage)

    def get_decimal(self) -> float:
        return self.decimal

    def get_american(self) -> float:
        pass

    def __str__(self) -> str:
        return str(self.get_decimal())

    def __repr__(self) -> str:
        return str(self)

    def reverse(self):
        return Odds.from_win_percentage(1 - self.get_chance_win())

    def get_winnings_multiplier(self):
        return self.decimal

    def get_chance_win(self):
        return 1 / self.decimal