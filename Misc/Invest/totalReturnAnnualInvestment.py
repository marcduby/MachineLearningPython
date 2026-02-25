

# constants
LIST_RETURNS = [11, 14]
LIST_CONTRIBUTIONS = [10000, 25000, 50000]
LIST_YEARS = [30, 40]

# methods
def total_return(annual_investment, annual_return_pct, years):
    """
    Compute total invested amount and total value with compound returns.

    :param annual_investment: amount invested each year (float)
    :param annual_return_pct: annual return percentage (e.g. 7 for 7%)
    :param years: number of years invested (int)
    :return: (total_invested, final_value, total_gain)
    """
    rate = annual_return_pct / 100
    balance = 0.0

    for _ in range(years):
        balance = balance * (1 + rate)
        balance += annual_investment

    total_invested = annual_investment * years
    total_gain = balance - total_invested

    return total_invested, balance, total_gain


# Example usage
if __name__ == "__main__":
    # annual_investment = 10000     # dollars per year
    # annual_return_pct = 7         # percent
    # years = 30

    for annual_investment in LIST_CONTRIBUTIONS:
        for annual_return_pct in LIST_RETURNS:
            for years in LIST_YEARS:
              invested, final_value, gain = total_return(
                  annual_investment,
                  annual_return_pct,
                  years
    )

              print(f"Annual investment: ${annual_investment:,.2f}, rate: {annual_return_pct:.2f}%, years: {years}, total: ${final_value:,.2f}")



    # print(f"Total invested: ${invested:,.2f}")
    # print(f"Final value:    ${final_value:,.2f}")
    # print(f"Total gain:     ${gain:,.2f}")


