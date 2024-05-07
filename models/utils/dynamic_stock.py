import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dynamic_stock(totalFund_, extraFund_, y_predict, y_true):
    totalFundArr, extraFundArr = [totalFund_], [extraFund_]
    totalFund = totalFund_
    extraFund = extraFund_ # split original fund for dynamic processing
    price_at_window_start, price_at_window_end = totalFund, 0
    stockNum = 0 # current number of stock at hand
    dayNum = len(y_predict)
    has_stock = False # has stock in hand currently
    for today in range(dayNum - 1): # take one day in advance
        totalFundArr.append(totalFund)
        extraFundArr.append(extraFund)
        if y_predict[today + 1] > y_true[today] and not has_stock:
            # stock predicted to be rising and dont have stock yet
            price_at_window_start = totalFund
            stockNum = totalFund / y_true[today]
            totalFund = 0
            has_stock = True
        elif y_predict[today + 1] < y_true[today] and has_stock:
            # price predicted to be falling, sell the stock
            totalFund = stockNum * y_true[today]
            has_stock = False
            # examine the stock
            price_at_window_end = totalFund
            ratio = price_at_window_start / price_at_window_end
            if ratio < 1:
                # promising stock, add more fund
                increment = totalFund * (1 - ratio)
                if extraFund < increment:
                    extraFund = 0
                    totalFund += extraFund
                else:
                    extraFund -= increment
                    totalFund += increment
            else:
                # reduce the fund in this stock
                decrement = totalFund * (ratio - 1)
                totalFund -= decrement
                extraFund += decrement
    if has_stock:
        # at end of simualtion, sell the stock
        totalFund = stockNum * y_true[-1]
    totalFundArr.append(totalFund)
    extraFundArr.append(extraFund)
    plt.plot(np.arange(dayNum+1), totalFundArr, 'r.', label='totalFund')
    plt.plot(np.arange(dayNum+1), extraFundArr, 'b*', label='extraFund')
    return totalFund + extraFund

if __name__ == '__main__':
    # company_list = ['A', 'AA', 'ABC', 'ABCB', 'ACLS', 
    #                 'ACNB', 'ADBE', 'ADP', 'AEG', 'AIR']
    # weights = {
    #     'A': 3.687049859524796, 'AA': 2.219831145816969, 'ABC': 3.1676117011522384, 'ABCB': 4.337304887824643,
    #     'ACLS': 3.4969123897615013, 'ACNB': 23.896297604274523, 'ADBE': 47.189297012758736, 'ADP': 6.802566702378293,
    #     'AEG': 1.6770488395957608, 'AIR': 3.5260798569125256
    # }
    # totalFund = 95
    # extraFund = 5 # split original fund for dynamic processing
    # profit = 0
    # for company in company_list:
    #     init_fund = totalFund * weights[company] / 100
    #     init_extra_fund = extraFund * weights[company] / 100
    #     path = f"../data/predictions/{company}_predictions.csv"
    #     predict_true = pd.read_csv(path)
    #     y_predict = np.array(predict_true['Predictions'])
    #     true_price = np.array(predict_true['Actual'])
    #     earn = dynamic_stock(init_fund, init_extra_fund, y_predict, true_price)
    #     profit += earn
    # earning_ratio = (profit - (totalFund + extraFund))/(totalFund + extraFund)
    # print(f"Earning: {earning_ratio * 100}")
    # print(f"Profit: {profit}")
    totalFund = 50
    extraFund = 50
    path = f"../data/predictions/AA_predictions.csv"
    predict_true = pd.read_csv(path)
    y_predict = np.array(predict_true['Predictions'])
    true_price = np.array(predict_true['Actual'])
    dynamic_stock(totalFund, extraFund, y_predict, true_price)
    plt.legend()
    plt.show()

