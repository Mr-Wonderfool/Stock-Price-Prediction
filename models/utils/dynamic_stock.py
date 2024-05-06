import numpy as np
import pandas as pd

def dynamic_stock(totalFund_, extraFund_, y_predict, y_true):
    totalFund = totalFund_
    extraFund = extraFund_ # split original fund for dynamic processing
    price_at_window_start, price_at_window_end = totalFund, 0
    stockNum = 0 # current number of stock at hand
    dayNum = len(y_predict)
    has_stock = False # has stock in hand currently
    for today in range(0, dayNum - 1): # take one day in advance
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
            print(f"Sold stock at day{today}, \n \
            totalFund: {totalFund}, extraFund: {extraFund}")

if __name__ == '__main__':
    predict_true = pd.read_csv('..\\data\\AA_predictions.csv')
    y_predict = np.array(predict_true['Predictions'])
    true_price = np.array(predict_true['Actual'])
    totalFund = 50
    extraFund = 50 # split original fund for dynamic processing
    dynamic_stock(totalFund, extraFund, y_predict, true_price)