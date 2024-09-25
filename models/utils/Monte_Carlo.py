import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

def get_company_data(company_name, start_date, end_date):
    """Read from CSV file and obtain specified company data

    Parameters
    ---------
    company_name: in set `{'A', 'AA', 'ABC', 'ABCB', 'ACLS',
    'ACNB', 'ADBE', 'ADP', 'AEG', 'AIR'}`

    Returns
    -------
    Pandas Dataframe with indexes: Date, company name
    """
    assert company_name in [
        'A', 'AA', 'ABC', 'ABCB', 'ACLS',
        'ACNB', 'ADBE', 'ADP', 'AEG', 'AIR'
    ], f"Company stock data not found"
    company_string = '..\\..\\data\\raw_data\\' + company_name + '.csv'
    data = pd.read_csv(company_string, parse_dates=['Date'])
    data['Date'] = pd.to_datetime(data['Date'])
    # drop columns except "Date, close price"
    data.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
    return data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

class MonteCarlo:
    def __init__(self, price_series: pd.DataFrame, simTimes=4000):
        """
        Parameters
        ----------
        price_series:
            pd DataFrame, with company name being column name
            and stock price being column data
        """
        self.price_series = price_series
        self.mean_income = price_series.mean()
        self.cov_returns = price_series.cov()
        self.simTimes = simTimes
        self.companyNum = len(price_series.columns)
    def __randWeights(self):
        """Generate random weights for shares"""
        share = np.exp(np.random.randn(self.companyNum))
        share = share / share.sum()
        return share
    def __randProfit(self, portfolio):
        """Calculate profit with given portfolio"""
        mean_price = np.array(self.mean_income)
        assert len(mean_price) == len(portfolio)
        return mean_price.dot(portfolio)
    def __randRisk(self, portfolio):
        """Examine risk by multiplying weight and variance"""
        # risk actually quadratic w.r.t. portfolio
        covariance = np.array(self.cov_returns)
        assert len(covariance) == len(portfolio)
        return np.sqrt(portfolio.T @ covariance @ portfolio)
    def build_portfolio_helper(self, visualize=False):
        """If visualize, show point with the max Sharpe ratio
        Parameters
        ----------
        visualize:
            if set to true, show point with the max Sharpe Ratio

        Returns
        ------
        optimal_protfolio:
            optimal weights according to Monte-Carlo
        """
        portfolio_sim = np.zeros((self.simTimes, self.companyNum)) # each row being weight assigned to company stock
        profit = np.zeros(self.simTimes)
        risk = np.zeros(self.simTimes)
        # run monte carlo simulation
        for i in range(self.simTimes):
            weights = self.__randWeights()
            portfolio_sim[i, :] = weights
            risk[i] = self.__randRisk(weights)
            profit[i] = self.__randProfit(weights)
            maxSharpeRatioPoint = np.argmax(np.divide(profit, risk))
        if visualize:
            plt.figure(figsize=(15,8))
            plt.scatter(risk*100, profit*100, marker='.', color='blue')
            plt.xlabel('risk')
            plt.ylabel('profit')
            plt.title("Simulation result of randomly generated portfolio")
            plt.scatter(risk[maxSharpeRatioPoint]*100,
                        profit[maxSharpeRatioPoint]*100,
                        marker='o', color='red', label='Point with Max Sharpe Ratio')
            plt.legend()
        return portfolio_sim[maxSharpeRatioPoint]
    def build_portfolio(self):
        portfolio = np.zeros(self.companyNum)
        itertimes = 200
        for _ in range(itertimes):
            optimal = self.build_portfolio_helper()
            portfolio = portfolio + optimal
        portfolio /= itertimes
        return portfolio
    def __monteCarlo(start_price, mu, sigma, simPeriod=365):
        """Run Monte-Carlo simulation for a period of one year"""
        step = 1 / simPeriod
        price = np.zeros(simPeriod)
        price[0] = start_price
        shock = np.zeros(simPeriod)
        drift = np.zeros(simPeriod)
        for i in range(1, simPeriod):
            shock[i] = np.random.normal(loc=mu*step, scale=sigma*np.sqrt(step))
            drift[i] = mu*step
            price[i] = price[i-1] + price[i-1]*(drift[i]+shock[i])
        return price
    def simulation(self, company, simTimes, start_price, simPeriod=365):
        """"Run monte-carlo on specified company for simTimes
        
        Returns
        -------
        simArr: 
            simulation result obtained for #simTimes
        """
        print(f"Running simulation for company {company}")
        simArr = np.zeros(simTimes)
        plt.figure(figsize=(15,8))
        for i in range(simTimes):
            result = self.__monteCarlo(start_price, self.mean_income[company], self.cov_returns[company], simPeriod)
            simArr[i] = result[simPeriod-1] # assume at last time point, reach stable point
            plt.plot(result) # one year simulation
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.title(f'Simulation of {simPeriod}days for {company}')
        plt.savefig(fname=f'..\\figures\\{company}_simulation.png')
        return simArr
    def getResult(self, company_list, simTimes, start_price, simPeriod=365):
        assert len(company_list) == self.companyNum
        company_stock_sim = {} # simulation result for each company
        for i in range(self.companyNum):
            company_stock_sim[company_list[i]] = \
            self.simulation(company=company_list[i], simTimes=simTimes,
                    start_price=start_price[i], simPeriod=simPeriod)
        sim_mean, sim_std = [], []
        for company in company_list:
            company_mean = np.mean(company_stock_sim[company])
            company_std = np.std(company_stock_sim[company])
            sim_mean.append(company_mean)
            sim_std.append(company_std)
        frame = np.array([start_price, sim_mean, sim_std])
        analysis = pd.DataFrame(frame.T,
            columns=['Start Price', 'Mean', 'Variance'],
            index=company_list)
        return analysis

if __name__ == '__main__':
    end_time = dt(2019, 9, 1)
    start_time = dt(2009, 1, 1)
    company_list = ['A', 'AA', 'ABC', 'ABCB', 'ACLS','ACNB', 'ADBE', 'ADP', 'AEG', 'AIR']
    stock_price, start_price = [], []
    for company_name in company_list:
        company_data = get_company_data(company_name, start_time, end_time)
        start_price.append(company_data.tail(1)['Close'].values)
        # compute fractional change of current and previous for measuring changes
        # which is given as (next - previous) / previous
        stock_data = np.array(company_data['Close'].pct_change().dropna())
        stock_price.append(stock_data)
    price_array = np.array(stock_price)
    price_series = pd.DataFrame(price_array.T, columns=company_list)
    start_price = np.array(start_price).flatten()
    # monte-carlo simulation
    model = MonteCarlo(price_series)
    portfolio = model.build_portfolio()
    company_dict = {}
    for company in company_list:
        company_dict[company] = portfolio[company_list.index(company)]
    print(f"{company_dict}")
