import os
import argparse
import numpy as np
import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='..\\..\\data', help='Director containing .csv files')

def get_full_company_data(company_name, start_date, end_date):
    """Read from CSV file and obtain specified company data

    Parameters
    ---------
    company_name: in set `{'A', 'AA' 'ABC', 'ABCB', 'ACLS',
    'ACNB', 'ADBE', 'ADP', 'AEG', 'AIR'}`

    Returns
    -------
    Pandas Dataframe with indexes: Date, company name
    """
    assert os.path.isdir(parser.data_dir), f"Couldn't find data at {parser.data_dir}"
    company_string = company_name + '.csv'
    path = os.path.join(parser.data_dir, company_string)
    data = pd.read_csv(path, parse_dates=['Date'])
    data['Date'] = pd.to_datetime(data['Date'])
    # drop columns except "Date, close price"
    data.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
    return data[(data['Date'] > start_date) & (data['Date'] < end_date)]
def company_data(data, company, start_time, end_time):
    """ extract specific column containing company name """
    column_list = ['Date']
    column_list.append(company)
    specific_data = data[column_list]
    return specific_data[(specific_data['Date'] > start_time) 
        & (specific_data['Date'] < end_time)]

class Long_Short_Term_Memory:
    def __init__(self, data, test_percent, seq_length):
        """data being ndarray, obtained from calling 
        DataFrame[companyName]
        """
        self.test_percent = test_percent
        self.data = data
        self.seq_length = seq_length
        self.model = None
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        self.scaler = MinMaxScaler()
    def data_processing(self, data):
        """Return data in form of LSTM training samples
        
        Parameters
        ----------
        data: 
            test or train data with type pandas dataframe

        Returns
        -------
        X_train, y_train: numpy array with y_train begin true price, 
        X_train being previous price (or X_test, y_test if test phase)
        """
        X_train, y_train = [], []
        for i in range(self.seq_length, len(data)):
            # X_train contain groups of #seq_length numbers
            X_train.append(data[i-self.seq_length:i, 0])
            # y_train contains the data right after X_train
            y_train.append(data[i, 0])
        return np.array(X_train), np.array(y_train)
    def fit(self, restore_weights=False, restore_path=None):
        """Train a LSTM model for a specific company

        Parameters
        ----------
        restore_weights:
            whether to restore weights (.h5 file)
        restore_path:
            if restore_weights=True, then specify path containing .h5 file
        """
        normalized_data = self.scaler.fit_transform(self.data.reshape(-1, 1))
        split = int(self.test_percent * len(normalized_data) * (1-self.test_percent))
        train_data = normalized_data[:split]
        test_data = normalized_data[split:]
        X_train, y_train = self.data_processing(train_data)
        X_test, y_test = self.data_processing(test_data)
        # reshape to tensor
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        self.X_train, self.X_test = X_train.copy(), X_test.copy()
        self.y_train, self.y_test = y_train.copy(), y_test.copy()
        if restore_weights:
            assert restore_path is not None, "Need to specify directory\
            containing the trained weights"
            self.model = load_model(restore_path)
            self.model.summary()
        else:
            self.model = Sequential()
            self.model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units=50, return_sequences=True))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units=50))
            self.model.add(Dense(units=1))
            self.model.compile(loss='mean_squared_error', optimizer='Adam')
            checkpoints = ModelCheckpoint(filepath='output\\LSTM_weights.h5', save_best_only=True)
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            self.model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test), 
                    epochs=100,
                    batch_size=32,
                    verbose=1,
                    callbacks=[checkpoints, early_stopping])
    def evaluate(self):
        """ Plot train and test data

        Returns
        -------
        test_MAE, train_MAE:
            training and test mean absolute error, respectively
        """
        train_predict = self.model.predict(self.X_train)
        test_predict = self.model.predict(self.X_test)
        # train error and test error while at same scale
        train_MAE = mean_absolute_error(train_predict, self.y_train)
        test_MAE = mean_absolute_error(test_predict, self.y_test)
        train_predict = self.scaler.inverse_transform(train_predict)
        test_predict = self.scaler.inverse_transform(test_predict)
        # begin plotting train data
        trainPredictPlot = np.empty_like(self.data)
        trainPredictPlot[:] = np.nan
        trainPredictPlot[self.seq_length:len(train_predict)+self.seq_length] = train_predict.flatten()
        # plotting test data
        testPredictPlot = np.empty_like(self.data)
        testPredictPlot[:] = np.nan
        test_start = len(self.data) - len(test_predict)
        testPredictPlot[test_start:] = test_predict.flatten()
        # Plotting the baseline data, training predictions, and test predictions
        plt.figure(figsize=(15, 6))
        plt.plot(self.data, color='black', label=f"Actual price")
        plt.plot(trainPredictPlot, color='r', label=f"Predicted price (train set)")
        plt.plot(testPredictPlot, color='b', label=f"Predicted price (test set)")

        plt.title(f"Stock Price Prediction")
        plt.xlabel("time")
        plt.ylabel(f"Closing price")
        plt.legend()
        plt.show()
        return test_MAE, train_MAE
    def predict(self, data=None):
        """
        Parameters
        ----------
        data:
            tensor of shape (None, None, 1)
            if not specified, use self.X_test instead

        Returns
        -------
        y_predict:
            stock value predicted using data (if specified)
            using seq_length days to predict one more day
        """
        if data is None:
            data = self.X_test
        # data = self.scaler.fit_transform(data.flatten().reshape(-1,1))
        predicted = self.model.predict(data)
        original_data = self.scaler.inverse_transform(predicted)
        return original_data
if __name__ == '__main__':
    data = pd.read_csv('..\\..\\data\\2005_2019_10_close.csv', parse_dates=['Date'])
    # convert date to time datetime
    data['Date'] = pd.to_datetime(data['Date'])
    company = 'AA'
    start_time = dt.datetime(2014,1,1)
    end_time = dt.datetime(2019,1,1)
    company_stock_price = company_data(data, company, start_time, end_time)
    price = np.array(company_stock_price[company])
    model = Long_Short_Term_Memory(price, test_percent=0.2, seq_length=60)
    model.fit(restore_weights=True, restore_path='..\\..\\weights\\LSTM_AA_weights.h5')
    test_MAE = model.evaluate()
    predicted = model.predict()
    print(f"The test error: {test_MAE} \n \
        The predicted value: {predicted}")