### Company Stock Price Prediction
- data gathered from [Kaggle](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)
- predicting stock price of 10 representative companies
- utilizing LASSO, LSTM, TSN, Transformers
### 提交文件说明
- models
    - utils
      - LSTM.py预测股票价格
      - Monte-Carlo.py通过蒙特卡罗方法模拟得到本金初始分配策略
      - dynamic_stock.py动态投资策略代码文件(针对一只股票)
    - dynamic_stock_all.ipynb: 对于所有股票的动态交易策略，notebook中包含投资策略调整的可视化以及最终的收益
    - LSTM_TCN_predictions.ipynb: LSTM和TCN模型并行预测股票价格
- results
  - predictions: 其中每个.csv文件包含两列，真实的股票价格(Actual列)、预测的股票价格(Predicted列)，是股票投资策略依赖的基础数据文件
  - 可视化图像，分别为TCN预测AA股票的效果、TCN+LSTM预测股票AA的效果
- Report.pptx: 汇报的ppt文件