# DS Term Project: Agricultural Price Prediction (농산물 물가 예측)

This repository contains the solution for the Dacon Agricultural Price Prediction competition ("데이터·AI를 활용한 물가 예측 경진대회"). 

## Goal
Predict the average prices of 10 agricultural products ('건고추', '사과', '감자', '배', '깐마늘(국산)', '무', '상추', '배추', '양파', '대파') at T+1, T+2, and T+3 soon (10-day periods) given past data from T-8 to T.

## Features
1. **Date Reconstruction**: Test periods ($T-8$ to $T$) are mapped back to their exact calendar dates by matching the sequence of 2021 previous-year prices in the test set metadata against actual 2021 training prices.
2. **Feature Engineering**:
   - Seasonal baseline feature (`train_seasonal_mean`) calculated for each of the 36 periods of the year.
   - Circular calendar features using Sine/Cosine encoding for month and soon.
   - One-hot encoded seasons.
3. **Instance Scale Normalization**: All prices in each window are normalized relative to the price at $T$. The model predicts change ratios, which are then scaled back to the actual price level during inference. This handles scale variation across items and long-term trend shifts.
4. **GRU-MLP Forecasting Model**: Separate models are trained for each of the 10 products. A GRU encoder captures temporal sequences, and an MLP decoder takes both the hidden state and target calendar features to output the multi-step predictions.

## Execution
Ensure you have `pandas`, `numpy`, and `torch` installed.

To train the models and generate the submission file:
```bash
python train_predict.py
```
The script will output `submission.csv` in the root directory.
