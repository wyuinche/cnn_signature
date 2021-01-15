# Online Signiture Verification Using CNN-LSTM and General Written Data

## Abstract
 This is the research about online signature verification using cnn(cnn-lstm) and general written data.

***

## Models

    NET1: CNN (3 Dropout Layer)
    
    NET2: CNN (6-8 Dropout Layer)
    
    NET3: CNN-LSTM (3 Dropout Layer)
    
    NET4: CNN-LSTM (6-8 Dropout Layer)
    
    NET5: CNN-LSTM (6-8 Dropout Layer / Low Propability)

## Data

    CASE1: Original(50), Forgery(30), General Written(1), Others(4)
    CASE2: Original(50), Forgery(30), General Written(1), Others(4)
    
### Density

    CASE1-normal: 0.08943315217391304
    CASE1-pen: 0.43821739130434784
    CASE2-normal:0.47111847826086956
    CASE2-pen: 0.6864907608695652
    
## Test

### Learning
    
    Data: Origin(30), Forgery(10), General Written(1), Others(4)
    Epoch: 25/100
    Batch Size: 10

### Test

    Data: Origin(50), Forgery(30)

## Results

    CASE1-normal
    - NET1: 0.3750 (Overfitting)
    - NET2: 0.3750 (Overfitting)
    - NET3: 0.6250 (Underfitting)
    - NET4: 0.6250
    - NET5: 0.3750 (Overfitting)
    
    CASE1-pen
    - NET1: 0.6250 (Underfitting)
    - NET2: 0.6250 (Underfitting)
    - NET3: 0.6250 (Underfitting)
    - NET4: 0.6250 (Underfitting)
    - NET5: 0.6250 (Underfitting)
    
    CASE2-normal
    - NET1: 0.9625
    - NET2: 0.3750 (Overfitting)
    - NET3: 0.6250 (Underfitting)
    - NET4: 0.6250 (Underfitting)
    - NET5: 0.6250 (Underfitting)
    
    CASE2-pen
    - NET1: 0.3750 (Overfitting)
    - NET2: 0.3750 (Overfitting)
    - NET3: 0.6250 (Underfitting)
    - NET4: 0.5500
    - NET5: 0.5625
  
***

## Folders
#### CNN_LATEST
- In this folder, there are two files of CNN. (NET1, NET2)
#### CNN_LSTM_LATEST
- In this folder, there are three files of CNN-LSTM. (NET3, NET4, NET5)
#### CASE1_result & CASE2_result
- Files which contains the result of each test (NET1-NET5) with CASE1, CASE2 signature data, resp. (epoch 25)
#### CASE1_result_epoch100 & CASE2_result_epoch100
- Files which contains the result of each test (NET1-NET5) with CASE1, CASE2 signature data, resp. (epoch 100)
#### CASE1_data & CASE2_data (moved to private folder)
- Signature data of CASE1, CASE2 resp.
#### CNN & CNN_LSTM
- Files used for tests. (not used)
#### tmp (moved to private folder)
- Temporal files. (not used)
***

2020 한국정보처리학회 춘계학술발표대회 투고 논문 연구노트 (KIPS_C2020A0124)
 - 일반 필기데이터와 CNN을 이용한 온라인 서명인식 (Online Signature Verification using General Handwriting Data and CNN)
 
