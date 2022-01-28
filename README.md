# KOSPI_Prediction
LSTM으로 KOSPI 상위 100개 기업에 대한 주가를 학습하여 만든 모델에 대한 테스트 데이터를 통해 해당 기업에 대해 예측한 주가와 실제 주가를 비교하여 LSTM이 어느정도의 성능을 가지는지 확인한다.

https://user-images.githubusercontent.com/86700191/148202120-496648ae-1055-44fc-b9c4-b4502c6b55d1.mp4

## 1. 사용 라이브러리
- PyTorch : LSTM Network 구성
- Pandas : 주식데이터 수신, 데이터프레임 활용
- scikit-learn : 데이터 전처리
- tkinter : 전체 UI
- matplotlib : 주식데이터 차트구현
- firebase_admin : firebase에서 주식 종목코드 수신

## 2. 프로그램 기능
 1) 초기화면
   	- 주식종목을 검색할 수 있는 검색창과 콤보박스, 예측기간을 선택하는 라디오버튼, 과거부터 현재까지의 KOSPI 지수를 나타내는 차트가 있다.

 2) 주식종목 검색
	- 예측 시 검색창에 적은 종목명이 콤보박스로 선택한 것보다 우선순위가 높게 설정되어있다.
	- 데이터가 없는 잘못된 주식 종목명을 검색 시 데이터가 없다는 에러창이 발생한다.
	- 예측기간 미설정시 예측기간을 선택하라는 에러창이 발생한다.
	- 위 두 에러창이 발생 시 검색창과 콤보박스가 초기화된다.

 3) 예측 창
	- 예측완료시 화면에 띄어져 있던 차트를 초기화 하고 예측 주가와 성능을 비교할 실제주가 차트가 띄어진다.
	- 예측완료시 검색창과 콤보박스가 초기화된다.

 4) 종료 확인
	- 창 닫기 버튼(X버튼)을 누르면 종료의사 여부를 물어보는 창이 나오며, 종료하게되면 프로세스가 종료되고, tkinter 창이 닫히게 된다.
    
## 3. 알고리즘 순서도
![Ai_project drawio](https://user-images.githubusercontent.com/86700191/148198941-9b3c62b3-a458-4e7e-ad0b-1e08f90de779.png)

## 4. 네트워크 구성도
![Stock_lstm drawio](https://user-images.githubusercontent.com/86700191/151542527-f822bb9d-b29a-4c63-ad17-d2581f41a769.png)

## 5. 유의점
- Pandas가 있거나 install했음에도 pandas_datareader가 없다는 에러가 뜬다면 pandas_datareader를 install 해야한다.
    (pip install pandas-datareader)
- matplotlib 한글 폰트는 따로 지원을 하지 않으므로 폰트를 가져와 코드를 작성해야한다.
- KOSPI의 주식 종목코드를 Firebase를 통해 수신하므로 Firebase Admin SDK를 만들어 코드에 작성하여야 한다. 그리고 Firestore Database 내에 주식 종목코드를 저장시켜야 정상적으로 실행이 가능하다. 
  
  예시
  ![capture](https://user-images.githubusercontent.com/86700191/148207599-f6410e99-0504-429c-9429-442a203bfc66.PNG)
 
 ## 6. 참고자료(사이트)
- [PyTorch 공식 설명](https://pytorch.org/docs/stable/index.html)
- [RNN & LSTM 설명과 Base 코드](https://cnvrg.io/pytorch-lstm/?gclid=Cj0KCQiA6t6ABhDMARIsAONIYyxsIXn6G6EcMLhGnPDxnsKiv3zLU49TRMxsyTPXZmOV3E-Hh4xeI2EaAugLEALw_wcB)
- [LSTM 은닉층 개수에 따른 모델의 정확성 비교](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/)
- [LSTM 은닉노드 개수에 따른 모델의 정확성 비교](https://forensics.tistory.com/22)