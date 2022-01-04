import datetime

import pandas as pd
import pandas_datareader.data as pdr

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Combobox

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# matplotlib 한글 폰트 문제해결
path = 'font file'
fontprop = fm.FontProperties(fname=path)

# Firebase 연동
cred = credentials.Certificate("your SDK json file")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Firebase에 저장된 종목의 이름과 종목코드 수신
doc_ref = db.collection(u'KOSPI').document(u'issue')
doc = doc_ref.get()

# GPU 준비(없다면 CPU 준비)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device cpu or gpu

# tkinter window창 실행
window = Tk()

window.title("KOSPI 100_STOCK_ai")
window.geometry("600x600+100+50")


# LSTM 네트워크
class LSTM_Model(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM_Model, self).__init__()
        self.num_classes = num_classes  # 출력 class의 개수
        self.num_layers = num_layers  # LSTM 층의 개수
        self.input_size = input_size  # 입력에 대한 feature 차원
        self.hidden_size = hidden_size  # hidden feature 차원

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)  # lstm
        self.fc = nn.Linear(hidden_size, num_classes)  # fully connected

        self.relu = nn.ReLU()  # sigmoid 의 정확도문제를 해결하기 위한 함수(Rectified Linear Unit)

    def forward(self, x):
        # 각 state의 초기값은 0으로 구성
        h_0 = torch.Tensor(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # hidden state 초기값
        c_0 = torch.Tensor(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # cell state 초기값

        # LSTM 출력을 통한 입력 전파
        (hn, cn) = self.lstm(x, (h_0, c_0))

        hn = hn.view(-1, self.hidden_size)  # fc를 위한 데이터 재배치

        out = self.relu(hn)     # 활성화 함수
        out = self.fc(out)  # 은닉층 -> 출력층

        return out


# 주식데이터 불러오기
def Stock_read(code):
    start = (1980, 1, 4)  # 1980년 1월 4일 = KOSPI 지수 기준날짜
    start = datetime.datetime(*start)
    end = datetime.date.today()  # 현재 오늘 날짜

    # yahoo finance에서 주식데이터 불러오기
    df = pdr.DataReader(code, 'yahoo', start, end)
    return df


# 학습 후 예측
def Predict_stock(code, num_predict):
    ss = StandardScaler()  # 평균 0, 분산 1인 데이터로 정규화
    mm = MinMaxScaler()  # 최대/최소값이 각각 1, 0인 데이터로 정규화

    data = Stock_read(code)
    data_X = data.drop(columns='Volume')  # 거래량 제거
    data_Y = data.iloc[:, 5:6]  # Adj Close값(주식의 분할, 배당, 배분 등을 고려해 조정한 종가)만 가져오기

    # X, Y의 데이터를 전처리, numpy 형태 (훈련데이터라 fit적용)
    X_ss_fit = ss.fit_transform(data_X)
    Y_mm_fit = mm.fit_transform(data_Y)

    # 장기, 단기에 따라 훈련데이터의 슬라이싱이 달라짐
    X_train = X_ss_fit[:-num_predict, :]
    Y_train = Y_mm_fit[:-num_predict, :]

    # numpy 형태에서 Tensor 형태로 변환, 크기는 (전체 행 개수 - 예측할 날의 개수 , 5)
    X_train_tensors = torch.Tensor(X_train)
    Y_train_tensors = torch.Tensor(Y_train)

    # Tensor 형태를 (전체 행 개수 - 예측할 날의 개수,1,5)로 재배치
    X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))

    num_epochs = 10000  # 10000 epochs
    learning_rate = 0.0001  # 0.0001 lr

    # parameter of lstm
    input_size = 5
    hidden_size = 10
    num_layers = 1
    num_classes = 1

    lstm = LSTM_Model(num_classes, input_size, hidden_size, num_layers).to(device)

    loss_function = nn.MSELoss()  # 손실함수 MSE 사용
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)  # optimizer의 종류중 adam 사용

    # 학습
    for epoch in range(num_epochs):
        outputs = lstm.forward(X_train_tensors_final.to(device))  # 순전파 (4500,1)

        # 역전파 사용
        optimizer.zero_grad()  # 한번 학습시 gradients값을 0으로 초기화

        # 손실함수의 loss값 구하기
        loss = loss_function(outputs, Y_train_tensors.to(device))
        loss.backward()

        optimizer.step()  # 역전파를 통한 loss값 업데이트
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    # 테스트 데이터 정규화
    X_test = ss.transform(data_X)

    # 테스트 데이터 예측일 수에 맞기 슬라이싱
    X_test = X_test[-num_predict:, :]
    Y_test = data_Y.iloc[-num_predict:, :]

    # numpy -> Tensor
    X_test_tensors = torch.Tensor(X_test)

    # Tensor 형태를 (예측할 날의 개수,1,5)로 재배치
    X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))
    # 테스트 데이터 예측
    predict_test = lstm(X_test_tensors_final.to(device))

    # Tensor -> numpy
    # 메모리측면에서 텐서는 GPU에서 연산처리, numpy는 cpu에서 처리하기 떄문에 gpu->cpu로 옮길 필요가 있음
    data_predict = predict_test.data.detach().cpu().numpy()

    # 예측 출력값은 y의 데이터와 비교하기 때문에 mm역변환
    data_predict = mm.inverse_transform(data_predict)

    # numpy 형태의 예측데이터를 인덱스를 Date로 하여 Pandas의 데이터프레임으로 변환
    data_predict = pd.DataFrame(data=data_predict, index=Y_test.index)

    return data_predict, Y_test


# 버튼 이벤트
def select():
    str_entry = entry.get()
    str_combobox = combobox.get()
    period = period_var.get()

    # 검색 창이 우선적, 파이어베이스에 등록된 종목과 입력된 종목을 비교하여 종목코드를 가져온다.
    if str_entry:
        issue_code = doc.to_dict().get(str_entry)
        if issue_code is not None:  # issue_code를 찾지못해 null값 조차도 가지고 있지 않을 수 있어 is not (나중에 지울것)
            Title = str_entry

    else:
        issue_code = doc.to_dict().get(str_combobox)
        if issue_code is not None:
            Title = str_combobox

    plt.clf()

    try:
        predict, real = Predict_stock(issue_code, period)  # 예측 데이터로 교체

        # x, y축 글자 크기조절
        plt.rc('xtick', labelsize=6)
        plt.rc('ytick', labelsize=8)

        # 실제주가와 예측주가 그래프 그리기
        plt.plot(real, label='Real Data')
        plt.plot(predict, label='Predict Data')
        plt.title(Title, fontproperties=fontprop)
        plt.xlabel('date')
        plt.ylabel('price')
        plt.grid(True)  # 격자표시
        plt.legend()  # 라벨표시를 위한 범례

        line1 = FigureCanvasTkAgg(fig1, window)  # pyplot을 Tkinter와 연동하여 표시
        line1.get_tk_widget().place(x=25, y=120)

    except RuntimeError:  # 기간을 선택하지 않으면 에러발생
        messagebox.showerror("Error", "예측할 기간을 선택해주세요!!")

    except TypeError:   # 등록되지 않은 종목을 검색시 에러발생
        messagebox.showerror("Error", "해당 주식은 데이터가 없습니다!!\n콤보박스를 확인해주세요.")

    entry.delete(0, len(str_entry))
    combobox.set("목록 선택")


# 창을 닫을시 Tcl 인터프리터 종료
def close():
    if messagebox.askokcancel("Exit", "종료하시겠습니까?"):
        window.quit()
        window.destroy()


# 초기화면 코스피 차트
data_init = Stock_read('^KS11')  # KOSPI의 코드는 ^KS11

fig1 = plt.figure(figsize=(5.5, 4), dpi=100)
data_init.Close.plot(grid=True)  # KOSPI 데이터프레임 matplotlib 로 뛰우기
plt.title('KOSPI')
plt.xlabel('year')
plt.ylabel('price index')

# pyplot을 Tkinter와 연동
line1 = FigureCanvasTkAgg(fig1, window)
line1.get_tk_widget().place(x=25, y=120)

# 검색창
entry = Entry(window, width=53)
entry.place(x=50, y=22.5)

# 콤보박스, 파이어베이스에서 가져온 종목의 딕셔너리의 키값만 리스트화하여 값으로 보여준다.
combobox = Combobox(window, width=50, height=15, values=list(doc.to_dict().keys()))
combobox.place(x=50, y=50)
combobox.set("목록 선택")

# 단기, 장기 라디오버튼(1개월 = 22일로 계산)
period_var = IntVar()
radiobutton1 = Radiobutton(window, text="단기예측 (3개월)", value=66, variable=period_var)
radiobutton2 = Radiobutton(window, text="장기예측 (12개월)", value=264, variable=period_var)
radiobutton1.place(x=75, y=75)
radiobutton2.place(x=250, y=75)

# 예측 실행버튼
button = Button(window, text="예측하기", width=10, height=5, command=select)
button.place(x=475, y=15)

window.wm_protocol("WM_DELETE_WINDOW", close)  # 창을 닫을때 프로토콜 핸들러 실행
window.mainloop()
