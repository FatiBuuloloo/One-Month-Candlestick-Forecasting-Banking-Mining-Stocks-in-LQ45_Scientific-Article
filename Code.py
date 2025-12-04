import pandas as pd
import os
import random,string
import math
import kaleido
from datetime import datetime
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential,save_model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout,Input
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Model
import joblib
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error, mean_absolute_percentage_error

class CompanyAwareNormalizer:
    def __init__(self):
        # Format: {company_id: scaler_all}
        self.scalers_all = {}  
        
        # Format: {company_id: {column: scaler}}
        self.scalers_col = {}  

    # Normalisasi semua kolom untuk satu perusahaan
    def fit_transform_all(self, company_id, data): 
        if company_id not in self.scalers_all:
            self.scalers_all[company_id] = MinMaxScaler()
        
        scaler = self.scalers_all[company_id]
        return scaler.fit_transform(data)
    
    # Inverse transform semua kolom untuk perusahaan tertentu
    def inverse_transform_all(self, company_id, scaled_data):       
        if company_id not in self.scalers_all:
            raise ValueError(f"Scaler untuk perusahaan {company_id} tidak ditemukan")
        
        return self.scalers_all[company_id].inverse_transform(scaled_data)
    
    # Normalisasi satu kolom untuk satu perusahaan
    def fit_transform_column(self, company_id, column, data):
        if company_id not in self.scalers_col:
            self.scalers_col[company_id] = {}
            
        if column not in self.scalers_col[company_id]:
            self.scalers_col[company_id][column] = MinMaxScaler()
        
        scaler = self.scalers_col[company_id][column]
        return scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
    
    #Inverse transform satu kolom untuk perusahaan tertentu
    def inverse_transform_column(self, company_id, column, scaled_data):
        if company_id not in self.scalers_col:
            raise ValueError(f"Scaler untuk perusahaan {company_id} tidak ditemukan")
            
        if column not in self.scalers_col[company_id]:
            raise ValueError(f"Kolom {column} tidak dinormalisasi untuk perusahaan {company_id}")
        
        scaler = self.scalers_col[company_id][column]
        return scaler.inverse_transform(scaled_data.reshape(-1, 1)).flatten()

def clean_data(datasets):
    datasets.columns = datasets.columns.str.strip()  
    datasets = datasets.drop(columns=["Volume", "Adj Close"])
    datasets = datasets[~datasets["Open"].str.contains("Dividend", na=False)]
    datasets = datasets[~datasets["Open"].str.contains("Stock Splits", na=False)]
    datasets["Open"] = datasets["Open"].str.replace(",", "", regex=True).astype(float)
    cols_to_convert = ["Open", "High", "Low", "Close"]
    for col in cols_to_convert:
        datasets[col] = datasets[col].astype(str).str.replace(",", "", regex=True).astype(float)
        datasets[col] = np.round(datasets[col])
    datasets = datasets.dropna()
    datasets = datasets[::-1].reset_index(drop=True)
    return datasets

def indikator(data):
    data = data.copy()  
    data[["Open", "High", "Low", "Close"]] = data[["Open", "High", "Low", "Close"]].replace(",", "", regex=True)

    rsi = ta.rsi(data["Close"])
    macd = ta.macd(data["Close"]).iloc[:, [0, 1]]
    atr = ta.atr(data["High"], data["Low"], data["Close"])
    bb = ta.bbands(data["Close"]).iloc[:, [1, 3, 4]]
    willr = ta.willr(data["High"], data["Low"], data["Close"])
    stoch = ta.stoch(data["High"], data["Low"], data["Close"])
    roc = ta.roc(data["Close"])

    data_gabung = pd.concat([data, rsi, macd, atr, bb, willr, stoch, roc], axis=1)

    data_gabung = data_gabung.dropna()
    return data_gabung

def metric_evaluationn(aktual, prediksi, companies, kategori):
    results = []
    
    for i in range(len(companies)):
        rmse_value = np.sqrt(mean_squared_error(aktual[i], prediksi[i]))
        mape_value = mean_absolute_percentage_error(aktual[i], prediksi[i]) * 100  

        mean_actual = np.mean(aktual[i])  
        nrmse_value = (rmse_value / mean_actual) * 100  

        results.append((companies[i], rmse_value, nrmse_value, mape_value))

    df = pd.DataFrame(results, columns=["Perusahaan", f"RMSE {kategori}", f"NRMSE {kategori} (%)", f"MAPE {kategori}"])
    return df

def plot_candlestick(df, nama_perusahaan,filename):
    if not isinstance(df, pd.DataFrame) or df.empty:
        print(f"Data untuk {nama_perusahaan} tidak valid atau kosong! Skipping...")
        return
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df['Tanggal'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candlestick"
    ))

    # Layout agar lebih rapi
    fig.update_layout(
        title=f"Candlestick Chart - {nama_perusahaan}",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        xaxis_rangeslider_visible=False,  
        template="plotly_dark"  
    )

    try:
        fig.write_image(filename, scale=2)  
        print(f"Gambar candlestick {nama_perusahaan} disimpan di {filename}")
    except Exception as e:
        print(f"Error menyimpan gambar {nama_perusahaan}: {e}")

    fig.show()
filenames = ["ADRO.csv", "ANTM.csv", "INCO.csv", "ITMG.csv", "MEDC.csv", 
             "PGAS.csv", "PTBA.csv", "BBCA.csv", "BBNI.csv", "BBTN.csv", "BMRI.csv"]

set_data = [pd.read_csv(file) for file in filenames]

companies = [os.path.splitext(file)[0] for file in filenames]
cleaned_data = [clean_data(dataset) for dataset in set_data]
indikator_data = [indikator(data) for data in cleaned_data]

list_tanggal = [indikator_data[i].Date for i in range (len(companies))]
list_tanggal = [np.array(list_tanggal[i]).reshape(-1, 1) for i in range (len(companies))]
tanggal_unproses_perusahaan = []
tanggal_unproses_training = []
tanggal_unproses_validasi = []
tanggal_unproses_tes = []

seq_len = 14
n_future = 1

for j in range(len(companies)):  
    tanggal_unproses = []  
    
    for i in range(seq_len, len(indikator_data[j]) - n_future + 1):
        
        tanggal_str = list_tanggal[j][i + n_future - 1, 0]

        if isinstance(tanggal_str, str):  
            tanggal_date = datetime.strptime(tanggal_str, "%d-%b-%y").date()  
        else:
            tanggal_date = tanggal_str  
        tanggal_unproses.append(tanggal_date)  
    n = len(tanggal_unproses)
    train_end = int(0.8 * n)
    val_end = train_end + int(0.1 * n)    

    tanggal_unproses_perusahaan.append(tanggal_unproses)  
    tanggal_unproses_training.append(tanggal_unproses[:train_end])
    tanggal_unproses_validasi.append(tanggal_unproses[train_end:val_end])
    tanggal_unproses_tes.append(tanggal_unproses[val_end:])

for j, perusahaan in enumerate(companies):
    print(f"Perusahaan: {perusahaan}, Tanggal Target: {len(tanggal_unproses_perusahaan[j])}, Training : {len(tanggal_unproses_training[j])}, Validasi : {len(tanggal_unproses_validasi[j])}, Tes : {len(tanggal_unproses_tes[j])}")  

indikator_data = [indikator_data[i].drop(columns="Date") for i in range (len(indikator_data))]

column = ["Open","High","Low","Close"]

normalizer = CompanyAwareNormalizer()

scaled_data_all = {}
scaled_data_open = {}
scaled_data_high = {}
scaled_data_low = {}
scaled_data_close = {}

for company, data in zip(companies, indikator_data):
    scaled_all = normalizer.fit_transform_all(company, data)
    scaled_data_all[company] = scaled_all 
    
    scaled_data_open[company] = normalizer.fit_transform_column(company, "Open", data["Open"])
    scaled_data_high[company] = normalizer.fit_transform_column(company, "High", data["High"])
    scaled_data_low[company] = normalizer.fit_transform_column(company, "Low", data["Low"])
    scaled_data_close[company] = normalizer.fit_transform_column(company, "Close", data["Close"])

onehot_encoder = OneHotEncoder(sparse_output=False)
company_ids = np.array(companies).reshape(-1, 1)
onehot_encoder.fit(company_ids) 
encoded_companies = onehot_encoder.transform(company_ids)

scaled_data = {}
scaled_open = {}
scaled_high = {}
scaled_low = {}
scaled_close = {}

if len(companies)==len(scaled_data_all):   
    for i in range (len(companies)): 
        scaled_data[companies[i]] = scaled_data_all[companies[i]]
else :
    print("Jumlah Perusahaan tidak sama dengan jumlah data indikator")

X_train_all, X_val_all, X_test_all = [], [], []
y1_train_all, y1_val_all, y1_test_all = [], [], []
y2_train_all, y2_val_all, y2_test_all = [], [], []
y3_train_all, y3_val_all, y3_test_all = [], [], []
y4_train_all, y4_val_all, y4_test_all = [], [], []

seq_len = 14
n_future = 1
for company, data in scaled_data.items():
    
    X_company, y1_company, y2_company, y3_company, y4_company= [], [], [], [], []
    for i in range(seq_len , len(data) - n_future +1):
        X_company.append(data[i - seq_len:i, 0:data.shape[1]])
        y1_company.append(data[i + n_future - 1:i + n_future, 0])
        y2_company.append(data[i + n_future - 1:i + n_future, 1])
        y3_company.append(data[i + n_future - 1:i + n_future, 2])
        y4_company.append(data[i + n_future - 1:i + n_future, 3])
    X_company = np.array(X_company)
    y1_company = np.array(y1_company)
    y2_company = np.array(y2_company)
    y3_company = np.array(y3_company)
    y4_company = np.array(y4_company)
    
    company_id = onehot_encoder.transform([[company]])
    company_id_repeated = np.repeat(company_id, seq_len, axis=0)
    
    X_company_combined = []
    for seq in X_company:
        seq_with_id = np.hstack([seq, company_id_repeated])
        X_company_combined.append(seq_with_id)
    
    X_company_combined = np.array(X_company_combined)
    
    n = len(X_company_combined)
    train_end = int(0.8 * n)
    val_end = train_end + int(0.1 * n)

    X_train_company = X_company_combined[:train_end]
    X_val_company = X_company_combined[train_end:val_end]
    X_test_company = X_company_combined[val_end:]
    
    y1_train_company, y1_val_company, y1_test_company = y1_company[:train_end], y1_company[train_end:val_end], y1_company[val_end:]
    y2_train_company, y2_val_company, y2_test_company = y2_company[:train_end], y2_company[train_end:val_end], y2_company[val_end:]
    y3_train_company, y3_val_company, y3_test_company = y3_company[:train_end], y3_company[train_end:val_end], y3_company[val_end:]
    y4_train_company, y4_val_company, y4_test_company = y4_company[:train_end], y4_company[train_end:val_end], y4_company[val_end:]

    X_train_all.append(X_train_company)
    X_val_all.append(X_val_company)
    X_test_all.append(X_test_company)
    
    y1_train_all.append(y1_train_company), y1_val_all.append(y1_val_company), y1_test_all.append(y1_test_company)
    y2_train_all.append(y2_train_company), y2_val_all.append(y2_val_company), y2_test_all.append(y2_test_company)
    y3_train_all.append(y3_train_company), y3_val_all.append(y3_val_company), y3_test_all.append(y3_test_company)
    y4_train_all.append(y4_train_company), y4_val_all.append(y4_val_company), y4_test_all.append(y4_test_company)

X_train = np.concatenate(X_train_all, axis=0)
X_val = np.concatenate(X_val_all, axis=0)
X_test = np.concatenate(X_test_all, axis=0)

y1_train, y1_val, y1_test = np.concatenate(y1_train_all, axis=0), np.concatenate(y1_val_all, axis=0), np.concatenate(y1_test_all, axis=0)
y2_train, y2_val, y2_test = np.concatenate(y2_train_all, axis=0), np.concatenate(y2_val_all, axis=0), np.concatenate(y2_test_all, axis=0)
y3_train, y3_val, y3_test = np.concatenate(y3_train_all, axis=0), np.concatenate(y3_val_all, axis=0), np.concatenate(y3_test_all, axis=0)
y4_train, y4_val, y4_test = np.concatenate(y4_train_all, axis=0), np.concatenate(y4_val_all, axis=0), np.concatenate(y4_test_all, axis=0)

print(f"Training shape: X = {X_train.shape}, y_Open = {y1_train.shape}, y_High = {y2_train.shape}, y_Low = {y3_train.shape}, y_Close = {y4_train.shape}")
print(f"Validation shape: X = {X_val.shape}, y_Open =  {y1_val.shape}, y_High = {y2_val.shape}, y_Low = {y3_val.shape}, y_Close = {y4_val.shape}")
print(f"Test shape: X = {X_test.shape}, y_Open =  {y1_test.shape}, y_High = {y2_test.shape}, y_Low = {y3_test.shape}, y_Close = {y4_test.shape}")

model_jaringan_open = Sequential()
model_jaringan_open.add(LSTM(100, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model_jaringan_open.add(LSTM(50, activation='relu', return_sequences=False))
model_jaringan_open.add(Dropout(0.2))
model_jaringan_open.add(Dense(y4_train.shape[1]))
model_jaringan_open.compile(optimizer='adam', loss='log_cosh')
model_jaringan_open.summary()

model_jaringan_high = Sequential()
model_jaringan_high.add(LSTM(150, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model_jaringan_high.add(LSTM(100, activation='relu', return_sequences=False))
model_jaringan_high.add(Dropout(0.2))
model_jaringan_high.add(Dense(y4_train.shape[1]))
model_jaringan_high.compile(optimizer='adam', loss='log_cosh')
model_jaringan_high.summary()

model_jaringan_low = Sequential()
model_jaringan_low.add(LSTM(120, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model_jaringan_low.add(LSTM(80, activation='relu', return_sequences=False))
model_jaringan_low.add(Dropout(0.2))
model_jaringan_low.add(Dense(y4_train.shape[1]))
model_jaringan_low.compile(optimizer='adam', loss='log_cosh')
model_jaringan_low.summary()

model_jaringan_close = Sequential()
model_jaringan_close.add(LSTM(150, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model_jaringan_close.add(LSTM(100, activation='relu', return_sequences=False))
model_jaringan_close.add(Dropout(0.2))
model_jaringan_close.add(Dense(y4_train.shape[1]))
model_jaringan_close.compile(optimizer='adam', loss='log_cosh')
model_jaringan_close.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
history1 = model_jaringan_open.fit(
    X_train, y1_train,
    validation_data=(X_val, y1_val),
    epochs=150,
    batch_size=32,
    callbacks=[early_stop]
)
history2 = model_jaringan_high.fit(
    X_train, y2_train,
    validation_data=(X_val, y2_val),
    epochs=150,
    batch_size=32,
    callbacks=[early_stop]
)
history3 = model_jaringan_low.fit(
    X_train, y3_train,
    validation_data=(X_val, y3_val),
    epochs=150,
    batch_size=32,
    callbacks=[early_stop]
)
history4 = model_jaringan_close.fit(
    X_train, y4_train,
    validation_data=(X_val, y4_val),
    epochs=150,
    batch_size=64,
    callbacks=[early_stop]
)
save_model(model_jaringan_open, 'lstm_model_open.h5')
save_model(model_jaringan_high, 'lstm_model_high.h5')
save_model(model_jaringan_low, 'lstm_model_low.h5')
save_model(model_jaringan_close, 'lstm_model_close.h5')

data_asli_open = []
data_prediksi_open = []

for i in range(len(companies)):  
    data_asli_open.append(normalizer.inverse_transform_column(companies[i], "Open", y1_test_all[i]))
    data_prediksi_open.append(normalizer.inverse_transform_column(companies[i], "Open", prediksi_tes_open[i]))

metric_evaluation_open = metric_evaluationn(data_asli_open, data_prediksi_open, companies, "Open")
metric_evaluation_open

data_asli_high = []
data_prediksi_high = []

for i in range(len(companies)):
    data_asli_high.append(normalizer.inverse_transform_column(companies[i], "High", y2_test_all[i]))
    data_prediksi_high.append(normalizer.inverse_transform_column(companies[i], "High", prediksi_tes_high[i]))

metric_evaluation_high = metric_evaluationn(data_asli_high, data_prediksi_high, companies, "High")
metric_evaluation_high

data_asli_low = []
data_prediksi_low = []

for i in range(len(companies)):  
    data_asli_low.append(normalizer.inverse_transform_column(companies[i], "Low", y3_test_all[i]))
    data_prediksi_low.append(normalizer.inverse_transform_column(companies[i], "Low", prediksi_tes_high[i]))

metric_evaluation_low = metric_evaluationn(data_asli_low, data_prediksi_low, companies, "Low")
metric_evaluation_low

data_asli_close = []
data_prediksi_close = []

for i in range(len(companies)):  
    data_asli_close.append(normalizer.inverse_transform_column(companies[i], "Close", y4_test_all[i]))
    data_prediksi_close.append(normalizer.inverse_transform_column(companies[i], "Close", prediksi_tes_close[i]))

metric_evaluation_close = metric_evaluationn(data_asli_close, data_prediksi_close, companies, "Close")
metric_evaluation_close

days = 22
predictions_all = {company: [] for company in companies}
historical_data = {company: cleaned_data[i].copy() for i, company in enumerate(companies)}
df_output = [pd.DataFrame(columns=["Open", "High", "Low", "Close"]) for _ in range(len(companies))]

for h in range (len(companies)):
    for j in range(days):
        company = companies[h] 
        data_processed = historical_data[company].copy()
        
        data_processed_indikator = indikator(data_processed)

        scaled_data_processed = normalizer.fit_transform_all(company, data_processed_indikator)

        company_id_after = onehot_encoder.transform([[company]])
        company_id_after_repeated = np.repeat(company_id_after, seq_len, axis=0)

        X_company_after = scaled_data_processed[-seq_len:]
    
        X_company_combined_after = []
        for seq in X_company_after:
            seq = seq.reshape(1, -1)  
            seq_with_id = np.hstack([seq, company_id_after_repeated[:1]])  
            X_company_combined_after.append(seq_with_id)
    
        X_company_combined_after = np.array(X_company_combined_after).squeeze()
        X_company_combined_after = X_company_combined_after.reshape(1, 14, 26)
    
        pred_open = model_jaringan_open.predict(X_company_combined_after)
        pred_high = model_jaringan_high.predict(X_company_combined_after)
        pred_low = model_jaringan_low.predict(X_company_combined_after)
        pred_close = model_jaringan_close.predict(X_company_combined_after)
    
        new_row = pd.DataFrame({
            "Open": normalizer.inverse_transform_column(company, "Open", pred_open),  
            "High": normalizer.inverse_transform_column(company, "High", pred_high),  
            "Low": normalizer.inverse_transform_column(company, "Low", pred_low),  
            "Close": normalizer.inverse_transform_column(company, "Close", pred_close)  
        })
    
        historical_data[company] = pd.concat([historical_data[company], new_row], ignore_index=True)
    
        
        df_output.append(new_row)

hasil_akhir = [historical_data[companies[i]][-23:] for i in range (len(companies))]
tanggal_akhir = [set_data[i].Date.iloc[:1] for i in range(len(companies))]  
tanggal_akhir = [pd.to_datetime(tgl.iloc[0], dayfirst=True) for tgl in tanggal_akhir]  

list_df_perdagangan = []

# Iterasi setiap tanggal akhir untuk menghitung 1 bulan kerja (Bursa Efek Indonesia)
for idx, tgl in enumerate(sorted(tanggal_akhir)):  
    nama_perusahaan = f"{companies[idx]}"  
    tanggal_perdagangan = pd.date_range(start=tgl, periods=days+1, freq="B")  

    df = pd.DataFrame({"Tanggal": tanggal_perdagangan})

    list_df_perdagangan.append(df)

hasil_final = [
    pd.concat([list_df_perdagangan[i].reset_index(drop=True), hasil_akhir[i].reset_index(drop=True)], axis=1)
    for i in range(len(list_df_perdagangan))
]

for i in range(len(hasil_final)):
    print(hasil_final[i].columns)

output_folderr = "C:/Users/ASUS/Downloads/candlestick_plots_baru"
if not os.path.exists(output_folderr):
    os.makedirs(output_folderr)
for i in range(len(companies)):
    filename = os.path.join(output_folderr, f"candlestick_{companies[i]}.png")  # Simpan di dalam folder
    plot_candlestick(hasil_final[i], companies[i], filename)
