import datetime
import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_DISPONIVEL = True
except:
    XGBOOST_DISPONIVEL = False

try:
    import lightgbm as lgb
    LIGHTGBM_DISPONIVEL = True
except:
    LIGHTGBM_DISPONIVEL = False

try:
    import catboost as cb
    CATBOOST_DISPONIVEL = True
except:
    CATBOOST_DISPONIVEL = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_DISPONIVEL = True
except:
    OPTUNA_DISPONIVEL = False
try:
    from arch import arch_model
    GARCH_DISPONIVEL = True
except:
    GARCH_DISPONIVEL = False



try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (LSTM, Dense, Dropout, Bidirectional, 
                                         GRU, Conv1D, MaxPooling1D, Flatten,
                                         MultiHeadAttention, LayerNormalization,
                                         Input, Embedding, GlobalAveragePooling1D)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    
    # Configurar GPU (se disponível)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    TENSORFLOW_DISPONIVEL = True
except:
    TENSORFLOW_DISPONIVEL = False
    
try:
    import torch
    import torch.nn as nn
    PYTORCH_DISPONIVEL = True
except:
    PYTORCH_DISPONIVEL = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    import io
    from datetime import datetime as dt
    PDF_DISPONIVEL = True
except:
    PDF_DISPONIVEL = False

st.set_page_config(page_title="QuantumStock", layout="wide")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    .main { background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); color: #1a237e; }
    .stMetric { background: linear-gradient(135deg, #ffffff 0%, #e3f2fd 100%); padding: 20px; border-radius: 12px; border: 2px solid #64b5f6; box-shadow: 0 4px 12px rgba(100, 181, 246, 0.3); }
    .stMetric label { color: #1565c0 !important; font-weight: 600 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #0d47a1 !important; font-size: 28px !important; font-weight: bold !important; }
    .stButton > button { background: linear-gradient(135deg, #64b5f6 0%, #42a5f5 100%); color: white; border: none; border-radius: 10px; padding: 12px 32px; font-weight: bold; }
    h1 { color: #0d47a1 !important; font-weight: 800 !important; }
    h2 { color: #1565c0 !important; }
    .stTabs [data-baseweb="tab-list"] { background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 10px; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #ffffff; color: #1565c0; font-weight: 600; border-radius: 8px; padding: 12px 24px; border: 2px solid #90caf9; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #64b5f6 0%, #42a5f5 100%) !important; color: white !important; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #e3f2fd 0%, #bbdefb 100%); }
    div[data-testid="stSidebar"] * { color: #0d47a1 !important; }
    </style>
""", unsafe_allow_html=True)

st.title('🚀 QuantumStock ')


# Sidebar
st.sidebar.header("⚙️ Configurações")
st.sidebar.info('🏆QuantumStock ')

empresas_disponiveis = {
    "Petrobras": "PETR4.SA", "Vale": "VALE3.SA", "Itaú": "ITUB4.SA",
    "Ambev": "ABEV3.SA", "Bradesco": "BBDC4.SA", "Magalu": "MGLU3.SA",
    "B3": "B3SA3.SA", "Eletrobras": "ELET3.SA", "WEG": "WEGE3.SA", "Suzano": "SUZB3.SA"
}

empresa_selecionada = st.sidebar.selectbox("🏢 Empresa", list(empresas_disponiveis.keys()))
ticker = empresas_disponiveis[empresa_selecionada]

num_dias = st.sidebar.number_input('📅 Dias Histórico', value=730, min_value=365, max_value=1095, step=365)

today = datetime.date.today()
data_padrao = today - datetime.timedelta(days=num_dias)
start_date = st.sidebar.date_input('Data Inicial', value=data_padrao)
end_date = st.sidebar.date_input('Data Final', today)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Modelos")

usar_modelos_regime = st.sidebar.checkbox("🎯 Modelos por Regime", value=True)
usar_optuna = st.sidebar.checkbox("🔥 Optuna AutoML", value=False, disabled=not OPTUNA_DISPONIVEL)
usar_stacking = st.sidebar.checkbox("📚 Stacking Ensemble", value=True)
usar_lightgbm = st.sidebar.checkbox("⚡ LightGBM", value=LIGHTGBM_DISPONIVEL, disabled=not LIGHTGBM_DISPONIVEL)
usar_catboost = st.sidebar.checkbox("🐱 CatBoost", value=False, disabled=not CATBOOST_DISPONIVEL)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Features")

usar_sentimento = st.sidebar.checkbox("📰 Google News", value=True)
usar_alertas = st.sidebar.checkbox("🔔 Alertas", value=True)
st.sidebar.markdown("---")
st.sidebar.subheader("🔥 NOVO - Alto Impacto")

usar_garch = st.sidebar.checkbox("📊 GARCH (Volatilidade)", value=GARCH_DISPONIVEL, disabled=not GARCH_DISPONIVEL)
usar_candlestick = st.sidebar.checkbox("🕯️ Padrões Candlestick", value=True)
usar_suporte_resistencia = st.sidebar.checkbox("📈 Support & Resistance", value=True)
usar_correlacao = st.sidebar.checkbox("🌐 Correlação Índices", value=True)

if not GARCH_DISPONIVEL:
    st.sidebar.warning("⚠️ pip install arch")

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Relatórios")

gerar_pdf = st.sidebar.checkbox("📄 Gerar PDF", value=PDF_DISPONIVEL, disabled=not PDF_DISPONIVEL)

if not PDF_DISPONIVEL:
    st.sidebar.warning("⚠️ pip install reportlab")


st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Deep Learning")

usar_transformer = st.sidebar.checkbox("🤖 Transformer", value=TENSORFLOW_DISPONIVEL, disabled=not TENSORFLOW_DISPONIVEL)
usar_bilstm = st.sidebar.checkbox("⚡ BiLSTM", value=TENSORFLOW_DISPONIVEL, disabled=not TENSORFLOW_DISPONIVEL)
usar_gru = st.sidebar.checkbox("🔄 GRU", value=TENSORFLOW_DISPONIVEL, disabled=not TENSORFLOW_DISPONIVEL)
usar_cnn_lstm = st.sidebar.checkbox("🎯 CNN-LSTM", value=False, disabled=not TENSORFLOW_DISPONIVEL)
usar_ensemble_neural = st.sidebar.checkbox("🏆 Ensemble Neural", value=TENSORFLOW_DISPONIVEL, disabled=not TENSORFLOW_DISPONIVEL)

if TENSORFLOW_DISPONIVEL:
    st.sidebar.markdown("**Configurações:**")
    seq_length = st.sidebar.slider("📏 Sequência (dias)", 30, 120, 60)
    epochs_dl = st.sidebar.slider("🔁 Epochs", 20, 100, 50)
    batch_size_dl = st.sidebar.slider("📦 Batch Size", 16, 128, 32)
else:
    st.sidebar.warning("⚠️ pip install tensorflow")
    seq_length = 60
    epochs_dl = 50
    batch_size_dl = 32



if not OPTUNA_DISPONIVEL:
    st.sidebar.warning("⚠️ pip install optuna")
if not LIGHTGBM_DISPONIVEL:
    st.sidebar.warning("⚠️ pip install lightgbm")
if not CATBOOST_DISPONIVEL:
    st.sidebar.warning("⚠️ pip install catboost")


@st.cache_data(ttl=3600)
def download_dados_cached(ticker, start_date, end_date):
    try:
        ativo = yf.Ticker(ticker)
        df = ativo.history(start=start_date, end=end_date)
        return df
    except Exception as e:
        st.error(f"Erro: {e}")
        return pd.DataFrame()


def analisar_sentimento_google_news(empresa_nome, dias=7):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        usar_vader = True
    except:
        usar_vader = False
    
    try:
        import feedparser
        query = empresa_nome.replace(' ', '+')
        rss_url = f"https://news.google.com/rss/search?q={query}+when:{dias}d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
        feed = feedparser.parse(rss_url)
        
        sentimentos = []
        noticias = []
        
        if feed.entries:
            for entry in feed.entries[:10]:
                titulo = entry.title
                if usar_vader and titulo:
                    scores = analyzer.polarity_scores(titulo)
                    sentimento = scores['compound']
                else:
                    sentimento = 0.0
                sentimentos.append(sentimento)
                noticias.append({'titulo': titulo, 'sentimento': sentimento, 'link': entry.link, 'data': entry.get('published', '')})
            
            if sentimentos:
                pesos = np.linspace(0.5, 1.0, len(sentimentos))
                pesos = pesos / pesos.sum()
                return np.average(sentimentos, weights=pesos), noticias
        return 0.0, []
    except:
        return 0.0, []


# ============ NOVO V12: GARCH ============
def calcular_garch_volatilidade(retornos, horizonte=5):
    """GARCH(1,1) para prever volatilidade futura"""
    if not GARCH_DISPONIVEL:
        return None
    
    try:
        retornos_clean = retornos.dropna() * 100
        
        if len(retornos_clean) < 100:
            return None
        
        model = arch_model(retornos_clean, vol='Garch', p=1, q=1)
        model_fit = model.fit(disp='off')
        forecast = model_fit.forecast(horizon=horizonte)
        vol_forecast = np.sqrt(forecast.variance.values[-1, :])
        
        return {
            'vol_atual': model_fit.conditional_volatility.iloc[-1],
            'vol_forecast': vol_forecast,
            'vol_media': vol_forecast.mean(),
            'vol_max': vol_forecast.max(),
            'vol_min': vol_forecast.min()
        }
    except:
        return None


# ============ NOVO V12: CANDLESTICK PATTERNS ============
def detectar_padroes_candlestick(df, lookback=100):
    """Detecta padrões de candlestick clássicos"""
    
    df_recent = df.tail(lookback).copy()
    padroes = []
    
    df_recent['body'] = abs(df_recent['Close'] - df_recent['Open'])
    df_recent['upper_shadow'] = df_recent['High'] - df_recent[['Close', 'Open']].max(axis=1)
    df_recent['lower_shadow'] = df_recent[['Close', 'Open']].min(axis=1) - df_recent['Low']
    df_recent['range'] = df_recent['High'] - df_recent['Low']
    
    if len(df_recent) >= 3:
        c1 = df_recent.iloc[-3]
        c2 = df_recent.iloc[-2]
        c3 = df_recent.iloc[-1]
        
        # DOJI
        if c3['body'] < c3['range'] * 0.1:
            padroes.append({'padrao': 'Doji', 'tipo': 'NEUTRO', 'confianca': 60})
        
        # HAMMER (bullish)
        if (c3['Close'] > c3['Open'] and 
            c3['lower_shadow'] > c3['body'] * 2 and 
            c3['upper_shadow'] < c3['body'] * 0.5):
            padroes.append({'padrao': 'Hammer', 'tipo': 'COMPRA', 'confianca': 70})
        
        # SHOOTING STAR (bearish)
        if (c3['Open'] > c3['Close'] and 
            c3['upper_shadow'] > c3['body'] * 2 and 
            c3['lower_shadow'] < c3['body'] * 0.5):
            padroes.append({'padrao': 'Shooting Star', 'tipo': 'VENDA', 'confianca': 70})
        
        # ENGULFING BULLISH
        if (c2['Close'] < c2['Open'] and c3['Close'] > c3['Open'] and
            c3['Open'] < c2['Close'] and c3['Close'] > c2['Open']):
            padroes.append({'padrao': 'Bullish Engulfing', 'tipo': 'COMPRA', 'confianca': 80})
        
        # ENGULFING BEARISH
        if (c2['Close'] > c2['Open'] and c3['Close'] < c3['Open'] and
            c3['Open'] > c2['Close'] and c3['Close'] < c2['Open']):
            padroes.append({'padrao': 'Bearish Engulfing', 'tipo': 'VENDA', 'confianca': 80})
        
        # MORNING STAR (bullish)
        if (c1['Close'] < c1['Open'] and
            c2['body'] < c1['body'] * 0.3 and
            c3['Close'] > c3['Open'] and
            c3['Close'] > (c1['Open'] + c1['Close']) / 2):
            padroes.append({'padrao': 'Morning Star', 'tipo': 'COMPRA', 'confianca': 85})
        
        # EVENING STAR (bearish)
        if (c1['Close'] > c1['Open'] and
            c2['body'] < c1['body'] * 0.3 and
            c3['Close'] < c3['Open'] and
            c3['Close'] < (c1['Open'] + c1['Close']) / 2):
            padroes.append({'padrao': 'Evening Star', 'tipo': 'VENDA', 'confianca': 85})
    
    return padroes


# ============ NOVO V12: SUPPORT & RESISTANCE ============
def calcular_suporte_resistencia(df, janela=50, num_niveis=3):
    """Identifica níveis de suporte e resistência automaticamente"""
    
    df_recent = df.tail(janela * 2)
    precos = df_recent['Close'].values
    
    peaks, _ = find_peaks(precos, distance=5)
    valleys, _ = find_peaks(-precos, distance=5)
    
    if len(peaks) > 0:
        resistencias = sorted(precos[peaks], reverse=True)[:num_niveis]
    else:
        resistencias = []
    
    if len(valleys) > 0:
        suportes = sorted(precos[valleys])[:num_niveis]
    else:
        suportes = []
    
    preco_atual = df['Close'].iloc[-1]
    
    # Fibonacci
    if len(df_recent) > 50:
        high = df_recent['High'].max()
        low = df_recent['Low'].min()
        diff = high - low
        
        fib_levels = {
            '0.236': high - 0.236 * diff,
            '0.382': high - 0.382 * diff,
            '0.500': high - 0.500 * diff,
            '0.618': high - 0.618 * diff,
            '0.786': high - 0.786 * diff
        }
    else:
        fib_levels = {}
    
    return {
        'suportes': suportes,
        'resistencias': resistencias,
        'fibonacci': fib_levels,
        'preco_atual': preco_atual,
        'prox_suporte': min(suportes) if suportes else None,
        'prox_resistencia': min([r for r in resistencias if r > preco_atual], default=None) if resistencias else None
    }


# ============ NOVO V12: SIGNAL STRENGTH ============
def calcular_signal_strength(df, previsao, variacao_prevista, padroes_candlestick, sr_levels):
    """Combina múltiplos indicadores em score 0-100"""
    
    score = 50
    sinais = []
    
    # 1. Modelo ML (peso 30)
    if variacao_prevista > 2:
        score += 15
        sinais.append({'fonte': 'Modelo ML', 'sinal': 'COMPRA', 'forca': 15})
    elif variacao_prevista < -2:
        score -= 15
        sinais.append({'fonte': 'Modelo ML', 'sinal': 'VENDA', 'forca': -15})
    
    # 2. RSI (peso 15)
    rsi = RSIIndicator(df['Close'], window=14).rsi().iloc[-1]
    if rsi < 30:
        score += 10
        sinais.append({'fonte': 'RSI', 'sinal': 'COMPRA', 'forca': 10})
    elif rsi > 70:
        score -= 10
        sinais.append({'fonte': 'RSI', 'sinal': 'VENDA', 'forca': -10})
    
    # 3. MACD (peso 15)
    macd = MACD(df['Close'])
    macd_diff = macd.macd_diff().iloc[-1]
    if macd_diff > 0:
        score += 8
        sinais.append({'fonte': 'MACD', 'sinal': 'COMPRA', 'forca': 8})
    elif macd_diff < 0:
        score -= 8
        sinais.append({'fonte': 'MACD', 'sinal': 'VENDA', 'forca': -8})
    
    # 4. Padrões Candlestick (peso 20)
    if padroes_candlestick:
        for padrao in padroes_candlestick:
            peso = padrao['confianca'] / 5
            if padrao['tipo'] == 'COMPRA':
                score += peso
                sinais.append({'fonte': f"Candlestick: {padrao['padrao']}", 'sinal': 'COMPRA', 'forca': peso})
            elif padrao['tipo'] == 'VENDA':
                score -= peso
                sinais.append({'fonte': f"Candlestick: {padrao['padrao']}", 'sinal': 'VENDA', 'forca': -peso})
    
    # 5. Support & Resistance (peso 10)
    preco_atual = df['Close'].iloc[-1]
    if sr_levels.get('prox_resistencia') and preco_atual > sr_levels['prox_resistencia'] * 0.98:
        score -= 7
        sinais.append({'fonte': 'S/R', 'sinal': 'VENDA', 'forca': -7})
    elif sr_levels.get('prox_suporte') and preco_atual < sr_levels['prox_suporte'] * 1.02:
        score += 7
        sinais.append({'fonte': 'S/R', 'sinal': 'COMPRA', 'forca': 7})
    
    # 6. Tendência (peso 10)
    sma20 = df['Close'].rolling(20).mean().iloc[-1]
    sma50 = df['Close'].rolling(50).mean().iloc[-1]
    if sma20 > sma50:
        score += 5
        sinais.append({'fonte': 'Tendência', 'sinal': 'COMPRA', 'forca': 5})
    else:
        score -= 5
        sinais.append({'fonte': 'Tendência', 'sinal': 'VENDA', 'forca': -5})
    
    score = max(0, min(100, score))
    
    if score >= 70:
        recomendacao = "COMPRA FORTE"
        cor = "success"
    elif score >= 55:
        recomendacao = "COMPRA"
        cor = "info"
    elif score <= 30:
        recomendacao = "VENDA FORTE"
        cor = "error"
    elif score <= 45:
        recomendacao = "VENDA"
        cor = "warning"
    else:
        recomendacao = "NEUTRO"
        cor = "secondary"
    
    return {
        'score': score,
        'recomendacao': recomendacao,
        'cor': cor,
        'sinais': sinais
    }


# ============ NOVO V12: CORRELAÇÃO COM ÍNDICES ============
def calcular_correlacao_indices(ticker, start_date, end_date):
    """Calcula correlação com IBOV, S&P500, Dólar"""
    
    indices = {
        'IBOV': '^BVSP',
        'S&P500': '^GSPC',
        'USD/BRL': 'BRL=X'
    }
    
    correlacoes = {}
    
    try:
        df_ativo = yf.Ticker(ticker).history(start=start_date, end=end_date)
        retornos_ativo = df_ativo['Close'].pct_change().dropna()
        
        for nome, ticker_indice in indices.items():
            try:
                df_indice = yf.Ticker(ticker_indice).history(start=start_date, end=end_date)
                retornos_indice = df_indice['Close'].pct_change().dropna()
                
                retornos_combined = pd.DataFrame({
                    'ativo': retornos_ativo,
                    'indice': retornos_indice
                }).dropna()
                
                if len(retornos_combined) > 20:
                    corr = retornos_combined['ativo'].corr(retornos_combined['indice'])
                    
                    cov = retornos_combined['ativo'].cov(retornos_combined['indice'])
                    var_indice = retornos_combined['indice'].var()
                    beta = cov / var_indice if var_indice != 0 else 0
                    
                    correlacoes[nome] = {
                        'correlacao': corr,
                        'beta': beta,
                        'interpretacao': interpretar_correlacao(corr)
                    }
            except:
                correlacoes[nome] = None
        
        return correlacoes
    except:
        return {}


def interpretar_correlacao(corr):
    """Interpreta valor de correlação"""
    abs_corr = abs(corr)
    if abs_corr > 0.7:
        forca = "FORTE"
    elif abs_corr > 0.4:
        forca = "MODERADA"
    else:
        forca = "FRACA"
    
    direcao = "positiva" if corr > 0 else "negativa"
    return f"{forca} {direcao}"


def preparar_sequencias_dl(df, features_cols, seq_length=60):
    """Prepara sequências para modelos de Deep Learning"""
    
    # VALIDAÇÃO 1: Dados suficientes
    min_required = seq_length * 5  # Pelo menos 5x a sequência
    if len(df) < min_required:
        st.error(f"❌ Dados insuficientes para DL: {len(df)} < {min_required}")
        st.warning(f"💡 Aumente 'Dias Histórico' para pelo menos {min_required * 1.5:.0f} dias")
        return None
    
    # Selecionar features numéricas
    data = df[features_cols].values
    target = df['target'].values
    
    # VALIDAÇÃO 2: Checar NaN/Inf
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        st.error("❌ Dados contêm NaN ou Inf - limpando...")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.any(np.isnan(target)) or np.any(np.isinf(target)):
        st.error("❌ Target contém NaN ou Inf - limpando...")
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalizar
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    data_scaled = scaler_X.fit_transform(data)
    target_scaled = scaler_y.fit_transform(target.reshape(-1, 1))
    
    # Criar sequências
    X_seq, y_seq = [], []
    for i in range(seq_length, len(data_scaled)):
        X_seq.append(data_scaled[i-seq_length:i])
        y_seq.append(target_scaled[i])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # VALIDAÇÃO 3: Sequências suficientes
    if len(X_seq) < 100:
        st.error(f"❌ Muito poucas sequências: {len(X_seq)} < 100")
        st.warning(f"💡 Reduza 'Sequência' para {seq_length // 2} ou aumente dados históricos")
        return None
    
    # Split train/val/test (70/15/15)
    train_size = int(len(X_seq) * 0.7)
    val_size = int(len(X_seq) * 0.15)
    
    # VALIDAÇÃO 4: Splits mínimos
    if train_size < 50 or val_size < 10:
        st.error(f"❌ Splits muito pequenos: train={train_size}, val={val_size}")
        st.warning("💡 Precisa de pelo menos 200 sequências totais")
        return None
    
    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]
    
    X_val = X_seq[train_size:train_size+val_size]
    y_val = y_seq[train_size:train_size+val_size]
    
    X_test = X_seq[train_size+val_size:]
    y_test = y_seq[train_size+val_size:]
    
    st.success(f"✅ Sequências criadas: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler_X': scaler_X, 'scaler_y': scaler_y
    }


# ============  TRANSFORMER ============
def criar_transformer_model(seq_length, n_features, d_model=64, num_heads=4, ff_dim=128, dropout=0.2):
    """Cria modelo Transformer para séries temporais"""
    
    inputs = Input(shape=(seq_length, n_features))
    
    # Projeção linear para d_model
    x = Dense(d_model)(inputs)
    
    # Positional Encoding (simplificado)
    positions = tf.range(start=0, limit=seq_length, delta=1)
    position_embedding = layers.Embedding(input_dim=seq_length, output_dim=d_model)(positions)
    x = x + position_embedding
    
    # Multi-Head Attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=d_model // num_heads,
        dropout=dropout
    )(x, x)
    
    # Add & Norm
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    # Feed Forward
    ffn = Sequential([
        Dense(ff_dim, activation='relu'),
        Dropout(dropout),
        Dense(d_model)
    ])
    ffn_output = ffn(x)
    
    # Add & Norm
    x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    # Global Average Pooling
    x = GlobalAveragePooling1D()(x)
    
    # Output
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model


# ============  BiLSTM ============
def criar_bilstm_model(seq_length, n_features):
    """Cria modelo BiLSTM (Bidirectional LSTM)"""
    
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(seq_length, n_features)),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


# ============ NOVO GRU ============
def criar_gru_model(seq_length, n_features):
    """Cria modelo GRU (Gated Recurrent Unit)"""
    
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.3),
        GRU(64, return_sequences=True),
        Dropout(0.3),
        GRU(32),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


# ============  CNN-LSTM ============
def criar_cnn_lstm_model(seq_length, n_features):
    """Cria modelo híbrido CNN-LSTM"""
    
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, n_features)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


# ============  TREINAR MODELOS DL ============
def treinar_modelos_deep_learning(dados_seq, seq_length, n_features, epochs=50, batch_size=32):
    """Treina todos os modelos de Deep Learning"""
    
    modelos_dl = {}
    historicos = {}
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    X_train = dados_seq['X_train']
    y_train = dados_seq['y_train']
    X_val = dados_seq['X_val']
    y_val = dados_seq['y_val']
    
    # Transformer
    if usar_transformer:
        st.info("🤖 Treinando Transformer...")
        transformer = criar_transformer_model(seq_length, n_features)
        history_transformer = transformer.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        modelos_dl['Transformer'] = transformer
        historicos['Transformer'] = history_transformer
        st.success(f"✅ Transformer treinado ({len(history_transformer.history['loss'])} epochs)")
    
    # BiLSTM
    if usar_bilstm:
        st.info("⚡ Treinando BiLSTM...")
        bilstm = criar_bilstm_model(seq_length, n_features)
        history_bilstm = bilstm.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        modelos_dl['BiLSTM'] = bilstm
        historicos['BiLSTM'] = history_bilstm
        st.success(f"✅ BiLSTM treinado ({len(history_bilstm.history['loss'])} epochs)")
    
    # GRU
    if usar_gru:
        st.info("🔄 Treinando GRU...")
        gru = criar_gru_model(seq_length, n_features)
        history_gru = gru.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        modelos_dl['GRU'] = gru
        historicos['GRU'] = history_gru
        st.success(f"✅ GRU treinado ({len(history_gru.history['loss'])} epochs)")
    
    # CNN-LSTM
    if usar_cnn_lstm:
        st.info("🎯 Treinando CNN-LSTM...")
        cnn_lstm = criar_cnn_lstm_model(seq_length, n_features)
        history_cnn = cnn_lstm.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        modelos_dl['CNN-LSTM'] = cnn_lstm
        historicos['CNN-LSTM'] = history_cnn
        st.success(f"✅ CNN-LSTM treinado ({len(history_cnn.history['loss'])} epochs)")
    
    return modelos_dl, historicos


# ============  ENSEMBLE NEURAL ============
def prever_ensemble_neural(modelos_dl, X_atual, scaler_y):
    """Combina previsões de todos modelos neurais"""
    
    previsoes = []
    pesos = []
    
    for nome, modelo in modelos_dl.items():
        pred_scaled = modelo.predict(X_atual, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled)
        previsoes.append(pred[0][0])
        
        # Pesos baseados na complexidade do modelo
        if nome == 'Transformer':
            pesos.append(2.0)
        elif nome == 'BiLSTM':
            pesos.append(1.8)
        elif nome == 'CNN-LSTM':
            pesos.append(1.5)
        elif nome == 'GRU':
            pesos.append(1.3)
        else:
            pesos.append(1.0)
    
    # Weighted average
    pesos = np.array(pesos) / np.sum(pesos)
    previsao_final = np.average(previsoes, weights=pesos)
    
    return previsao_final, dict(zip(modelos_dl.keys(), previsoes))


# ============  AVALIAR MODELOS DL ============
def avaliar_modelos_dl(modelos_dl, X_test, y_test, scaler_y):
    """Avalia performance de cada modelo DL"""
    
    resultados = {}
    
    for nome, modelo in modelos_dl.items():
        # Previsões
        y_pred_scaled = modelo.predict(X_test, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = scaler_y.inverse_transform(y_test)
        
        # Métricas
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        resultados[nome] = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'rmse': np.sqrt(mse)
        }
    
    return resultados



# ============  RELATÓRIO PDF ============
def gerar_relatorio_pdf(resultado, df_dados, empresa_nome):
    """Gera relatório PDF profissional com todas as análises"""
    
    if not PDF_DISPONIVEL:
        return None
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
        
        story = []
        styles = getSampleStyleSheet()
        
        # Estilo customizado
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#0d47a1'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1565c0'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Título
        story.append(Paragraph(f"Relatório de Análise - {empresa_nome}", title_style))
        story.append(Paragraph(f"Gerado em: {dt.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # ===== SEÇÃO 1: RESUMO EXECUTIVO =====
        story.append(Paragraph("📊 Resumo Executivo", heading_style))
        
        data_resumo = [
            ['Métrica', 'Valor'],
            ['Preço Atual', f"R$ {resultado['preco_atual']:.2f}"],
            ['Previsão (1d)', f"R$ {resultado['previsao']:.2f}"],
            ['Variação', f"{resultado['variacao']:+.2f}%"],
            ['Score', f"{resultado['score']:.1f}/100"],
            ['Regime', resultado['regime']],
            ['MAPE', f"{resultado['metricas']['mape']:.2f}%"],
            ['R²', f"{resultado['metricas']['r2']:.4f}"]
        ]
        
        table_resumo = Table(data_resumo, colWidths=[3*inch, 2*inch])
        table_resumo.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#64b5f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table_resumo)
        story.append(Spacer(1, 0.3*inch))
        
        # ===== SEÇÃO 2: SIGNAL STRENGTH =====
        if 'signal_strength' in resultado:
            story.append(Paragraph("🎯 Signal Strength Indicator", heading_style))
            ss = resultado['signal_strength']
            
            story.append(Paragraph(f"<b>Score:</b> {ss['score']:.0f}/100", styles['Normal']))
            story.append(Paragraph(f"<b>Recomendação:</b> {ss['recomendacao']}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            
            story.append(Paragraph("<b>Sinais Detectados:</b>", styles['Normal']))
            for sinal in ss['sinais'][:10]:
                story.append(Paragraph(f"• {sinal['fonte']}: {sinal['sinal']} (Força: {sinal['forca']:+.1f})", styles['Normal']))
            
            story.append(Spacer(1, 0.3*inch))
        
        # ===== SEÇÃO 3: MULTI-HORIZONTE =====
        story.append(Paragraph("📈 Previsões Multi-Horizonte", heading_style))
        
        data_horizonte = [['Horizonte', 'Previsão', 'Variação']]
        for p in resultado['previsoes_multi']:
            data_horizonte.append([
                f"{p['horizonte']} dias",
                f"R$ {p['previsao']:.2f}",
                f"{p['variacao_pct']:+.2f}%"
            ])
        
        table_horizonte = Table(data_horizonte, colWidths=[1.5*inch, 2*inch, 1.5*inch])
        table_horizonte.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#64b5f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table_horizonte)
        story.append(Spacer(1, 0.3*inch))
        
        # ===== SEÇÃO 4: PADRÕES CANDLESTICK =====
        if resultado.get('padroes_candlestick'):
            story.append(Paragraph("🕯️ Padrões Candlestick", heading_style))
            
            for padrao in resultado['padroes_candlestick']:
                story.append(Paragraph(
                    f"• <b>{padrao['padrao']}</b> - {padrao['tipo']} (Confiança: {padrao['confianca']}%)",
                    styles['Normal']
                ))
            
            story.append(Spacer(1, 0.3*inch))
        
        # ===== SEÇÃO 5: SUPPORT & RESISTANCE =====
        if resultado.get('suporte_resistencia'):
            story.append(Paragraph("📊 Support & Resistance", heading_style))
            sr = resultado['suporte_resistencia']
            
            if sr['suportes']:
                story.append(Paragraph("<b>Suportes:</b>", styles['Normal']))
                for s in sr['suportes']:
                    story.append(Paragraph(f"• R$ {s:.2f}", styles['Normal']))
            
            story.append(Spacer(1, 0.1*inch))
            
            if sr['resistencias']:
                story.append(Paragraph("<b>Resistências:</b>", styles['Normal']))
                for r in sr['resistencias']:
                    story.append(Paragraph(f"• R$ {r:.2f}", styles['Normal']))
            
            story.append(Spacer(1, 0.3*inch))
        
        # ===== SEÇÃO 6: GARCH =====
        if resultado.get('garch'):
            story.append(Paragraph("📉 GARCH - Volatilidade", heading_style))
            garch = resultado['garch']
            
            data_garch = [
                ['Métrica', 'Valor'],
                ['Volatilidade Atual', f"{garch['vol_atual']:.2f}%"],
                ['Vol Média (5d)', f"{garch['vol_media']:.2f}%"],
                ['Vol Máxima (5d)', f"{garch['vol_max']:.2f}%"],
                ['Vol Mínima (5d)', f"{garch['vol_min']:.2f}%"]
            ]
            
            table_garch = Table(data_garch, colWidths=[3*inch, 2*inch])
            table_garch.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#64b5f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige)
            ]))
            
            story.append(table_garch)
            story.append(Spacer(1, 0.3*inch))
        
        # ===== SEÇÃO 7: CORRELAÇÃO =====
        if resultado.get('correlacoes'):
            story.append(Paragraph("🌐 Correlação com Índices", heading_style))
            
            data_corr = [['Índice', 'Correlação', 'Beta', 'Interpretação']]
            for nome, corr_data in resultado['correlacoes'].items():
                if corr_data:
                    data_corr.append([
                        nome,
                        f"{corr_data['correlacao']:.3f}",
                        f"{corr_data['beta']:.3f}",
                        corr_data['interpretacao']
                    ])
            
            if len(data_corr) > 1:
                table_corr = Table(data_corr, colWidths=[1.5*inch, 1.3*inch, 1*inch, 2*inch])
                table_corr.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#64b5f6')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige)
                ]))
                
                story.append(table_corr)
        
        story.append(Spacer(1, 0.3*inch))
        
        # ===== SEÇÃO 8: SENTIMENTO =====
        if resultado.get('sentimento') is not None and resultado.get('noticias'):
            story.append(Paragraph("📰 Análise de Sentimento", heading_style))
            
            sent = resultado['sentimento']
            if sent > 0.2:
                sent_label = "POSITIVO"
            elif sent < -0.2:
                sent_label = "NEGATIVO"
            else:
                sent_label = "NEUTRO"
            
            story.append(Paragraph(f"<b>Sentimento Geral:</b> {sent_label} ({sent:.2f})", styles['Normal']))
            story.append(Paragraph(f"<b>Notícias Analisadas:</b> {len(resultado['noticias'])}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            
            story.append(Paragraph("<b>Top 5 Notícias:</b>", styles['Normal']))
            for i, noticia in enumerate(resultado['noticias'][:5], 1):
                story.append(Paragraph(f"{i}. {noticia['titulo'][:100]}...", styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # ===== SEÇÃO 9: DEEP LEARNING =====
        if resultado.get('modelos_dl') and resultado.get('metricas_dl'):
            story.append(Paragraph("🧠 Deep Learning", heading_style))
            
            if resultado.get('previsao_dl'):
                story.append(Paragraph(f"<b>Previsão Ensemble Neural:</b> R$ {resultado['previsao_dl']:.2f}", styles['Normal']))
                variacao_dl = ((resultado['previsao_dl'] - resultado['preco_atual']) / resultado['preco_atual']) * 100
                story.append(Paragraph(f"<b>Variação Esperada:</b> {variacao_dl:+.2f}%", styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
            
            # Tabela de métricas DL
            data_dl = [['Modelo', 'R²', 'MAPE', 'MAE']]
            for nome, metricas in resultado['metricas_dl'].items():
                data_dl.append([
                    nome,
                    f"{metricas['r2']:.4f}",
                    f"{metricas['mape']:.2f}%",
                    f"R$ {metricas['mae']:.2f}"
                ])
            
            if len(data_dl) > 1:
                table_dl = Table(data_dl, colWidths=[2*inch, 1.3*inch, 1.3*inch, 1.3*inch])
                table_dl.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#64b5f6')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige)
                ]))
                
                story.append(table_dl)
            
            story.append(Spacer(1, 0.2*inch))
            
            # Previsões individuais
            if resultado.get('previsoes_individuais_dl'):
                story.append(Paragraph("<b>Previsões Individuais:</b>", styles['Normal']))
                for nome, valor in resultado['previsoes_individuais_dl'].items():
                    story.append(Paragraph(f"• {nome}: R$ {valor:.2f}", styles['Normal']))
        
        # Rodapé
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("_" * 80, styles['Normal']))
        story.append(Paragraph(
            "Relatório gerado automaticamente pelo Sistema - Deep Learning Edition",
            styles['Normal']
        ))
        story.append(Paragraph(
            "⚠️ Este relatório é apenas informativo. Não constitui recomendação de investimento.",
            styles['Normal']
        ))
        
        # Construir PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
    
    except Exception as e:
        st.error(f"Erro ao gerar PDF: {e}")
        return None




def filtrar_outliers_iqr(df, coluna='Close', fator=1.5):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - fator * IQR
    limite_superior = Q3 + fator * IQR
    df_filtrado = df[(df[coluna] >= limite_inferior) & (df[coluna] <= limite_superior)]
    return df_filtrado, len(df) - len(df_filtrado)


def detectar_regime_historico(df, idx):
    if idx < 200:
        return "INDEFINIDO"
    janela = df.iloc[max(0, idx-200):idx]
    if len(janela) < 50:
        return "INDEFINIDO"
    sma_50 = janela['Close'].rolling(50).mean().iloc[-1]
    sma_200 = janela['Close'].rolling(200).mean().iloc[-1] if len(janela) >= 200 else sma_50
    preco_atual = janela['Close'].iloc[-1]
    
    if preco_atual > sma_50 > sma_200:
        return "ALTA"
    elif preco_atual < sma_50 < sma_200:
        return "BAIXA"
    else:
        return "LATERAL"


def detectar_regime_atual(df, janela=50):
    if len(df) < janela * 4:
        return "INDEFINIDO", "BAIXA"
    sma_50 = df['Close'].rolling(janela).mean().iloc[-1]
    sma_200 = df['Close'].rolling(janela*4).mean().iloc[-1]
    preco_atual = df['Close'].iloc[-1]
    volatilidade = df['Close'].pct_change().rolling(janela).std().iloc[-1]
    
    if preco_atual > sma_50 > sma_200:
        regime = "ALTA"
    elif preco_atual < sma_50 < sma_200:
        regime = "BAIXA"
    else:
        regime = "LATERAL"
    
    vol_label = "ALTA" if volatilidade > 0.03 else "BAIXA"
    return regime, vol_label


def criar_features_avancadas(df):
    df_f = pd.DataFrame(index=df.index)
    df_f['Close'] = df['Close']
    df_f['Volume'] = df['Volume']
    df_f['Returns'] = df['Close'].pct_change()
    
    for p in [5, 10, 20, 50]:
        df_f[f'SMA_{p}'] = SMAIndicator(df['Close'], window=p).sma_indicator()
        df_f[f'EMA_{p}'] = EMAIndicator(df['Close'], window=p).ema_indicator()
    
    for p in [7, 14, 21]:
        df_f[f'RSI_{p}'] = RSIIndicator(df['Close'], window=p).rsi()
    
    bb = BollingerBands(df['Close'])
    df_f['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df_f['BB_Position'] = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    
    macd = MACD(df['Close'])
    df_f['MACD_Diff'] = macd.macd_diff()
    df_f['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    for p in [5, 10, 20]:
        df_f[f'Volatility_{p}'] = df_f['Returns'].rolling(p).std()
    
    for i in [1, 2, 3, 5]:
        df_f[f'Close_Lag_{i}'] = df['Close'].shift(i)
    
    df_f['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
    
    return df_f


def criar_stacking_ensemble(X_train, y_train):
    base_models = [
        ('ridge', Ridge(alpha=10.0)),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42))
    ]
    
    if usar_lightgbm and LIGHTGBM_DISPONIVEL:
        base_models.append(('lgbm', lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)))
    
    if usar_catboost and CATBOOST_DISPONIVEL:
        base_models.append(('catboost', cb.CatBoostRegressor(iterations=100, depth=5, learning_rate=0.05, random_state=42, verbose=0)))
    
    meta_model = Ridge(alpha=1.0)
    stacking = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5)
    stacking.fit(X_train, y_train)
    
    return stacking


def criar_ensemble_melhorado(X_train, y_train):
    modelos = []
    
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train, y_train)
    modelos.append(('Ridge', ridge))
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    modelos.append(('RF', rf))
    
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    gb.fit(X_train, y_train)
    modelos.append(('GB', gb))
    
    if usar_lightgbm and LIGHTGBM_DISPONIVEL:
        lgbm = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
        lgbm.fit(X_train, y_train)
        modelos.append(('LGBM', lgbm))
    
    if usar_catboost and CATBOOST_DISPONIVEL:
        cat = cb.CatBoostRegressor(iterations=100, depth=5, learning_rate=0.05, random_state=42, verbose=0)
        cat.fit(X_train, y_train)
        modelos.append(('CatBoost', cat))
    
    return modelos


def prever_ensemble(modelos, X):
    previsoes, pesos = [], []
    for nome, modelo in modelos:
        previsoes.append(modelo.predict(X))
        if nome == 'CatBoost':
            pesos.append(2.5)
        elif nome == 'LGBM':
            pesos.append(2.0)
        elif nome == 'GB':
            pesos.append(1.5)
        elif nome == 'RF':
            pesos.append(1.2)
        else:
            pesos.append(1.0)
    pesos = np.array(pesos) / np.sum(pesos)
    return np.average(previsoes, weights=pesos, axis=0)


def calcular_feature_importance(modelos, features_cols):
    importances = {}
    for nome, modelo in modelos:
        if hasattr(modelo, 'feature_importances_'):
            imp = modelo.feature_importances_
            for i, col in enumerate(features_cols):
                if col not in importances:
                    importances[col] = []
                importances[col].append(imp[i])
    
    if not importances:
        return []
    
    importances_media = {k: np.mean(v) for k, v in importances.items()}
    top_features = sorted(importances_media.items(), key=lambda x: x[1], reverse=True)[:15]
    return top_features


def verificar_alertas(preco_atual, previsao, score, variacao, regime):
    alertas = []
    
    if score > 70:
        alertas.append({'tipo': 'success', 'icone': '🎯', 'msg': f'Score Alto ({score:.1f}/100)'})
    
    if abs(variacao) > 3:
        if variacao > 0:
            alertas.append({'tipo': 'success', 'icone': '📈', 'msg': f'Alta Prevista: +{variacao:.2f}%'})
        else:
            alertas.append({'tipo': 'warning', 'icone': '📉', 'msg': f'Queda Prevista: {variacao:.2f}%'})
    
    if regime == "ALTA":
        alertas.append({'tipo': 'success', 'icone': '🟢', 'msg': 'Regime ALTA - Favorável'})
    elif regime == "BAIXA":
        alertas.append({'tipo': 'error', 'icone': '🔴', 'msg': 'Regime BAIXA - Cautela'})
    
    if score < 50:
        alertas.append({'tipo': 'warning', 'icone': '⚠️', 'msg': f'Score Baixo ({score:.1f}/100)'})
    
    return alertas


def treinar_modelos_por_regime(df_historico, features_cols):
    regimes = [detectar_regime_historico(df_historico, i) for i in range(len(df_historico))]
    df_historico['regime'] = regimes
    
    contagem = df_historico['regime'].value_counts()
    regimes_validos = [r for r in ['ALTA', 'BAIXA', 'LATERAL'] if contagem.get(r, 0) >= 150]
    
    if len(regimes_validos) == 0:
        st.warning("⚠️ Usando modelo único")
        X = df_historico[features_cols]
        y = df_historico['target'].values
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if usar_stacking:
            modelo = criar_stacking_ensemble(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)
            modelos_lista = [('Stacking', modelo)]
        else:
            modelos_lista = criar_ensemble_melhorado(X_train_scaled, y_train)
            y_pred = prever_ensemble(modelos_lista, X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        modelo_unico = {'modelos': modelos_lista, 'scaler': scaler}
        stats_unico = {'n_exemplos': len(df_historico), 'mae': mae, 'r2': r2, 'mape': mape}
        
        st.success(f"✅ Modelo Único: R²={r2:.3f} | MAPE={mape:.1f}%")
        
        return {'ALTA': modelo_unico, 'BAIXA': modelo_unico, 'LATERAL': modelo_unico}, \
               {'ALTA': stats_unico, 'BAIXA': stats_unico, 'LATERAL': stats_unico}
    
    modelos_regime = {}
    stats_regime = {}
    
    for regime in regimes_validos:
        df_regime = df_historico[df_historico['regime'] == regime].copy()
        X = df_regime[features_cols]
        y = df_regime['target'].values
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if usar_stacking:
            modelo = criar_stacking_ensemble(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)
            modelos_lista = [('Stacking', modelo)]
        else:
            modelos_lista = criar_ensemble_melhorado(X_train_scaled, y_train)
            y_pred = prever_ensemble(modelos_lista, X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        modelos_regime[regime] = {'modelos': modelos_lista, 'scaler': scaler}
        stats_regime[regime] = {'n_exemplos': len(df_regime), 'mae': mae, 'r2': r2, 'mape': mape}
        
        st.success(f"✅ {regime}: R²={r2:.3f} | MAPE={mape:.1f}%")
    
    if len(modelos_regime) > 0:
        modelo_fallback = list(modelos_regime.values())[0]
        stats_fallback = list(stats_regime.values())[0]
        for regime in ['ALTA', 'BAIXA', 'LATERAL']:
            if regime not in modelos_regime:
                modelos_regime[regime] = modelo_fallback
                stats_regime[regime] = stats_fallback
    
    return modelos_regime, stats_regime


def otimizar_hiperparametros_optuna(X_train, y_train, X_val, y_val, n_trials=30):
    def objective(trial):
        modelo_tipo = trial.suggest_categorical('modelo', ['ridge', 'rf', 'gb'])
        
        if modelo_tipo == 'ridge':
            alpha = trial.suggest_float('alpha', 0.1, 100.0, log=True)
            modelo = Ridge(alpha=alpha)
        elif modelo_tipo == 'rf':
            n_est = trial.suggest_int('n_estimators', 50, 200)
            max_d = trial.suggest_int('max_depth', 3, 10)
            modelo = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=-1)
        else:
            n_est = trial.suggest_int('n_estimators', 50, 200)
            max_d = trial.suggest_int('max_depth', 2, 5)
            lr = trial.suggest_float('learning_rate', 0.01, 0.2)
            modelo = GradientBoostingRegressor(n_estimators=n_est, max_depth=max_d, learning_rate=lr, random_state=42)
        
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_val)
        mape = mean_absolute_percentage_error(y_val, y_pred)
        return mape
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def backtesting_completo(df, df_features, modelo_info, dias_previsao=1):
    capital_inicial = 10000
    capital = capital_inicial
    posicao = 0
    acoes = 0
    trades = []
    portfolio_value = []
    
    features_cols = [col for col in df_features.columns if col not in ['target', 'regime']]
    inicio_teste = int(len(df_features) * 0.8)
    
    for i in range(inicio_teste, len(df_features) - dias_previsao - 1):
        X_atual = df_features.iloc[i][features_cols].values.reshape(1, -1)
        
        if usar_modelos_regime and 'modelos' in modelo_info and modelo_info.get('tipo') == 'regime':
            regime_atual = df_features.iloc[i].get('regime', 'LATERAL')
            if regime_atual in modelo_info['modelos'] and regime_atual != 'INDEFINIDO':
                modelo_usar = modelo_info['modelos'][regime_atual]['modelos']
                scaler_usar = modelo_info['modelos'][regime_atual]['scaler']
            else:
                if 'LATERAL' in modelo_info['modelos']:
                    modelo_usar = modelo_info['modelos']['LATERAL']['modelos']
                    scaler_usar = modelo_info['modelos']['LATERAL']['scaler']
                else:
                    continue
        else:
            modelo_usar = modelo_info['modelos']
            scaler_usar = modelo_info['scaler']
        
        X_scaled = scaler_usar.transform(X_atual)
        
        if isinstance(modelo_usar, list):
            if len(modelo_usar) == 1 and modelo_usar[0][0] == 'Stacking':
                previsao = modelo_usar[0][1].predict(X_scaled)[0]
            else:
                previsao = prever_ensemble(modelo_usar, X_scaled)[0]
        else:
            previsao = modelo_usar.predict(X_scaled)[0]
        
        preco_atual = df['Close'].iloc[i]
        preco_futuro_real = df['Close'].iloc[i + dias_previsao]
        variacao_prevista = ((previsao - preco_atual) / preco_atual) * 100
        
        if variacao_prevista > 1.5 and posicao == 0:
            acoes = capital / preco_atual
            capital = 0
            posicao = 1
            trades.append({'tipo': 'COMPRA', 'data': df.index[i], 'preco': preco_atual, 'acoes': acoes})
        
        elif ((variacao_prevista < -1.5) or (i == len(df_features) - dias_previsao - 2)) and posicao == 1:
            capital = acoes * preco_futuro_real
            lucro_trade = ((preco_futuro_real - trades[-1]['preco']) / trades[-1]['preco']) * 100
            posicao = 0
            trades.append({'tipo': 'VENDA', 'data': df.index[i + dias_previsao], 'preco': preco_futuro_real, 'lucro': lucro_trade})
            acoes = 0
        
        valor_atual = capital if posicao == 0 else acoes * preco_atual
        portfolio_value.append({'data': df.index[i], 'valor': valor_atual})
    
    if posicao == 1:
        capital = acoes * df['Close'].iloc[-1]
        lucro_trade = ((df['Close'].iloc[-1] - trades[-1]['preco']) / trades[-1]['preco']) * 100
        trades.append({'tipo': 'VENDA', 'data': df.index[-1], 'preco': df['Close'].iloc[-1], 'lucro': lucro_trade})
    
    capital_final = capital
    acoes_buyhold = capital_inicial / df['Close'].iloc[inicio_teste]
    capital_buyhold = acoes_buyhold * df['Close'].iloc[-1]
    
    retorno_estrategia = ((capital_final - capital_inicial) / capital_inicial) * 100
    retorno_buyhold = ((capital_buyhold - capital_inicial) / capital_inicial) * 100
    
    trades_venda = [t for t in trades if t['tipo'] == 'VENDA' and 'lucro' in t]
    num_trades = len(trades_venda)
    win_rate = (len([t for t in trades_venda if t['lucro'] > 0]) / num_trades * 100) if num_trades > 0 else 0
    
    if trades_venda:
        retornos = [t['lucro'] for t in trades_venda]
        sharpe = (np.mean(retornos) / np.std(retornos)) * np.sqrt(252) if np.std(retornos) > 0 else 0
    else:
        sharpe = 0
    
    valores = [capital_inicial] + [pv['valor'] for pv in portfolio_value]
    pico = valores[0]
    max_dd = 0
    for valor in valores:
        if valor > pico:
            pico = valor
        dd = (pico - valor) / pico * 100
        max_dd = max(max_dd, dd)
    
    return {
        'capital_inicial': capital_inicial, 'capital_final': capital_final,
        'retorno_estrategia': retorno_estrategia, 'capital_buyhold': capital_buyhold,
        'retorno_buyhold': retorno_buyhold, 'vantagem': retorno_estrategia - retorno_buyhold,
        'num_trades': num_trades, 'win_rate': win_rate, 'sharpe_ratio': sharpe,
        'max_drawdown': max_dd, 'trades': trades, 'portfolio_evolution': portfolio_value
    }


def criar_dashboard_profissional(df_dados):
    df_viz = df_dados.tail(90)
    
    fig = make_subplots(rows=4, cols=1, row_heights=[0.5, 0.15, 0.15, 0.2],
                       subplot_titles=('Preço + Indicadores', 'Volume', 'RSI', 'MACD'), vertical_spacing=0.03)
    
    fig.add_trace(go.Candlestick(x=df_viz.index, open=df_viz['Open'], high=df_viz['High'],
                                 low=df_viz['Low'], close=df_viz['Close'], name='Preço',
                                 increasing_line_color='#00ff00', decreasing_line_color='#ff0000'), row=1, col=1)
    
    sma20 = df_viz['Close'].rolling(20).mean()
    sma50 = df_viz['Close'].rolling(50).mean()
    
    fig.add_trace(go.Scatter(x=df_viz.index, y=sma20, name='SMA 20', line=dict(color='#3b82f6', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_viz.index, y=sma50, name='SMA 50', line=dict(color='#f59e0b', width=1.5)), row=1, col=1)
    
    bb = BollingerBands(df_viz['Close'])
    fig.add_trace(go.Scatter(x=df_viz.index, y=bb.bollinger_hband(), line=dict(color='gray', dash='dash', width=1), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_viz.index, y=bb.bollinger_lband(), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', line=dict(color='gray', dash='dash', width=1), showlegend=False), row=1, col=1)
    
    colors_volume = ['#00ff00' if df_viz['Close'].iloc[i] >= df_viz['Open'].iloc[i] else '#ff0000' for i in range(len(df_viz))]
    fig.add_trace(go.Bar(x=df_viz.index, y=df_viz['Volume'], marker_color=colors_volume, showlegend=False), row=2, col=1)
    
    rsi = RSIIndicator(df_viz['Close']).rsi()
    fig.add_trace(go.Scatter(x=df_viz.index, y=rsi, name='RSI', line=dict(color='#a855f7', width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    macd = MACD(df_viz['Close'])
    fig.add_trace(go.Scatter(x=df_viz.index, y=macd.macd(), name='MACD', line=dict(color='#3b82f6', width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_viz.index, y=macd.macd_signal(), name='Signal', line=dict(color='#f59e0b', width=2)), row=4, col=1)
    
    fig.update_layout(height=1200, showlegend=True, xaxis_rangeslider_visible=False, template='plotly_white', hovermode='x unified')
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='#e3f2fd')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#e3f2fd')
    
    return fig


def prever_multiplos_horizontes(df_dados, modelo_info, scaler, features_cols):
    df = criar_features_avancadas(df_dados)
    
    if 'sentimento' in features_cols:
        df['sentimento'] = 0.0
    
    df = df.dropna()
    features_disponiveis = [f for f in features_cols if f in df.columns]
    
    X_atual = df[features_disponiveis].tail(1).values
    X_atual_scaled = scaler.transform(X_atual)
    
    horizontes = [1, 2, 3, 5, 7]
    previsoes = []
    preco_atual = df_dados['Close'].iloc[-1]
    
    if 'tipo' in modelo_info and modelo_info['tipo'] == 'regime':
        if 'LATERAL' in modelo_info['modelos']:
            modelos_usar = modelo_info['modelos']['LATERAL']['modelos']
        else:
            primeiro_regime = list(modelo_info['modelos'].keys())[0]
            modelos_usar = modelo_info['modelos'][primeiro_regime]['modelos']
    else:
        modelos_usar = modelo_info['modelos']
    
    for h in horizontes:
        if isinstance(modelos_usar, list):
            if len(modelos_usar) == 1 and modelos_usar[0][0] == 'Stacking':
                pred = modelos_usar[0][1].predict(X_atual_scaled)[0]
            else:
                pred = prever_ensemble(modelos_usar, X_atual_scaled)[0]
        else:
            pred = modelos_usar.predict(X_atual_scaled)[0]
        
        variacao = pred - preco_atual
        pred_ajustada = preco_atual + (variacao * (0.95 ** (h-1)))
        var_pct = ((pred_ajustada - preco_atual) / preco_atual) * 100
        
        previsoes.append({'horizonte': h, 'previsao': pred_ajustada, 'variacao_pct': var_pct})
    
    return previsoes


def walk_forward_analysis(df_dados, df_features, features_cols, n_splits=5):
    resultados = []
    tamanho_split = len(df_features) // n_splits
    
    for i in range(1, n_splits):
        train_end = i * tamanho_split
        test_start = train_end
        test_end = min(test_start + tamanho_split, len(df_features))
        
        X_train = df_features.iloc[:train_end][features_cols].values
        y_train = df_features.iloc[:train_end]['target'].values
        
        X_test = df_features.iloc[test_start:test_end][features_cols].values
        y_test = df_features.iloc[test_start:test_end]['target'].values
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        modelo = Ridge(alpha=10.0)
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        resultados.append({
            'split': i,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'r2': r2,
            'mape': mape,
            'periodo': f"{df_features.index[test_start].date()} a {df_features.index[test_end-1].date()}"
        })
    
    return pd.DataFrame(resultados)


def monte_carlo_simulation(preco_atual, retorno_medio, volatilidade, dias=30, simulacoes=10000):
    resultados = []
    
    for _ in range(simulacoes):
        precos = [preco_atual]
        for dia in range(dias):
            retorno_diario = np.random.normal(retorno_medio, volatilidade)
            novo_preco = precos[-1] * (1 + retorno_diario)
            precos.append(novo_preco)
        resultados.append(precos[-1])
    
    resultados = np.array(resultados)
    
    return {
        'preco_atual': preco_atual,
        'media': np.mean(resultados),
        'mediana': np.median(resultados),
        'p5': np.percentile(resultados, 5),
        'p25': np.percentile(resultados, 25),
        'p75': np.percentile(resultados, 75),
        'p95': np.percentile(resultados, 95),
        'prob_alta': (resultados > preco_atual).mean() * 100,
        'prob_baixa': (resultados < preco_atual).mean() * 100,
        'var_95': np.percentile(resultados, 5) - preco_atual,
        'distribuicao': resultados
    }


def portfolio_optimizer(tickers_lista, start_date, end_date):
    dados = {}
    for ticker in tickers_lista:
        df = yf.Ticker(ticker).history(start=start_date, end=end_date)
        if not df.empty:
            dados[ticker] = df['Close']
    
    if len(dados) < 2:
        return None
    
    df_precos = pd.DataFrame(dados)
    retornos = df_precos.pct_change().dropna()
    
    retorno_medio = retornos.mean() * 252
    cov_matrix = retornos.cov() * 252
    
    n_ativos = len(tickers_lista)
    
    def portfolio_variance(pesos):
        return np.dot(pesos, np.dot(cov_matrix, pesos))
    
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n_ativos))
    initial_guess = np.array([1/n_ativos] * n_ativos)
    
    result = minimize(portfolio_variance, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    pesos_otimos = result.x
    retorno_portfolio = np.dot(pesos_otimos, retorno_medio)
    risco_portfolio = np.sqrt(portfolio_variance(pesos_otimos))
    sharpe_ratio = retorno_portfolio / risco_portfolio if risco_portfolio > 0 else 0
    
    return {
        'tickers': tickers_lista,
        'pesos': pesos_otimos,
        'retorno_esperado': retorno_portfolio * 100,
        'risco': risco_portfolio * 100,
        'sharpe': sharpe_ratio,
        'alocacao': {ticker: peso for ticker, peso in zip(tickers_lista, pesos_otimos)}
    }


def calcular_risk_metrics(retornos, capital_inicial=10000):
    retornos_anualizados = retornos.mean() * 252
    volatilidade = retornos.std() * np.sqrt(252)
    
    sharpe = retornos_anualizados / volatilidade if volatilidade > 0 else 0
    
    downside = retornos[retornos < 0].std() * np.sqrt(252)
    sortino = retornos_anualizados / downside if downside > 0 else 0
    
    cumulative = (1 + retornos).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min() * 100
    
    calmar = retornos_anualizados / abs(max_dd) if max_dd != 0 else 0
    
    var_95 = np.percentile(retornos, 5) * capital_inicial
    cvar_95 = retornos[retornos <= np.percentile(retornos, 5)].mean() * capital_inicial
    
    return {
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_drawdown': max_dd,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'volatilidade_anual': volatilidade * 100
    }


def criar_grafico_correlacao(df_features, features_cols):
    corr_matrix = df_features[features_cols[:20]].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 8}
    ))
    
    fig.update_layout(title='Heatmap Correlação (Top 20)', height=600, template='plotly_white')
    return fig


def criar_grafico_montecarlo(mc_results):
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(x=mc_results['distribuicao'], nbinsx=50, name='Distribuição', marker_color='#64b5f6'))
    
    fig.add_vline(x=mc_results['preco_atual'], line_dash="dash", line_color="black", annotation_text="Atual", annotation_position="top")
    fig.add_vline(x=mc_results['media'], line_dash="dash", line_color="blue", annotation_text="Média", annotation_position="top")
    fig.add_vline(x=mc_results['p5'], line_dash="dot", line_color="red", annotation_text="P5", annotation_position="bottom left")
    fig.add_vline(x=mc_results['p95'], line_dash="dot", line_color="green", annotation_text="P95", annotation_position="bottom right")
    
    fig.update_layout(title='Monte Carlo - Distribuição (30d)', xaxis_title='Preço (R$)', yaxis_title='Frequência', template='plotly_white')
    return fig


def criar_grafico_walkforward(df_wf):
    fig = make_subplots(rows=2, cols=1, subplot_titles=('R² por Split', 'MAPE por Split'))
    
    fig.add_trace(go.Bar(x=df_wf['split'], y=df_wf['r2'], name='R²', marker_color='#64b5f6'), row=1, col=1)
    fig.add_trace(go.Bar(x=df_wf['split'], y=df_wf['mape'], name='MAPE (%)', marker_color='#f59e0b'), row=2, col=1)
    
    fig.update_layout(height=600, template='plotly_white', showlegend=False)
    return fig


def criar_grafico_multi_horizonte(preco_atual, previsoes):
    fig = go.Figure()
    
    dias = [0] + [p['horizonte'] for p in previsoes]
    precos = [preco_atual] + [p['previsao'] for p in previsoes]
    
    fig.add_trace(go.Scatter(x=dias, y=precos, mode='lines+markers', name='Previsão',
                            line=dict(color='#2196f3', width=3), marker=dict(size=10)))
    fig.add_hline(y=preco_atual, line_dash="dash", line_color="gray", annotation_text="Atual", annotation_position="right")
    
    fig.update_layout(title='Previsões Multi-Horizonte', xaxis_title='Dias', yaxis_title='Preço (R$)', template='plotly_white')
    return fig


def criar_grafico_feature_importance(top_features):
    features = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]
    
    fig = go.Figure(go.Bar(x=importances, y=features, orientation='h', marker_color='#64b5f6',
                          text=[f'{i:.4f}' for i in importances], textposition='auto'))
    
    fig.update_layout(title='Top 15 Features', xaxis_title='Importância', yaxis_title='Feature', template='plotly_white', height=500)
    return fig


def criar_grafico_analise_erro(y_test, y_pred):
    residuos = y_test - y_pred
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Distribuição Erros', 'Erros no Tempo'))
    
    fig.add_trace(go.Histogram(x=residuos, nbinsx=30, marker_color='#64b5f6'), row=1, col=1)
    fig.add_trace(go.Scatter(y=residuos, mode='markers', marker=dict(color='#42a5f5', size=5)), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
    
    fig.update_layout(showlegend=False, template='plotly_white', height=400)
    return fig


def fazer_previsao (df_dados, ticker_nome):
    with st.spinner(f'🔄 Processando {ticker_nome} com V11.0 MEGA ULTIMATE...'):
        
        df_dados_clean, n_outliers = filtrar_outliers_iqr(df_dados)
        if n_outliers > 0:
            st.info(f"🎯 Outliers removidos: {n_outliers}")
        df_dados = df_dados_clean
        
        regime, vol_regime = detectar_regime_atual(df_dados)
        st.info(f"📊 **Regime:** {regime} | **Volatilidade:** {vol_regime}")
        
        sentimento_score = 0
        noticias = []
        if usar_sentimento:
            st.subheader("📰 Analisando Google News...")
            sentimento_score, noticias = analisar_sentimento_google_news(ticker_nome, dias=7)
            emoji = "😊" if sentimento_score > 0.2 else "😞" if sentimento_score < -0.2 else "😐"
            st.info(f"📰 **Sentimento:** {emoji} {sentimento_score:.2f} | **Notícias:** {len(noticias)}")
        
        df = criar_features_avancadas(df_dados)
        if usar_sentimento:
            df['sentimento'] = sentimento_score
        df = df.dropna()
        
        if len(df) < 200:
            st.error(f'❌ Dados insuficientes')
            return None
        
        df['target'] = df['Close'].shift(-1)
        df_clean = df.dropna()
        
        features_cols = [col for col in df_clean.columns if col not in ['target', 'regime']]
        
        if usar_modelos_regime:
            st.subheader("🎯 Modelos por Regime")
            modelos_regime, stats_regime = treinar_modelos_por_regime(df_clean.copy(), features_cols)
            
            if regime in modelos_regime and regime != 'INDEFINIDO':
                modelo_usar = modelos_regime[regime]['modelos']
                scaler_usar = modelos_regime[regime]['scaler']
            else:
                regime = 'LATERAL'
                if 'LATERAL' in modelos_regime:
                    modelo_usar = modelos_regime['LATERAL']['modelos']
                    scaler_usar = modelos_regime['LATERAL']['scaler']
                else:
                    return None
            
            metricas = stats_regime[regime]
            modelo_info = {'modelos': modelos_regime, 'stats': stats_regime, 'tipo': 'regime'}
        else:
            X = df_clean[features_cols].values
            y = df_clean['target'].values
            
            split_idx = int(len(X) * 0.7)
            val_idx = int(len(X) * 0.85)
            
            X_train = X[:split_idx]
            y_train = y[:split_idx]
            X_val = X[split_idx:val_idx]
            y_val = y[split_idx:val_idx]
            X_test = X[val_idx:]
            y_test = y[val_idx:]
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            if usar_optuna and OPTUNA_DISPONIVEL:
                st.info("🎯 Optuna AutoML...")
                best_params, best_score = otimizar_hiperparametros_optuna(X_train_scaled, y_train, X_val_scaled, y_val, n_trials=20)
                st.success(f"✅ Optuna: MAPE = {best_score:.2f}%")
            
            if usar_stacking:
                st.info("🔥 Stacking Ensemble...")
                modelo = criar_stacking_ensemble(X_train_scaled, y_train)
                y_pred = modelo.predict(X_test_scaled)
                modelos_lista = [('Stacking', modelo)]
            else:
                modelos_lista = criar_ensemble_melhorado(X_train_scaled, y_train)
                y_pred = prever_ensemble(modelos_lista, X_test_scaled)
            
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            
            metricas = {'mae': mae, 'r2': r2, 'mape': mape}
            modelo_usar = modelos_lista
            scaler_usar = scaler
            modelo_info = {'modelos': modelos_lista, 'scaler': scaler, 'tipo': 'unico'}
        
        X_atual = df_clean[features_cols].tail(1).values
        X_atual_scaled = scaler_usar.transform(X_atual)
        
        if isinstance(modelo_usar, list):
            if len(modelo_usar) == 1 and modelo_usar[0][0] == 'Stacking':
                previsao = modelo_usar[0][1].predict(X_atual_scaled)[0]
                top_features = []
            else:
                previsao = prever_ensemble(modelo_usar, X_atual_scaled)[0]
                top_features = calcular_feature_importance(modelo_usar, features_cols)
        else:
            previsao = modelo_usar.predict(X_atual_scaled)[0]
            top_features = []
        
        preco_atual = df_dados['Close'].iloc[-1]
        variacao = ((previsao - preco_atual) / preco_atual) * 100
        
        score = max(0, 100 - metricas['mape'] * 5)
        
        alertas = []
        if usar_alertas:
            alertas = verificar_alertas(preco_atual, previsao, score, variacao, regime)
        
        # y_test e y_pred para análise de erro
        X = df_clean[features_cols].values
        y = df_clean['target'].values
        split_idx = int(len(X) * 0.8)
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        X_test_scaled = scaler_usar.transform(X_test)
        
        if isinstance(modelo_usar, list):
            if len(modelo_usar) == 1 and modelo_usar[0][0] == 'Stacking':
                y_pred = modelo_usar[0][1].predict(X_test_scaled)
            else:
                y_pred = prever_ensemble(modelo_usar, X_test_scaled)
        else:
            y_pred = modelo_usar.predict(X_test_scaled)
        
        previsoes_multi = prever_multiplos_horizontes(df_dados, modelo_info, scaler_usar, features_cols)

        # ========== NOVAS ANÁLISES V12 ==========
        garch_resultado = None
        if usar_garch and GARCH_DISPONIVEL:
            st.info("📊 Calculando GARCH...")
            retornos = df_dados['Close'].pct_change()
            garch_resultado = calcular_garch_volatilidade(retornos, horizonte=5)

        padroes_candle = []
        if usar_candlestick:
            st.info("🕯️ Detectando padrões...")
            padroes_candle = detectar_padroes_candlestick(df_dados, lookback=100)

        sr_levels = {}
        if usar_suporte_resistencia:
            st.info("📈 Calculando S/R...")
            sr_levels = calcular_suporte_resistencia(df_dados, janela=50, num_niveis=3)

        signal_strength = calcular_signal_strength(
            df_dados, previsao, variacao, padroes_candle, 
            sr_levels if sr_levels else {'prox_suporte': None, 'prox_resistencia': None}
        )

        correlacoes = {}
        if usar_correlacao:
            st.info("🌐 Calculando correlações...")
            correlacoes = calcular_correlacao_indices(ticker, start_date, end_date)
        
        # ==========  DEEP LEARNING ==========
        modelos_dl = {}
        historicos_dl = {}
        metricas_dl = {}
        previsao_dl = None
        previsoes_individuais_dl = {}
        
        if TENSORFLOW_DISPONIVEL and (usar_transformer or usar_bilstm or usar_gru or usar_cnn_lstm):
            st.markdown("---")
            st.subheader("🧠 Deep Learning")
            
            # Preparar sequências
            st.info("📊 Preparando sequências temporais...")
            dados_seq = preparar_sequencias_dl(df_clean, features_cols, seq_length)
            
            # VALIDAÇÃO: Se preparação falhou, pular DL
            if dados_seq is None:
                st.error("❌ Deep Learning desabilitado - dados insuficientes")
                st.info("💡 Para usar DL: Aumente 'Dias Histórico' para 1095+ ou reduza 'Sequência' para 30")
            else:
                n_features = len(features_cols)
                
                # Treinar modelos
                modelos_dl, historicos_dl = treinar_modelos_deep_learning(
                    dados_seq, seq_length, n_features, epochs_dl, batch_size_dl
                )
                
                # Avaliar modelos
                if modelos_dl:
                    st.info("📊 Avaliando modelos...")
                    metricas_dl = avaliar_modelos_dl(
                        modelos_dl, 
                        dados_seq['X_test'], 
                        dados_seq['y_test'],
                        dados_seq['scaler_y']
                    )
                    
                    # Fazer previsão com ensemble
                    if usar_ensemble_neural and len(modelos_dl) > 1:
                        st.info("🏆 Gerando Ensemble Neural...")
                        # Preparar sequência atual
                        X_atual_seq = dados_seq['X_test'][-1:] # Última sequência
                        previsao_dl, previsoes_individuais_dl = prever_ensemble_neural(
                            modelos_dl, X_atual_seq, dados_seq['scaler_y']
                        )
                        st.success(f"✅ Ensemble Neural: R$ {previsao_dl:.2f}")
                    elif modelos_dl:
                        # Usar apenas um modelo
                        modelo_nome = list(modelos_dl.keys())[0]
                        modelo = modelos_dl[modelo_nome]
                        X_atual_seq = dados_seq['X_test'][-1:]
                        pred_scaled = modelo.predict(X_atual_seq, verbose=0)
                        previsao_dl = dados_seq['scaler_y'].inverse_transform(pred_scaled)[0][0]
                        previsoes_individuais_dl = {modelo_nome: previsao_dl}
                        st.success(f"✅ {modelo_nome}: R$ {previsao_dl:.2f}")
                    
                    # Mostrar métricas
                    st.markdown("**Métricas dos Modelos:**")
                    for nome, metricas in metricas_dl.items():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"{nome} - R²", f"{metricas['r2']:.4f}")
                        with col2:
                            st.metric("MAPE", f"{metricas['mape']:.2f}%")
                        with col3:
                            st.metric("MAE", f"R$ {metricas['mae']:.2f}")
                        
                        # ALERTA se R² negativo
                        if metricas['r2'] < 0:
                            st.error(f"⚠️ {nome}: R² NEGATIVO! Modelo pior que baseline!")
                            st.warning("💡 Tente: Mais dados históricos OU Reduzir sequência OU Menos epochs")

        return {
            'ticker': ticker_nome, 'preco_atual': preco_atual, 'previsao': previsao,
            'variacao': variacao, 'regime': regime, 'vol_regime': vol_regime,
            'metricas': metricas, 'score': score, 'modelo_info': modelo_info,
            'df_features': df_clean, 'features_cols': features_cols,
            'sentimento': sentimento_score, 'noticias': noticias, 'alertas': alertas,
            'top_features': top_features, 'y_test': y_test, 'y_pred': y_pred,
            'previsoes_multi': previsoes_multi,
            'garch': garch_resultado,
            'padroes_candlestick': padroes_candle,
            'suporte_resistencia': sr_levels,
            'signal_strength': signal_strength,
            'correlacoes': correlacoes,
            'modelos_dl': modelos_dl,
            'historicos_dl': historicos_dl,
            'metricas_dl': metricas_dl,
            'previsao_dl': previsao_dl,
            'previsoes_individuais_dl': previsoes_individuais_dl
        }


# === INTERFACE ===
st.markdown("---")

df_dados = download_dados_cached(ticker, start_date, end_date)

if df_dados.empty:
    st.error("❌ Erro")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Dados", len(df_dados))
with col2:
    st.metric("💰 Preço", f"R$ {df_dados['Close'].iloc[-1]:.2f}")
with col3:
    var = ((df_dados['Close'].iloc[-1] / df_dados['Close'].iloc[0]) - 1) * 100
    st.metric("📈 Var", f"{var:.2f}%")
with col4:
    vol = df_dados['Close'].pct_change().std() * np.sqrt(252) * 100
    st.metric("📊 Vol", f"{vol:.1f}%")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "🔮 Previsão", "🏆 Backtesting", "📊 Dashboard", "📈 Multi-Horizonte",
    "📉 Walk-Forward", "🎲 Monte Carlo", "💼 Portfolio", "📰 Notícias", "🔬 Análise", "🔥 Alto Impacto", "🧠 Deep Learning"
])

with tab1:
    st.header("🔮 Previsão ")
    
    if st.button("🚀 Gerar Previsão ", type="primary"):
        resultado = fazer_previsao(df_dados, empresa_selecionada)
        
        if resultado:
            st.success("✅ Concluído!")
            
            if resultado['alertas']:
                st.markdown("### 🔔 Alertas")
                for alerta in resultado['alertas']:
                    if alerta['tipo'] == 'success':
                        st.success(f"{alerta['icone']} {alerta['msg']}")
                    elif alerta['tipo'] == 'warning':
                        st.warning(f"{alerta['icone']} {alerta['msg']}")
                    elif alerta['tipo'] == 'error':
                        st.error(f"{alerta['icone']} {alerta['msg']}")
            
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🏆 Score", f"{resultado['score']:.1f}/100")
            with col2:
                st.metric("🎯 MAPE", f"{resultado['metricas']['mape']:.2f}%")
            with col3:
                st.metric("📊 R²", f"{resultado['metricas']['r2']:.4f}")
            with col4:
                st.metric("💰 MAE", f"R$ {resultado['metricas']['mae']:.2f}")
            
            st.markdown("---")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Atual", f"R$ {resultado['preco_atual']:.2f}")
            with col_b:
                st.metric("Previsão (1d)", f"R$ {resultado['previsao']:.2f}", f"{resultado['variacao']:+.2f}%")
            with col_c:
                st.metric("Regime", resultado['regime'])
            
            if resultado['sentimento'] is not None and len(resultado['noticias']) > 0:
                st.markdown("---")
                st.subheader("📰 Sentimento")
                
                sent = resultado['sentimento']
                col1, col2 = st.columns(2)
                with col1:
                    if sent > 0.2:
                        st.success(f"😊 POSITIVO: {sent:.2f}")
                    elif sent < -0.2:
                        st.error(f"😞 NEGATIVO: {sent:.2f}")
                    else:
                        st.info(f"😐 NEUTRO: {sent:.2f}")
                with col2:
                    st.metric("Notícias", len(resultado['noticias']))
            
            st.session_state['resultado'] = resultado
            st.session_state['df_dados'] = df_dados
            
            # BOTÃO DOWNLOAD PDF
            if gerar_pdf and PDF_DISPONIVEL:
                st.markdown("---")
                st.subheader("📄 Relatório PDF")
                
                with st.spinner("🔄 Gerando PDF..."):
                    pdf_buffer = gerar_relatorio_pdf(resultado, df_dados, empresa_selecionada)
                
                if pdf_buffer:
                    st.download_button(
                        label="📥 Download Relatório PDF",
                        data=pdf_buffer,
                        file_name=f"relatorio_{empresa_selecionada.lower().replace(' ', '_')}_{dt.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
                    st.success("✅ PDF pronto para download!")
                else:
                    st.error("❌ Erro ao gerar PDF")
            elif gerar_pdf and not PDF_DISPONIVEL:
                st.warning("⚠️ Instale: pip install reportlab")


with tab2:
    st.header("🏆 Backtesting")
    
    if st.button("🏆 Executar", type="primary"):
        if 'resultado' not in st.session_state:
            st.warning("⚠️ Gere previsão primeiro")
        else:
            resultado = st.session_state['resultado']
            df_dados = st.session_state['df_dados']
            
            with st.spinner("🔄 Backtesting..."):
                backtest = backtesting_completo(df_dados, resultado['df_features'], resultado['modelo_info'], 1)
            
            st.success("✅ Concluído!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💰 Inicial", f"R$ {backtest['capital_inicial']:,.2f}")
            with col2:
                st.metric("💼 Final", f"R$ {backtest['capital_final']:,.2f}", f"{backtest['retorno_estrategia']:+.2f}%")
            with col3:
                st.metric("📊 B&H", f"R$ {backtest['capital_buyhold']:,.2f}", f"{backtest['retorno_buyhold']:+.2f}%")
            with col4:
                st.metric("🎯 Vantagem", f"{backtest['vantagem']:+.2f}pp")
            
            st.markdown("---")
            
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("🔢 Trades", backtest['num_trades'])
            with col_b:
                st.metric("🎯 Win Rate", f"{backtest['win_rate']:.1f}%")
            with col_c:
                st.metric("📊 Sharpe", f"{backtest['sharpe_ratio']:.2f}")
            with col_d:
                st.metric("📉 Max DD", f"{backtest['max_drawdown']:.2f}%")

with tab3:
    st.header("📊 Dashboard")
    
    if st.button("📊 Gerar", type="primary"):
        with st.spinner("🔄 Criando..."):
            fig = criar_dashboard_profissional(df_dados)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("📈 Multi-Horizonte")
    
    if 'resultado' in st.session_state:
        resultado = st.session_state['resultado']
        
        st.subheader("🎯 Previsões Múltiplas")
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        for i, col in enumerate([col_m1, col_m2, col_m3, col_m4, col_m5]):
            if i < len(resultado['previsoes_multi']):
                p = resultado['previsoes_multi'][i]
                with col:
                    st.metric(f"{p['horizonte']}d", f"R$ {p['previsao']:.2f}", f"{p['variacao_pct']:+.2f}%")
        
        fig = criar_grafico_multi_horizonte(resultado['preco_atual'], resultado['previsoes_multi'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Gere previsão primeiro")

with tab5:
    st.header("📉 Walk-Forward")
    
    if st.button("📉 Executar", type="primary"):
        if 'resultado' not in st.session_state:
            st.warning("⚠️ Gere previsão primeiro")
        else:
            resultado = st.session_state['resultado']
            
            with st.spinner("🔄 Walk-Forward..."):
                df_wf = walk_forward_analysis(df_dados, resultado['df_features'], resultado['features_cols'], n_splits=5)
            
            st.success("✅ Concluído!")
            st.dataframe(df_wf, use_container_width=True, hide_index=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Médio", f"{df_wf['r2'].mean():.3f}")
            with col2:
                st.metric("MAPE Médio", f"{df_wf['mape'].mean():.1f}%")
            
            fig = criar_grafico_walkforward(df_wf)
            st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.header("🎲 Monte Carlo")
    
    if st.button("🎲 Executar", type="primary"):
        with st.spinner("🔄 10.000 simulações..."):
            retornos = df_dados['Close'].pct_change().dropna()
            mc = monte_carlo_simulation(df_dados['Close'].iloc[-1], retornos.mean(), retornos.std(), dias=30, simulacoes=10000)
        
        st.success("✅ Concluído!")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Média (30d)", f"R$ {mc['media']:.2f}")
        with col2:
            st.metric("Prob Alta", f"{mc['prob_alta']:.1f}%")
        with col3:
            st.metric("P5", f"R$ {mc['p5']:.2f}")
        with col4:
            st.metric("P95", f"R$ {mc['p95']:.2f}")
        
        fig = criar_grafico_montecarlo(mc)
        st.plotly_chart(fig, use_container_width=True)

with tab7:
    st.header("💼 Portfolio")
    
    tickers_selecionados = st.multiselect(
        "Escolha ações:",
        list(empresas_disponiveis.keys()),
        default=list(empresas_disponiveis.keys())[:3]
    )
    
    if st.button("💼 Otimizar", type="primary"):
        if len(tickers_selecionados) < 2:
            st.warning("⚠️ Selecione 2+ ações")
        else:
            tickers_lista = [empresas_disponiveis[t] for t in tickers_selecionados]
            
            with st.spinner("🔄 Otimizando..."):
                portfolio = portfolio_optimizer(tickers_lista, start_date, end_date)
            
            if portfolio:
                st.success("✅ Otimizado!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Retorno", f"{portfolio['retorno_esperado']:.2f}%")
                with col2:
                    st.metric("Risco", f"{portfolio['risco']:.2f}%")
                with col3:
                    st.metric("Sharpe", f"{portfolio['sharpe']:.2f}")
                
                st.markdown("---")
                st.subheader("🎯 Alocação")
                for ticker_nome, peso in zip(tickers_selecionados, portfolio['pesos']):
                    st.progress(peso, text=f"{ticker_nome}: {peso*100:.1f}%")

with tab8:
    st.header("📰 Google News")
    
    if 'resultado' in st.session_state:
        resultado = st.session_state['resultado']
        
        if resultado['sentimento'] is not None and len(resultado['noticias']) > 0:
            sent = resultado['sentimento']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if sent > 0.2:
                    st.success("😊 POSITIVO")
                elif sent < -0.2:
                    st.error("😞 NEGATIVO")
                else:
                    st.info("😐 NEUTRO")
            with col2:
                st.metric("Score", f"{sent:.2f}")
            with col3:
                st.metric("Notícias", len(resultado['noticias']))
            
            st.markdown("---")
            
            for i, n in enumerate(resultado['noticias'], 1):
                emoji = "🟢" if n['sentimento'] > 0.2 else "🔴" if n['sentimento'] < -0.2 else "⚪"
                with st.expander(f"{i}. {emoji} {n['titulo']}", expanded=(i <= 3)):
                    st.write(f"**Sentimento:** {n['sentimento']:.2f}")
                    if n.get('data'):
                        st.write(f"**Data:** {n['data']}")
                    if n.get('link'):
                        st.markdown(f"[🔗 Ler notícia completa]({n['link']})")
        else:
            st.info("ℹ️ Ative Google News na sidebar")
    else:
        st.info("👆 Gere previsão primeiro")

with tab9:
    st.header("🔬 Análise")
    
    if 'resultado' in st.session_state:
        resultado = st.session_state['resultado']
        
        # Risk Metrics
        st.subheader("📊 Risk Metrics")
        retornos = df_dados['Close'].pct_change().dropna()
        metrics = calcular_risk_metrics(retornos)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sharpe", f"{metrics['sharpe']:.2f}")
        with col2:
            st.metric("Sortino", f"{metrics['sortino']:.2f}")
        with col3:
            st.metric("Calmar", f"{metrics['calmar']:.2f}")
        with col4:
            st.metric("Max DD", f"{metrics['max_drawdown']:.2f}%")
        
        st.markdown("---")
        
        # Feature Importance
        if resultado['top_features']:
            st.subheader("🎯 Feature Importance")
            fig = criar_grafico_feature_importance(resultado['top_features'])
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Análise Erro
        st.subheader("📉 Análise Erro")
        fig = criar_grafico_analise_erro(resultado['y_test'], resultado['y_pred'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Correlação
        st.subheader("🔥 Correlação")
        fig_corr = criar_grafico_correlacao(resultado['df_features'], resultado['features_cols'])
        st.plotly_chart(fig_corr, use_container_width=True)
    
    else:
        st.info("👆 Gere previsão primeiro")


with tab10:
    st.header("🔥 Alto Impacto - Análises Avançadas")
    
    if 'resultado' in st.session_state:
        resultado = st.session_state['resultado']
        
        # SIGNAL STRENGTH
        st.subheader("🎯 Signal Strength Indicator")
        ss = resultado['signal_strength']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if ss['cor'] == 'success':
                st.success(f"Score: {ss['score']:.0f}/100")
            elif ss['cor'] == 'error':
                st.error(f"Score: {ss['score']:.0f}/100")
            elif ss['cor'] == 'warning':
                st.warning(f"Score: {ss['score']:.0f}/100")
            else:
                st.info(f"Score: {ss['score']:.0f}/100")
        with col2:
            st.metric("Recomendação", ss['recomendacao'])
        with col3:
            st.metric("Sinais", len(ss['sinais']))
        
        st.markdown("**Sinais Detectados:**")
        for sinal in ss['sinais']:
            st.write(f"• {sinal['fonte']}: {sinal['sinal']} (Força: {sinal['forca']:+.1f})")
        
        st.markdown("---")
        
        # PADRÕES CANDLESTICK
        if resultado['padroes_candlestick']:
            st.subheader("🕯️ Padrões Candlestick Detectados")
            for padrao in resultado['padroes_candlestick']:
                emoji = "🟢" if padrao['tipo'] == 'COMPRA' else "🔴" if padrao['tipo'] == 'VENDA' else "⚪"
                st.write(f"{emoji} **{padrao['padrao']}** - {padrao['tipo']} (Confiança: {padrao['confianca']}%)")
        
        st.markdown("---")
        
        # SUPPORT & RESISTANCE
        if resultado['suporte_resistencia']:
            st.subheader("📈 Support & Resistance")
            sr = resultado['suporte_resistencia']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Suportes:**")
                for s in sr['suportes']:
                    st.write(f"• R$ {s:.2f}")
                if sr.get('prox_suporte'):
                    st.metric("Próximo Suporte", f"R$ {sr['prox_suporte']:.2f}")
            
            with col2:
                st.write("**Resistências:**")
                for r in sr['resistencias']:
                    st.write(f"• R$ {r:.2f}")
                if sr.get('prox_resistencia'):
                    st.metric("Próxima Resistência", f"R$ {sr['prox_resistencia']:.2f}")
            
            if sr.get('fibonacci'):
                st.write("**Fibonacci:**")
                for nivel, valor in sr['fibonacci'].items():
                    st.write(f"• {nivel}: R$ {valor:.2f}")
        
        st.markdown("---")
        
        # GARCH
        if resultado.get('garch'):
            st.subheader("📊 GARCH - Previsão de Volatilidade")
            garch = resultado['garch']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Vol Atual", f"{garch['vol_atual']:.2f}%")
            with col2:
                st.metric("Vol Média (5d)", f"{garch['vol_media']:.2f}%")
            with col3:
                st.metric("Vol Máxima", f"{garch['vol_max']:.2f}%")
        
        st.markdown("---")
        
        # CORRELAÇÃO
        if resultado.get('correlacoes'):
            st.subheader("🌐 Correlação com Índices")
            for nome, corr_data in resultado['correlacoes'].items():
                if corr_data:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**{nome}**")
                    with col2:
                        st.metric("Correlação", f"{corr_data['correlacao']:.3f}")
                    with col3:
                        st.metric("Beta", f"{corr_data['beta']:.3f}")
                    st.write(f"_{corr_data['interpretacao']}_")
    
    else:
        st.info("👆 Gere previsão primeiro")


with tab11:
    st.header("🧠 Deep Learning - Análise Avançada")
    
    if 'resultado' in st.session_state:
        resultado = st.session_state['resultado']
        
        if resultado.get('modelos_dl'):
            # PREVISÕES DOS MODELOS
            st.subheader("🤖 Previsões dos Modelos Neurais")
            
            if resultado.get('previsao_dl'):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Previsão Ensemble Neural", f"R$ {resultado['previsao_dl']:.2f}")
                with col2:
                    variacao_dl = ((resultado['previsao_dl'] - resultado['preco_atual']) / resultado['preco_atual']) * 100
                    st.metric("Variação Esperada", f"{variacao_dl:+.2f}%")
                with col3:
                    st.metric("Preço Atual", f"R$ {resultado['preco_atual']:.2f}")
            
            st.markdown("---")
            
            # PREVISÕES INDIVIDUAIS
            if resultado.get('previsoes_individuais_dl'):
                st.subheader("📊 Previsões Individuais")
                
                prev_ind = resultado['previsoes_individuais_dl']
                cols = st.columns(len(prev_ind))
                
                for idx, (nome, valor) in enumerate(prev_ind.items()):
                    with cols[idx]:
                        variacao = ((valor - resultado['preco_atual']) / resultado['preco_atual']) * 100
                        st.metric(nome, f"R$ {valor:.2f}", f"{variacao:+.2f}%")
            
            st.markdown("---")
            
            # MÉTRICAS DE PERFORMANCE
            if resultado.get('metricas_dl'):
                st.subheader("📈 Performance dos Modelos")
                
                metricas_df = []
                for nome, metricas in resultado['metricas_dl'].items():
                    metricas_df.append({
                        'Modelo': nome,
                        'R²': f"{metricas['r2']:.4f}",
                        'MAPE': f"{metricas['mape']:.2f}%",
                        'MAE': f"R$ {metricas['mae']:.2f}",
                        'RMSE': f"R$ {metricas['rmse']:.2f}"
                    })
                
                df_metricas = pd.DataFrame(metricas_df)
                st.dataframe(df_metricas, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # LOSS CURVES
            if resultado.get('historicos_dl'):
                st.subheader("📉 Curvas de Treinamento")
                
                for nome, historico in resultado['historicos_dl'].items():
                    st.markdown(f"**{nome}:**")
                    
                    fig = go.Figure()
                    
                    # Training Loss
                    fig.add_trace(go.Scatter(
                        y=historico.history['loss'],
                        mode='lines',
                        name='Training Loss',
                        line=dict(color='#64b5f6', width=2)
                    ))
                    
                    # Validation Loss
                    fig.add_trace(go.Scatter(
                        y=historico.history['val_loss'],
                        mode='lines',
                        name='Validation Loss',
                        line=dict(color='#f44336', width=2, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"Loss Curve - {nome}",
                        xaxis_title="Epoch",
                        yaxis_title="Loss (MSE)",
                        template="plotly_white",
                        hovermode='x unified',
                        height=350
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # COMPARAÇÃO COM MODELOS TRADICIONAIS
            st.subheader("⚖️ Deep Learning vs Modelos Tradicionais")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🤖 Deep Learning:**")
                if resultado.get('previsao_dl'):
                    st.metric("Previsão", f"R$ {resultado['previsao_dl']:.2f}")
                if resultado.get('metricas_dl'):
                    melhor_r2 = max([m['r2'] for m in resultado['metricas_dl'].values()])
                    st.metric("Melhor R²", f"{melhor_r2:.4f}")
            
            with col2:
                st.markdown("**📊 Tradicionais (ML):**")
                st.metric("Previsão", f"R$ {resultado['previsao']:.2f}")
                st.metric("R²", f"{resultado['metricas']['r2']:.4f}")
            
            # Diferença
            if resultado.get('previsao_dl'):
                diff = abs(resultado['previsao_dl'] - resultado['previsao'])
                diff_pct = (diff / resultado['preco_atual']) * 100
                st.info(f"📊 **Diferença entre previsões:** R$ {diff:.2f} ({diff_pct:.2f}%)")
        
        else:
            st.warning("⚠️ Nenhum modelo Deep Learning foi treinado.")
            st.info("💡 Ative pelo menos um modelo DL na sidebar e gere nova previsão.")
    
    else:
        st.info("👆 Gere previsão primeiro")
