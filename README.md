# 📈 QuantumStock

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Sistema inteligente de previsão de ações usando Machine Learning e Deep Learning**

[🚀 Começar](#-instalação) • [📊 Features](#-features) • [🎯 Como Usar](#-como-usar) 

</div>

---

## 🎯 Sobre

QuantumStock é um sistema completo de análise quantitativa e previsão de ações que combina 14 modelos de IA para gerar previsões precisas e análises profissionais do mercado.

**Principais características:**
- 🤖 14 modelos de ML/DL (Ridge, RF, XGBoost, LightGBM, CatBoost, Transformer, BiLSTM, GRU e mais)
- 📊 50+ indicadores técnicos automatizados
- 🔬 Validação robusta (Backtesting, Walk-Forward, Monte Carlo)
- 📄 Relatórios PDF profissionais
- 🧠 Análise de sentimento de notícias
- 📈 Dashboard interativo com 11 visualizações

> ⚠️ **Disclaimer:** Este sistema é para fins educacionais. Não constitui recomendação de investimento.

---

## 🚀 Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/quantum-stock.git
cd quantum-stock

# Instale as dependências
pip install -r requirements.txt

# Execute o sistema
streamlit run app.py
```

Acesse em: `http://localhost:8501`

---

## 📊 Features

### 🤖 Modelos de IA

**Machine Learning:**
- Ridge, Random Forest, Gradient Boosting
- XGBoost, LightGBM, CatBoost
- Stacking Ensemble, Optuna AutoML

**Deep Learning:**
- Transformer (Multi-Head Attention)
- BiLSTM, GRU, CNN-LSTM
- Ensemble Neural

### 📈 Análises

- **Indicadores Técnicos:** SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, ADX, Stochastic
- **Padrões:** 7 padrões de candlestick
- **Volatilidade:** GARCH forecasting
- **Suporte/Resistência:** Detecção automática + Fibonacci
- **Correlação:** IBOV, S&P500, USD/BRL

### 🔬 Validação

- Backtesting com simulação de trades
- Walk-Forward Analysis
- Monte Carlo Simulation (10k cenários)
- Portfolio Optimization
- Risk Metrics (Sharpe, Sortino, VaR, CVaR)

### 📄 Outputs

- Dashboard interativo com 11 tabs
- Relatórios PDF automáticos
- Alertas inteligentes

---

## 🎯 Como Usar

### 1. Configure

```
Sidebar:
• Empresa: Petrobras
• Dias Histórico: 730

Modelos:
☑️ Modelos por Regime
☑️ Stacking Ensemble
☑️ LightGBM

Features:
☑️ Google News
☑️ GARCH
☑️ Candlestick
☑️ Support/Resistance
```

### 2. Gere Previsão

```
[🚀 Gerar Previsão]
```

### 3. Analise Resultados

Explore as 11 tabs: Previsão, Backtesting, Dashboard, Multi-Horizonte, Walk-Forward, Monte Carlo, Portfolio, Notícias, Análise, Alto Impacto, Deep Learning

### 4. Baixe PDF

```
[📥 Download Relatório PDF]
```

---

## 📊 Performance

| Modelo | R² | MAPE | Tempo |
|--------|-----|------|-------|
| LightGBM | 0.35 | 5.2% | 3s |
| GRU | 0.42 | 4.5% | 45s |
| Transformer | 0.48 | 3.8% | 60s |
| **Ensemble** | **0.52** | **3.5%** | **90s** |

*Métricas com 730 dias de histórico em PETR4*

---

## 🛠️ Tecnologias

- **Interface:** Streamlit, Plotly
- **Data:** Pandas, NumPy, yfinance
- **ML:** scikit-learn, LightGBM, CatBoost, XGBoost, Optuna
- **DL:** TensorFlow/Keras
- **Análise:** TA-Lib, ARCH, VADER
- **Reports:** ReportLab

---

## 🐛 Troubleshooting

**R² Negativo?**
→ Aumente dias históricos (1095) ou reduza sequência DL (30)

**Sistema Lento?**
→ Desligue Optuna e Deep Learning

**Erro TensorFlow?**
→ `pip install tensorflow==2.15.0`
---

## 📄 Licença

MIT License - Ver [LICENSE](LICENSE)

---

## 👤 Autor

**Lucas Stalter**
- GitHub: https://github.com/Lucasstalter/
- LinkedIn: www.linkedin.com/in/lucas-martins-stalter


---

<div align="center">

**QuantumStock** - Previsões quânticas para o mercado de ações

[⬆ Voltar ao topo](#-quantumstock)

</div>
