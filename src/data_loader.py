"""
Módulo de Carregamento de Dados (Versão YFinance).
Substitui a API CoinGecko por Yahoo Finance (gratuito e sem API Key).
Realiza resampling automático para timeframes não nativos (ex: 4h).
"""
import pandas as pd
import yfinance as yf
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

try:
    from config import Config
except ImportError:
    class Config:
        MAX_DOWNLOAD_TENTATIVAS = 3
        RETRY_DELAY_SEGUNDOS = 5

# ==============================================================================
# 1. HELPERS DE DATA E RESAMPLING
# ==============================================================================


def safe_to_naive_datetime(date_series: pd.Series) -> pd.Series:
    """Remove fuso horário para evitar erros de comparação no Pandas."""
    try:
        return pd.to_datetime(date_series, errors='coerce', utc=True).dt.tz_localize(None)
    except (ValueError, TypeError):
        return pd.to_datetime(date_series, errors='coerce')


def _resample_ohlcv(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """
    Transforma dados de tempo menor em maior (ex: 1h -> 4h).
    """
    if df.empty:
        return df

    # Mapeia string do config para regra do pandas (ex: '4h' -> '4H', '1d' -> '1D')
    rule = target_interval.upper().replace(
        'M', 'T') if 'm' in target_interval and 'mo' not in target_interval else target_interval.upper()

    # Dicionário de agregação
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    # Se tiver colunas extras (como Adj Close), ignora no resample básico
    cols = {c: agg_dict[c] for c in agg_dict if c in df.columns}

    try:
        df_resampled = df.resample(rule).agg(cols)
        df_resampled.dropna(inplace=True)  # Remove candles incompletos gerados
        return df_resampled
    except Exception as e:
        logging.warning(
            f"Falha ao fazer resample para {target_interval}: {e}. Retornando original.")
        return df


def _map_interval_to_yf(interval: str) -> str:
    """
    Mapeia intervalo do nosso sistema para o mais próximo disponível no YFinance.
    YF suporta: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    """
    if interval == '4h':
        return '1h'  # Baixa 1h e depois fazemos resample
    if interval == '12h':
        return '1h'
    if interval == '3d':
        return '1d'
    if 'm' in interval and interval not in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']:
        return '1m'  # Fallback para minuto
    return interval

# ==============================================================================
# 2. FUNÇÕES PRINCIPAIS DE DOWNLOAD
# ==============================================================================


def _process_yf_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e padroniza o DataFrame retornado pelo YFinance."""
    if df is None or df.empty:
        return pd.DataFrame()

    # --- CORREÇÃO DE SEGURANÇA (MultiIndex) ---
    # Se o yfinance retornar colunas duplas (ex: Preço, Ticker), pega só o primeiro nível
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # ------------------------------------------

    # Reset index se a data for o índice (para garantir que temos acesso à coluna de data)
    if df.index.name in ['Date', 'Datetime', 'date', 'datetime']:
        df = df.reset_index()

    # Padronizar nomes de colunas para minúsculo
    df.columns = [str(c).lower() for c in df.columns]

    # Normalização de nomes de coluna de data
    rename_map = {'date': 'timestamp', 'datetime': 'timestamp'}
    df.rename(columns=rename_map, inplace=True)

    # Verifica se a coluna de tempo existe, senão tenta assumir a primeira
    if 'timestamp' not in df.columns:
        # Se não achou 'timestamp', assume que a primeira coluna é a data
        cols = list(df.columns)
        if cols:
            df.rename(columns={cols[0]: 'timestamp'}, inplace=True)

    if 'timestamp' in df.columns:
        df['timestamp'] = safe_to_naive_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    # Manter apenas colunas essenciais
    cols_desejadas = ['open', 'high', 'low', 'close', 'volume']
    # Filtra apenas as colunas que realmente existem no df baixado
    cols_finais = [c for c in cols_desejadas if c in df.columns]

    if not cols_finais:
        return pd.DataFrame()  # Retorna vazio se não achou as colunas OHLC

    df = df[cols_finais]

    # Converter para float e limpar
    df = df.astype(float)
    if 'high' in df.columns:
        df = df[df['high'] > 0]

    return df


def buscar_dados(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Baixa dados históricos (Para o Gerador).
    Ajusta automaticamente o período para limites do Yahoo Finance.
    """
    yf_interval = _map_interval_to_yf(interval)

    # --- CORREÇÃO DO LIMITE DO YAHOO ---
    # Se o intervalo tiver 'm' (minutos) e não for '1mo' (mensal),
    # o Yahoo limita a 60 dias. Forçamos 59d para garantir.
    if 'm' in yf_interval and 'mo' not in yf_interval:
        if period not in ['1d', '5d', '1mo']:  # Se pediu algo longo (1y, 2y)
            logging.warning(
                f"Yahoo Limite: {interval} suporta máx 60 dias. Ajustando período de {period} para '59d'.")
            period = '59d'
    # -----------------------------------

    # Ajuste de período para garantir dados suficientes para o resample de 4h
    if interval == '4h' and period in ['1mo', '3mo']:
        period = '6mo'

    for tentativa in range(Config.MAX_DOWNLOAD_TENTATIVAS):
        try:
            df = yf.download(
                tickers=ticker,
                period=period,
                interval=yf_interval,
                progress=False,
                auto_adjust=True
            )

            df = _process_yf_data(df)

            if interval != yf_interval:
                df = _resample_ohlcv(df, interval)

            if not df.empty:
                return df

        except Exception as e:
            logging.warning(f"Erro YF {ticker} (t={tentativa+1}): {e}")
            time.sleep(Config.RETRY_DELAY_SEGUNDOS)

    logging.error(f"Falha total download {ticker}")
    return pd.DataFrame()


def buscar_dados_range(ticker: str, start_dt: datetime, end_dt: datetime, interval: str) -> pd.DataFrame:
    """
    Baixa dados por data específica (Para o Visualizador/Labeler).
    """
    yf_interval = _map_interval_to_yf(interval)

    # YFinance precisa de strings ou datetime objects
    s_str = start_dt.strftime('%Y-%m-%d')
    # Adiciona um dia no fim para garantir que o último candle venha (limitação do YF)
    e_str = (end_dt + timedelta(days=1)).strftime('%Y-%m-%d')

    for tentativa in range(Config.MAX_DOWNLOAD_TENTATIVAS):
        try:
            df = yf.download(
                tickers=ticker,
                start=s_str,
                end=e_str,
                interval=yf_interval,
                progress=False,
                auto_adjust=True
            )

            df = _process_yf_data(df)

            if interval != yf_interval:
                df = _resample_ohlcv(df, interval)

            if not df.empty:
                # Filtra exatamente o range pedido (o download pode trazer excesso)
                # Como removemos o fuso em _process_yf_data, a comparação é segura
                mask = (df.index >= start_dt) & (df.index <= end_dt)
                return df.loc[mask]

        except Exception as e:
            logging.warning(f"Erro Range YF {ticker}: {e}")
            time.sleep(Config.RETRY_DELAY_SEGUNDOS)

    return pd.DataFrame()
