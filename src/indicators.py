"""
Módulo de Indicadores Técnicos e Lógica Matemática.
Contém funções para calcular ZigZag, RSI, MACD, Estocástico e validar condições de preço.
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
import os
from typing import List, Dict, Any, Optional, Tuple

try:
    from config import Config
except ImportError:
    import sys
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))
    from config import Config


# ==============================================================================
# 1. CÁLCULO DE INDICADORES EM LOTE (Pandas TA)
# ==============================================================================

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula e cacheia indicadores técnicos no DataFrame.
    Gera colunas: RSI_*, MACD_*, STOCH_*, OBV, ATR_14.
    """
    try:
        # RSI (close, high, low)
        rsi_len = getattr(Config, 'RSI_LENGTH', 14)
        for src in ['close', 'high', 'low']:
            try:
                col_name = f'RSI_{rsi_len}_{src.upper()}'
                df[col_name] = ta.rsi(df[src], length=rsi_len)
            except Exception:
                pass

        # MACD (close)
        try:
            macd_fast = getattr(Config, 'MACD_FAST', 12)
            macd_slow = getattr(Config, 'MACD_SLOW', 26)
            macd_signal = getattr(Config, 'MACD_SIGNAL', 9)
            macd_df = df.ta.macd(fast=macd_fast, slow=macd_slow,
                                 signal=macd_signal, append=False)
            if macd_df is not None:
                # pandas_ta retorna colunas com nomes específicos, concatenamos ao df
                df = pd.concat([df, macd_df], axis=1)
        except Exception:
            pass

        # Stochastic Oscillator
        try:
            k = getattr(Config, 'STOCH_K', 14)
            d = getattr(Config, 'STOCH_D', 3)
            smooth_k = getattr(Config, 'STOCH_SMOOTH_K', 3)
            stoch_df = df.ta.stoch(
                high=df['high'], low=df['low'], close=df['close'], k=k, d=d, smooth_k=smooth_k)
            if stoch_df is not None:
                df = pd.concat([df, stoch_df], axis=1)
        except Exception:
            pass

        # OBV
        try:
            obv_series = df.ta.obv(append=False)
            if obv_series is not None:
                df['OBV'] = obv_series
        except Exception:
            pass

        # ATR
        try:
            atr_series = df.ta.atr(length=14, append=False)
            if atr_series is not None:
                df['ATR_14'] = atr_series
        except Exception:
            pass

    except Exception as e:
        print(f"Erro ao calcular indicadores: {e}")
        # Não quebra o fluxo, apenas segue com o que conseguiu calcular
        pass

    return df


# ==============================================================================
# 2. ALGORITMO ZIGZAG (Pivôs)
# ==============================================================================

def calcular_zigzag_oficial(df: pd.DataFrame, depth: int, deviation_percent: float) -> List[Dict[str, Any]]:
    """
    Calcula pivôs ZigZag exigindo alternância e desvio percentual mínimo.
    Retorna uma lista de dicionários [{'idx': timestamp, 'preco': float, 'tipo': 'PICO'|'VALE'}, ...].
    """
    peak_series, valley_series = df['high'], df['low']
    window_size = 2 * depth + 1

    # Identifica máximas e mínimas locais na janela definida pelo depth
    rolling_max = peak_series.rolling(
        window=window_size, center=True, min_periods=1).max()
    rolling_min = valley_series.rolling(
        window=window_size, center=True, min_periods=1).min()

    candidate_peaks_df = df[peak_series == rolling_max]
    candidate_valleys_df = df[valley_series == rolling_min]

    candidates = []
    for idx, row in candidate_peaks_df.iterrows():
        candidates.append(
            {'idx': idx, 'preco': row[peak_series.name], 'tipo': 'PICO'})
    for idx, row in candidate_valleys_df.iterrows():
        candidates.append(
            {'idx': idx, 'preco': row[valley_series.name], 'tipo': 'VALE'})

    # Ordena por data
    candidates = sorted(candidates, key=lambda x: x['idx'])

    if len(candidates) < 2:
        return []

    # Filtragem de Alternância e Desvio
    confirmed_pivots = [candidates[0]]
    last_pivot = candidates[0]

    for i in range(1, len(candidates)):
        candidate = candidates[i]

        # --- Lógica de desempate para pivôs no mesmo índice ---
        if candidate['idx'] == last_pivot['idx']:
            if candidate['tipo'] != last_pivot['tipo']:
                # Se houver conflito PICO/VALE no mesmo candle, tenta manter alternância com o anterior
                if len(confirmed_pivots) >= 2:
                    prev_prev = confirmed_pivots[-2]
                    if (candidate['tipo'] != prev_prev['tipo'] and last_pivot['tipo'] == prev_prev['tipo']):
                        confirmed_pivots[-1] = candidate
                        last_pivot = candidate
                else:
                    confirmed_pivots[-1] = candidate
                    last_pivot = candidate
            else:
                # Mesmo tipo e índice: mantém o preço mais extremo
                if (candidate['tipo'] == 'PICO' and candidate['preco'] > last_pivot['preco']) or \
                   (candidate['tipo'] == 'VALE' and candidate['preco'] < last_pivot['preco']):
                    confirmed_pivots[-1] = candidate
                    last_pivot = candidate
            continue

        # --- Lógica de Atualização do Pivô Atual (Higher High ou Lower Low sem reversão) ---
        if candidate['tipo'] == last_pivot['tipo']:
            if (candidate['tipo'] == 'PICO' and candidate['preco'] > last_pivot['preco']) or \
               (candidate['tipo'] == 'VALE' and candidate['preco'] < last_pivot['preco']):
                confirmed_pivots[-1], last_pivot = candidate, candidate
            continue

        if last_pivot['preco'] == 0:
            continue

        # --- Confirmação de Reversão (Novo Pivô) ---
        price_dev = abs(
            candidate['preco'] - last_pivot['preco']) / last_pivot['preco'] * 100
        if price_dev >= deviation_percent:
            confirmed_pivots.append(candidate)
            last_pivot = candidate

    # --- Extensão para o Último Candle (Opcional, configurável em Config) ---
    if getattr(Config, 'ZIGZAG_EXTEND_TO_LAST_BAR', True) and confirmed_pivots:
        last_confirmed_pivot = confirmed_pivots[-1]
        last_bar = df.iloc[-1]

        if last_confirmed_pivot['tipo'] == 'PICO':
            if last_bar['high'] > last_confirmed_pivot['preco']:
                last_confirmed_pivot['preco'] = last_bar['high']
                last_confirmed_pivot['idx'] = df.index[-1]
            else:
                potential_pivot = {
                    'idx': df.index[-1], 'tipo': 'VALE', 'preco': last_bar['low']}
                if potential_pivot['idx'] != last_confirmed_pivot['idx']:
                    min_ext_dev = deviation_percent * \
                        getattr(Config, 'ZIGZAG_EXTENSION_DEVIATION_FACTOR', 0.25)
                    if last_confirmed_pivot['preco'] != 0:
                        ext_dev = abs(
                            potential_pivot['preco'] - last_confirmed_pivot['preco']) / last_confirmed_pivot['preco'] * 100
                        if ext_dev >= min_ext_dev:
                            confirmed_pivots.append(potential_pivot)
        else:  # last pivot is a VALLEY
            if last_bar['low'] < last_confirmed_pivot['preco']:
                last_confirmed_pivot['preco'] = last_bar['low']
                last_confirmed_pivot['idx'] = df.index[-1]
            else:
                potential_pivot = {
                    'idx': df.index[-1], 'tipo': 'PICO', 'preco': last_bar['high']}
                if potential_pivot['idx'] != last_confirmed_pivot['idx']:
                    min_ext_dev = deviation_percent * \
                        getattr(Config, 'ZIGZAG_EXTENSION_DEVIATION_FACTOR', 0.25)
                    if last_confirmed_pivot['preco'] != 0:
                        ext_dev = abs(
                            potential_pivot['preco'] - last_confirmed_pivot['preco']) / last_confirmed_pivot['preco'] * 100
                        if ext_dev >= min_ext_dev:
                            confirmed_pivots.append(potential_pivot)

    return confirmed_pivots


# ==============================================================================
# 3. VALIDAÇÕES DE CONTEXTO DE PREÇO (Head Extremes, Breakouts)
# ==============================================================================

def is_head_extreme(df: pd.DataFrame, head_pivot: Dict, avg_pivot_dist_bars: int) -> bool:
    """Verifica se a cabeça é um extremo (max/min) único em uma janela de tempo."""
    base_lookback = int(avg_pivot_dist_bars *
                        getattr(Config, 'HEAD_EXTREME_LOOKBACK_FACTOR', 1))
    lookback_bars = max(base_lookback, getattr(
        Config, 'HEAD_EXTREME_LOOKBACK_MIN_BARS', 30))

    if lookback_bars <= 0:
        return True

    try:
        head_loc = df.index.get_loc(head_pivot['idx'])
        start_loc = max(0, head_loc - lookback_bars)
        end_loc = min(len(df), head_loc + lookback_bars + 1)

        context_df = df.iloc[start_loc:end_loc]

        # Remove o próprio pivô da comparação
        try:
            context_wo_head = context_df.drop(index=head_pivot['idx'])
        except Exception:
            context_wo_head = context_df

        if context_wo_head.empty:
            return False

        if head_pivot['tipo'] == 'PICO':
            return head_pivot['preco'] > context_wo_head['high'].max()
        else:  # VALE
            return head_pivot['preco'] < context_wo_head['low'].min()

    except (KeyError, IndexError, Exception):
        return False


def is_head_extreme_past_only(df: pd.DataFrame, head_pivot: Dict, avg_pivot_dist_bars: int) -> bool:
    """Verifica se o pivô é extremo considerando apenas dados passados (backtest/realtime)."""
    try:
        base_lookback = int(avg_pivot_dist_bars *
                            getattr(Config, 'HEAD_EXTREME_LOOKBACK_FACTOR', 1))
        lookback_bars = max(base_lookback, getattr(
            Config, 'HEAD_EXTREME_LOOKBACK_MIN_BARS', 30))

        head_loc = df.index.get_loc(head_pivot['idx'])
        start_loc = max(0, head_loc - lookback_bars)
        end_loc = head_loc  # Exclui o pivô atual (olha apenas para trás)

        context_df = df.iloc[start_loc:end_loc]
        if context_df.empty:
            return False

        if head_pivot['tipo'] == 'PICO':
            return head_pivot['preco'] > context_df['high'].max()
        else:
            return head_pivot['preco'] < context_df['low'].min()
    except Exception:
        return False


def find_breakout_index(df: pd.DataFrame, neckline_price: float, start_idx, direction: str, max_bars: int = 60):
    """Encontra o índice onde o preço fecha rompendo a neckline."""
    try:
        if start_idx not in df.index:
            return None
        start_pos = df.index.get_loc(start_idx)
        end_pos = min(len(df) - 1, start_pos + max_bars)

        for pos in range(start_pos + 1, end_pos + 1):
            idx = df.index[pos]
            close_val = float(df.loc[idx, 'close'])
            if direction == 'bearish' and close_val < neckline_price:
                return idx
            if direction == 'bullish' and close_val > neckline_price:
                return idx
        return None
    except Exception:
        return None


def check_breakout_volume(df: pd.DataFrame, breakout_idx, lookback_bars: int = 20, multiplier: float = 1.8) -> bool:
    """Verifica se houve volume expressivo no rompimento."""
    try:
        if breakout_idx not in df.index:
            return False
        pos = df.index.get_loc(breakout_idx)
        start = max(0, pos - lookback_bars)

        base_vol = df.iloc[start:pos]['volume']
        if base_vol.empty:
            return False

        base_mean = float(base_vol.mean())
        breakout_vol = float(df.loc[breakout_idx, 'volume'])

        return base_mean > 0 and breakout_vol >= (multiplier * base_mean)
    except Exception:
        return False


# ==============================================================================
# 4. VALIDACAO DE DIVERGÊNCIAS E OSCILADORES (RSI, MACD, Stoch, OBV)
# ==============================================================================

def assess_rsi_divergence_strength(df: pd.DataFrame, p1_idx, p3_idx, p1_price: float, p3_price: float, direction: str, source_series: pd.Series) -> Tuple[bool, bool]:
    """Avalia divergência de RSI com classificação de força."""
    try:
        rsi_len = getattr(Config, 'RSI_LENGTH', 14)
        src_name = getattr(source_series, 'name', 'close')
        rsi_col = f'RSI_{rsi_len}_{src_name.upper()}'

        # Fallback se coluna não existir
        if rsi_col not in df.columns:
            rsi_series = ta.rsi(source_series, length=rsi_len)
        else:
            rsi_series = df[rsi_col]

        if rsi_series is None or p1_idx not in rsi_series.index or p3_idx not in rsi_series.index:
            return False, False

        rsi1, rsi3 = float(rsi_series.loc[p1_idx]), float(
            rsi_series.loc[p3_idx])
        if np.isnan(rsi1) or np.isnan(rsi3):
            return False, False

        ob = getattr(Config, 'RSI_OVERBOUGHT', 70)
        os_val = getattr(Config, 'RSI_OVERSOLD', 30)
        strong_ob = getattr(Config, 'RSI_STRONG_OVERBOUGHT', 80)
        strong_os = getattr(Config, 'RSI_STRONG_OVERSOLD', 20)
        min_delta = getattr(Config, 'RSI_DIVERGENCE_MIN_DELTA', 5.0)

        # OCO (Topo mais alto no preço, RSI mais baixo)
        if direction == 'bearish':
            valid_start = rsi1 >= ob
            div = (p3_price > p1_price) and (rsi3 < rsi1) and valid_start
            strong = div and (rsi1 >= strong_ob or (rsi1 - rsi3) >= min_delta)
            return div, strong
        else:  # OCOI (Fundo mais baixo no preço, RSI mais alto)
            valid_start = rsi1 <= os_val
            div = (p3_price < p1_price) and (rsi3 > rsi1) and valid_start
            strong = div and (rsi1 <= strong_os or (rsi3 - rsi1) >= min_delta)
            return div, strong
    except Exception:
        return False, False


def check_rsi_divergence(df: pd.DataFrame, p1_idx, p3_idx, p1_price, p3_price, tipo_padrao: str) -> bool:
    """Wrapper simples para checar divergência RSI (booleano)."""
    direction = 'bearish' if tipo_padrao in ('OCO', 'DT', 'TT') else 'bullish'
    # OCO usa High, OCOI usa Low geralmente
    src = df['high'] if direction == 'bearish' else df['low']
    div, _ = assess_rsi_divergence_strength(
        df, p1_idx, p3_idx, p1_price, p3_price, direction, src)
    return bool(div)


def check_macd_divergence(df: pd.DataFrame, p1_idx, p3_idx, p1_price, p3_price, tipo_padrao: str) -> bool:
    """Detecta divergência no Histograma do MACD."""
    try:
        macd_fast = getattr(Config, 'MACD_FAST', 12)
        macd_slow = getattr(Config, 'MACD_SLOW', 26)
        macd_signal = getattr(Config, 'MACD_SIGNAL', 9)
        hist_col = f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}'

        if hist_col not in df.columns or p1_idx not in df.index or p3_idx not in df.index:
            return False

        hist_p1, hist_p3 = df.loc[p1_idx, hist_col], df.loc[p3_idx, hist_col]

        bearish = tipo_padrao in ('OCO', 'DT', 'TT')

        if bearish:  # Preço sobe, MACD Hist desce
            return p3_price > p1_price and hist_p3 < hist_p1
        else:  # Preço cai, MACD Hist sobe
            return p3_price < p1_price and hist_p3 > hist_p1
    except Exception:
        return False


def detect_macd_signal_cross(df: pd.DataFrame, idx_ref, direction: str, lookback_bars: int = 7) -> bool:
    """Verifica cruzamento da linha MACD com Sinal próximo ao índice de referência."""
    try:
        macd_fast = getattr(Config, 'MACD_FAST', 12)
        macd_slow = getattr(Config, 'MACD_SLOW', 26)
        macd_signal = getattr(Config, 'MACD_SIGNAL', 9)
        macd_col = f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'
        sig_col = f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}'

        if macd_col not in df.columns or sig_col not in df.columns:
            return False

        if idx_ref not in df.index:
            return False

        ref_pos = df.index.get_loc(idx_ref)
        start_pos = max(0, ref_pos - lookback_bars)
        window = df.index[start_pos:ref_pos + 1]

        diff = (df.loc[window, macd_col] - df.loc[window, sig_col]).dropna()
        if len(diff) < 2:
            return False

        # Verifica se houve cruzamento nos últimos candles da janela
        prev_val = diff.iloc[-2]
        curr_val = diff.iloc[-1]

        # Cruzou pra baixo
        if direction == 'bearish' and (prev_val >= 0 and curr_val < 0):
            return True
        # Cruzou pra cima
        if direction == 'bullish' and (prev_val <= 0 and curr_val > 0):
            return True

        return False
    except Exception:
        return False


def check_stochastic_confirmation(df: pd.DataFrame, p1_idx, p3_idx, p1_price: float, p3_price: float, direction: str) -> dict:
    """Verifica confirmações do Estocástico (Divergência e Cruzamento)."""
    result = {'valid_estocastico_divergencia': False,
              'valid_estocastico_cross': False}
    try:
        # Tenta achar as colunas do pandas_ta
        k = getattr(Config, 'STOCH_K', 14)
        d = getattr(Config, 'STOCH_D', 3)
        smooth = getattr(Config, 'STOCH_SMOOTH_K', 3)
        k_col = f"STOCHk_{k}_{d}_{smooth}"
        d_col = f"STOCHd_{k}_{d}_{smooth}"

        if k_col not in df.columns:  # Tenta nome default se falhar
            k_col, d_col = 'STOCHk_14_3_3', 'STOCHd_14_3_3'
            if k_col not in df.columns:
                return result

        k_series, d_series = df[k_col], df[d_col]
        if p1_idx not in k_series.index or p3_idx not in k_series.index:
            return result

        k1, k3 = float(k_series.loc[p1_idx]), float(k_series.loc[p3_idx])

        ob = getattr(Config, 'STOCH_OVERBOUGHT', 80)
        os_val = getattr(Config, 'STOCH_OVERSOLD', 20)
        requires_obos = getattr(Config, 'STOCH_DIVERGENCE_REQUIRES_OBOS', True)

        # 1. Divergência
        if direction == 'bearish':
            cond_start = (k1 >= ob) if requires_obos else True
            if cond_start and (p3_price > p1_price) and (k3 < k1):
                result['valid_estocastico_divergencia'] = True
        else:  # bullish
            cond_start = (k1 <= os_val) if requires_obos else True
            if cond_start and (p3_price < p1_price) and (k3 > k1):
                result['valid_estocastico_divergencia'] = True

        # 2. Cruzamento
        lookback = getattr(Config, 'STOCH_CROSS_LOOKBACK_BARS', 7)
        ref_pos = df.index.get_loc(p3_idx)
        start_pos = max(0, ref_pos - lookback)
        window = df.index[start_pos:ref_pos + 1]

        diff = (k_series.loc[window] - d_series.loc[window]).dropna()
        if len(diff) >= 2:
            prev_val = diff.iloc[-2]
            curr_val = diff.iloc[-1]
            if direction == 'bearish' and k1 >= ob and (prev_val >= 0 and curr_val < 0):
                result['valid_estocastico_cross'] = True
            if direction == 'bullish' and k1 <= os_val and (prev_val <= 0 and curr_val > 0):
                result['valid_estocastico_cross'] = True

    except Exception:
        pass
    return result


def check_volume_profile(df: pd.DataFrame, pivots: List[Dict[str, Any]], p1_idx, p3_idx, p5_idx) -> bool:
    """(Para OCO) Compara volume da Cabeça vs Ombro Direito."""
    try:
        # Mapeia idx -> posição na lista de pivots
        indices = {p['idx']: i for i, p in enumerate(pivots)}
        idx_p1 = indices.get(p1_idx)
        idx_p3 = indices.get(p3_idx)
        idx_p5 = indices.get(p5_idx)

        if any(x is None for x in [idx_p1, idx_p3, idx_p5]):
            return False

        # Intervalos: OmbroEsq->Cabeça (vol_cabeca) vs Cabeça->OmbroDir (vol_od)
        # Nota: Lógica original compara o volume "perto" da cabeça e "perto" do OD
        p2_idx = pivots[idx_p3-1]['idx']
        p4_idx = pivots[idx_p5-1]['idx']

        vol_cabeca = df.loc[p2_idx:p3_idx]['volume'].mean()
        vol_od = df.loc[p4_idx:p5_idx]['volume'].mean()

        return vol_cabeca > vol_od
    except Exception:
        return False


def check_volume_profile_dtb(df: pd.DataFrame, p0, p1, p2, p3) -> bool:
    """(Para DT/DB/TT/TB) Verifica volume decrescente entre extremos."""
    try:
        idx0, idx1, idx2, idx3 = p0['idx'], p1['idx'], p2['idx'], p3['idx']
        if any(idx not in df.index for idx in [idx0, idx1, idx2, idx3]):
            return False

        # Intervalo da primeira perna vs segunda perna
        start1, end1 = (idx0, idx1) if idx0 <= idx1 else (idx1, idx0)
        start2, end2 = (idx2, idx3) if idx2 <= idx3 else (idx3, idx2)

        vol_extremo_1 = df.loc[start1:end1]['volume'].mean()
        vol_extremo_2 = df.loc[start2:end2]['volume'].mean()

        if np.isnan(vol_extremo_1) or np.isnan(vol_extremo_2):
            return False

        return vol_extremo_2 < vol_extremo_1
    except Exception:
        return False


def check_obv_divergence_dtb(df: pd.DataFrame, p1, p3, tipo_padrao: str) -> bool:
    """Verifica divergência de OBV entre dois extremos."""
    try:
        if 'OBV' not in df.columns:
            return False

        idx1, idx3 = p1['idx'], p3['idx']
        if idx1 not in df.index or idx3 not in df.index:
            return False

        obv1 = df.loc[idx1, 'OBV']
        obv3 = df.loc[idx3, 'OBV']

        if tipo_padrao in ('DT', 'TT'):  # Topos
            # Preço fez topo igual/perto, mas OBV caiu (pressão vendedora)
            return obv3 < obv1
        else:  # Fundos (DB/TB)
            # Preço fez fundo igual/perto, mas OBV subiu (pressão compradora)
            return obv3 > obv1
    except Exception:
        return False
