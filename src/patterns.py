"""
Módulo de Detecção e Validação de Padrões.
Identifica sequências de pivôs (ZigZag) e valida regras de OCO, Topo Duplo e Triplo.
"""
import pandas as pd
import numpy as np
import logging
import os
import re
from typing import List, Dict, Any, Optional
from colorama import Fore, Style, init

# Inicializa colorama para logs coloridos
init(autoreset=True)

# Importações internas
try:
    from config import Config
    from src.indicators import (
        is_head_extreme,
        is_head_extreme_past_only,
        assess_rsi_divergence_strength,
        check_macd_divergence,
        check_stochastic_confirmation,
        find_breakout_index,
        detect_macd_signal_cross,
        check_breakout_volume,
        check_volume_profile,
        check_volume_profile_dtb,
        check_obv_divergence_dtb
    )
except ImportError:
    import sys
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))
    from config import Config
    from src.indicators import *

# ==============================================================================
# 0. DEBUGGING HELPER
# ==============================================================================


def _pattern_debug(pattern_type: str, msg: str) -> None:
    """Escreve logs detalhados de validação em arquivos separados por padrão."""
    try:
        group = None
        enabled = False
        if pattern_type in ('OCO', 'OCOI'):
            group = 'HNS'
            enabled = bool(getattr(Config, 'HNS_DEBUG', False))
        elif pattern_type in ('DT', 'DB'):
            group = 'DTB'
            enabled = bool(getattr(Config, 'DTB_DEBUG', False))
        elif pattern_type in ('TT', 'TB'):
            group = 'TTB'
            enabled = bool(getattr(Config, 'TTB_DEBUG', False))
        else:
            enabled = False

        if not enabled:
            return

        # Log no console (nível debug)
        logging.debug(msg)

        # Log em arquivo
        debug_dir = getattr(Config, 'DEBUG_DIR', 'logs')
        os.makedirs(debug_dir, exist_ok=True)
        sanitized = re.sub(r'\x1b\[[0-9;]*m', '', msg)  # Remove cores ANSI

        if group == 'DTB':
            filepath = getattr(Config, 'DTB_DEBUG_FILE',
                               os.path.join(debug_dir, 'dtb_debug.log'))
        elif group == 'HNS':
            filepath = os.path.join(debug_dir, 'hns_debug.log')
        elif group == 'TTB':
            filepath = os.path.join(debug_dir, 'ttb_debug.log')
        else:
            filepath = os.path.join(debug_dir, 'patterns_debug.log')

        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(f"{sanitized}\n")
    except Exception:
        pass


# ==============================================================================
# 1. HEAD & SHOULDERS (OCO / OCOI)
# ==============================================================================

def validate_and_score_hns_pattern(p0, p1, p2, p3, p4, p5, p6, tipo_padrao, df_historico, pivots, avg_pivot_dist_bars):
    """Valida e pontua OCO/OCOI aplicando regras eliminatórias e confirmatórias."""
    details = {key: False for key in Config.SCORE_WEIGHTS_HNS.keys()}
    ombro_esq, neckline1, cabeca, neckline2, ombro_dir = p1, p2, p3, p4, p5

    # --- 1. Estrutura de Preço (Hard Rules) ---
    altura_cabeca = abs(
        cabeca['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
    altura_ombro_esq = abs(
        ombro_esq['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
    altura_ombro_dir = abs(
        ombro_dir['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))

    # Cabeça deve ser o extremo
    details['valid_extremo_cabeca'] = (tipo_padrao == 'OCO' and cabeca['preco'] > ombro_esq['preco'] and cabeca['preco'] > ombro_dir['preco']) or \
                                      (tipo_padrao == 'OCOI' and cabeca['preco'] <
                                       ombro_esq['preco'] and cabeca['preco'] < ombro_dir['preco'])
    if not details['valid_extremo_cabeca']:
        return None

    # Contexto local (cabeça é extremo na janela)
    details['valid_contexto_cabeca'] = is_head_extreme(
        df_historico, cabeca, avg_pivot_dist_bars)
    if not details['valid_contexto_cabeca']:
        return None

    # Simetria Ombros
    details['valid_simetria_ombros'] = altura_cabeca > 0 and \
        abs(altura_ombro_esq - altura_ombro_dir) <= altura_cabeca * \
        Config.SHOULDER_SYMMETRY_TOLERANCE
    if not details['valid_simetria_ombros']:
        return None

    # Neckline Plana
    altura_media_ombros = np.mean([altura_ombro_esq, altura_ombro_dir])
    details['valid_neckline_plana'] = altura_media_ombros > 0 and \
        abs(neckline1['preco'] - neckline2['preco']
            ) <= altura_media_ombros * Config.NECKLINE_FLATNESS_TOLERANCE
    if not details['valid_neckline_plana']:
        return None

    # Tendência Base
    if tipo_padrao == 'OCO':
        details['valid_base_tendencia'] = (p0['preco'] < neckline1['preco']) and (
            p0['preco'] < neckline2['preco'])
    else:  # 'OCOI'
        details['valid_base_tendencia'] = (p0['preco'] > neckline1['preco']) and (
            p0['preco'] > neckline2['preco'])
    if not details['valid_base_tendencia']:
        return None

    # Reteste da Neckline (p6)
    neckline_price = np.mean([neckline1['preco'], neckline2['preco']])

    # Busca ATR para tolerância
    atr_series = df_historico.get('ATR_14', pd.Series(dtype=float))
    if p6['idx'] in atr_series.index and not np.isnan(atr_series.loc[p6['idx']]):
        atr_val = float(atr_series.loc[p6['idx']])
    else:
        atr_val = float(
            atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else 0.0

    if atr_val > 0:
        max_variation = Config.NECKLINE_RETEST_ATR_MULTIPLIER * atr_val
    else:
        max_variation = float(neckline_price) * 0.005

    details['valid_neckline_retest_p6'] = abs(
        p6['preco'] - neckline_price) <= max_variation
    if not details['valid_neckline_retest_p6']:
        return None

    # --- 2. Indicadores e Confirmações (Soft Rules) ---
    direction = 'bearish' if tipo_padrao == 'OCO' else 'bullish'
    rsi_src = df_historico['high'] if tipo_padrao == 'OCO' else df_historico['low']

    # RSI
    rsi_div, rsi_strong = assess_rsi_divergence_strength(
        df_historico, p1['idx'], p3['idx'], p1['preco'], p3['preco'], direction, rsi_src)
    if rsi_div:
        details['valid_divergencia_rsi'] = True
    if rsi_strong:
        details['valid_divergencia_rsi_strong'] = True

    # MACD
    if check_macd_divergence(df_historico, p1['idx'], p3['idx'], p1['preco'], p3['preco'], tipo_padrao):
        details['valid_divergencia_macd'] = True

    # Estocástico
    stoch_flags = check_stochastic_confirmation(
        df_historico, p1['idx'], p3['idx'], p1['preco'], p3['preco'], direction)
    details.update(stoch_flags)

    # Breakout e Volume
    breakout_idx = find_breakout_index(df_historico, neckline_price, p5['idx'], direction, getattr(
        Config, 'BREAKOUT_SEARCH_MAX_BARS', 60))

    if breakout_idx is not None and detect_macd_signal_cross(df_historico, breakout_idx, direction):
        details['valid_macd_signal_cross'] = True
    elif detect_macd_signal_cross(df_historico, p6['idx'], direction):
        details['valid_macd_signal_cross'] = True

    if breakout_idx is not None and check_breakout_volume(df_historico, breakout_idx):
        details['valid_volume_breakout_neckline'] = True

    # Outras confirmações
    if altura_ombro_esq > 0 and (altura_cabeca / altura_ombro_esq >= Config.HEAD_SIGNIFICANCE_RATIO):
        details['valid_proeminencia_cabeca'] = True
    if check_volume_profile(df_historico, pivots, p1['idx'], p3['idx'], p5['idx']):
        details['valid_perfil_volume'] = True
    if (tipo_padrao == 'OCO' and ombro_dir['preco'] < ombro_esq['preco']) or (tipo_padrao == 'OCOI' and ombro_dir['preco'] > ombro_esq['preco']):
        details['valid_ombro_direito_fraco'] = True

    # Pontuação Final
    score = sum(Config.SCORE_WEIGHTS_HNS.get(rule, 0)
                for rule, passed in details.items() if passed)

    if score >= Config.MINIMUM_SCORE_HNS:
        base_data = {
            'padrao_tipo': tipo_padrao, 'score_total': score,
            'p0_idx': p0['idx'],
            'ombro1_idx': p1['idx'], 'ombro1_preco': p1['preco'],
            'neckline1_idx': p2['idx'], 'neckline1_preco': p2['preco'],
            'cabeca_idx': p3['idx'], 'cabeca_preco': p3['preco'],
            'neckline2_idx': p4['idx'], 'neckline2_preco': p4['preco'],
            'ombro2_idx': p5['idx'], 'ombro2_preco': p5['preco'],
            'retest_p6_idx': p6['idx'], 'retest_p6_preco': p6['preco']
        }
        base_data.update(details)
        return base_data

    return None


def identificar_padroes_hns(pivots: List[Dict[str, Any]], df_historico: pd.DataFrame) -> List[Dict[str, Any]]:
    """Gera janelas de 7 pivôs e identifica OCO/OCOI."""
    padroes_encontrados = []
    n = len(pivots)
    if n < 7:
        return []

    # Calcula distância média para validação de contexto
    try:
        locs = pd.Series(range(len(df_historico)), index=df_historico.index)
        distancias = [locs[pivots[i]['idx']] - locs[pivots[i-1]['idx']]
                      for i in range(1, n) if pivots[i]['idx'] in locs]
        avg_dist = np.mean(distancias) if distancias else 0
    except:
        avg_dist = 0

    start_index = max(0, n - 6 - Config.RECENT_PATTERNS_LOOKBACK_COUNT)

    for i in range(start_index, n - 6):
        janela = pivots[i:i+7]
        tipo_padrao = None

        # OCO: V-P-V-P-V-P-V (Pico no meio é a cabeça)
        if all(p['tipo'] == t for p, t in zip(janela, ['VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE'])):
            tipo_padrao = 'OCO'
        elif all(p['tipo'] == t for p, t in zip(janela, ['PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO'])):
            tipo_padrao = 'OCOI'

        if tipo_padrao:
            dados = validate_and_score_hns_pattern(
                *janela, tipo_padrao, df_historico, pivots, avg_dist)
            if dados:
                padroes_encontrados.append(dados)

    return padroes_encontrados


# ==============================================================================
# 2. DOUBLE TOP / BOTTOM (DT/DB)
# ==============================================================================

def validate_and_score_double_pattern(p0, p1, p2, p3, p4, tipo_padrao, df_historico, avg_pivot_dist_bars: int):
    """Valida e pontua DT/DB."""
    if tipo_padrao not in ('DT', 'DB'):
        return None
    details = {key: False for key in Config.SCORE_WEIGHTS_DTB.keys()}

    preco_p0, preco_p1 = float(p0['preco']), float(p1['preco'])
    preco_p2, preco_p3 = float(p2['preco']), float(p3['preco'])

    # Estrutura básica
    if tipo_padrao == 'DT':
        estrutura_ok = (p1['tipo'] == 'PICO' and p2['tipo']
                        == 'VALE' and p3['tipo'] == 'PICO')
        precos_ok = (preco_p1 > preco_p0) and (
            preco_p1 > preco_p2 and preco_p3 > preco_p2)
    else:  # DB
        estrutura_ok = (p1['tipo'] == 'VALE' and p2['tipo']
                        == 'PICO' and p3['tipo'] == 'VALE')
        precos_ok = (preco_p1 < preco_p0) and (
            preco_p1 < preco_p2 and preco_p3 < preco_p2)

    details['valid_estrutura_picos_vales'] = estrutura_ok and precos_ok
    if not details['valid_estrutura_picos_vales']:
        return None

    # Contexto (p1 deve ser extremo)
    details['valid_contexto_extremos'] = is_head_extreme(
        df_historico, p1, avg_pivot_dist_bars)
    if not details['valid_contexto_extremos']:
        return None

    # Tendência (HL para DT, LH para DB)
    min_sep = Config.DTB_TREND_MIN_DIFF_FACTOR * \
        max(1.0, abs(preco_p1 - preco_p2))
    if tipo_padrao == 'DT':
        details['valid_contexto_tendencia'] = (preco_p2 >= preco_p0 - min_sep)
    else:
        details['valid_contexto_tendencia'] = (preco_p2 <= preco_p0 + min_sep)
    if not details['valid_contexto_tendencia']:
        return None

    # Simetria
    altura = abs(preco_p1 - preco_p2)
    diff_picos = abs(preco_p1 - preco_p3)
    details['valid_simetria_extremos'] = diff_picos <= Config.DTB_SYMMETRY_TOLERANCE_FACTOR * altura
    if not details['valid_simetria_extremos']:
        return None

    # Profundidade
    perna_anterior = abs(preco_p1 - preco_p0)
    profundidade = abs(preco_p1 - preco_p2)
    details['valid_profundidade_vale_pico'] = perna_anterior > 0 and profundidade >= Config.DTB_VALLEY_PEAK_DEPTH_RATIO * perna_anterior
    if not details['valid_profundidade_vale_pico']:
        return None

    # Reteste (p4)
    neckline = preco_p2
    atr_series = df_historico.get('ATR_14', pd.Series(dtype=float))
    atr_val = float(atr_series.loc[p4['idx']]
                    ) if p4['idx'] in atr_series.index else 0.0
    tol = (Config.NECKLINE_RETEST_ATR_MULTIPLIER *
           atr_val) if atr_val > 0 else float(neckline) * 0.005

    details['valid_neckline_retest_p4'] = abs(
        float(p4['preco']) - neckline) <= tol
    if not details['valid_neckline_retest_p4']:
        return None

    # Indicadores
    details['valid_perfil_volume_decrescente'] = check_volume_profile_dtb(
        df_historico, p0, p1, p2, p3)
    details['valid_divergencia_obv'] = check_obv_divergence_dtb(
        df_historico, p1, p3, tipo_padrao)

    direction = 'bearish' if tipo_padrao == 'DT' else 'bullish'
    rsi_div, rsi_strong = assess_rsi_divergence_strength(
        df_historico, p1['idx'], p3['idx'], preco_p1, preco_p3, direction, df_historico['close'])
    if rsi_div:
        details['valid_divergencia_rsi'] = True
    if rsi_strong:
        details['valid_divergencia_rsi_strong'] = True

    if check_macd_divergence(df_historico, p1['idx'], p3['idx'], preco_p1, preco_p3, tipo_padrao):
        details['valid_divergencia_macd'] = True

    stoch_flags = check_stochastic_confirmation(
        df_historico, p1['idx'], p3['idx'], preco_p1, preco_p3, direction)
    details.update(stoch_flags)

    breakout_idx = find_breakout_index(df_historico, neckline, p3['idx'], direction, getattr(
        Config, 'BREAKOUT_SEARCH_MAX_BARS', 60))
    if breakout_idx:
        if detect_macd_signal_cross(df_historico, breakout_idx, direction):
            details['valid_macd_signal_cross'] = True
        if check_breakout_volume(df_historico, breakout_idx):
            details['valid_volume_breakout_neckline'] = True
    elif detect_macd_signal_cross(df_historico, p4['idx'], direction):
        details['valid_macd_signal_cross'] = True

    # 2º Topo menor (DT) ou fundo maior (DB)
    if tipo_padrao == 'DT' and preco_p3 < preco_p1:
        details['valid_extremo_secundario_fraco'] = True
    elif tipo_padrao == 'DB' and preco_p3 > preco_p1:
        details['valid_extremo_secundario_fraco'] = True

    score = sum(Config.SCORE_WEIGHTS_DTB.get(rule, 0)
                for rule, passed in details.items() if passed)

    if score >= Config.MINIMUM_SCORE_DTB:
        return {
            'padrao_tipo': tipo_padrao, 'score_total': score,
            'p0_idx': p0['idx'], 'p0_preco': preco_p0,
            'p1_idx': p1['idx'], 'p1_preco': preco_p1,
            'p2_idx': p2['idx'], 'p2_preco': preco_p2,
            'p3_idx': p3['idx'], 'p3_preco': preco_p3,
            'p4_idx': p4['idx'], 'p4_preco': float(p4['preco']),
            **details
        }
    return None


def identificar_padroes_double_top_bottom(pivots: List[Dict[str, Any]], df_historico: pd.DataFrame) -> List[Dict[str, Any]]:
    """Gera janelas de 5 pivôs e identifica DT/DB."""
    padroes_encontrados = []
    n = len(pivots)
    if n < 5:
        return []

    try:
        locs = pd.Series(range(len(df_historico)), index=df_historico.index)
        distancias = [locs[pivots[i]['idx']] - locs[pivots[i-1]['idx']]
                      for i in range(1, n) if pivots[i]['idx'] in locs]
        avg_dist = np.mean(distancias) if distancias else 0
    except:
        avg_dist = 0

    start_index = max(0, n - 4 - Config.RECENT_PATTERNS_LOOKBACK_COUNT)

    for i in range(start_index, n - 4):
        janela = pivots[i:i+5]
        tipo = None
        if all(p['tipo'] == t for p, t in zip(janela, ['VALE', 'PICO', 'VALE', 'PICO', 'VALE'])):
            tipo = 'DT'
        elif all(p['tipo'] == t for p, t in zip(janela, ['PICO', 'VALE', 'PICO', 'VALE', 'PICO'])):
            tipo = 'DB'

        if tipo:
            dados = validate_and_score_double_pattern(
                *janela, tipo, df_historico, avg_dist)
            if dados:
                padroes_encontrados.append(dados)

    return padroes_encontrados


# ==============================================================================
# 3. TRIPLE TOP / BOTTOM (TT/TB)
# ==============================================================================

def identificar_padroes_ttb(pivots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identifica sequências de 7 pivôs candidatos a TT/TB."""
    resultados = []
    n = len(pivots)
    if n < 7:
        return resultados

    start_index = max(0, n - 6 - Config.RECENT_PATTERNS_LOOKBACK_COUNT)

    for i in range(start_index, n - 6):
        janela = pivots[i:i+7]
        tipos = [p['tipo'] for p in janela]
        if tipos == ['VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE']:
            resultados.append(
                {'padrao_tipo': 'TT', **{f'p{k}_obj': janela[k] for k in range(7)}})
        elif tipos == ['PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO']:
            resultados.append(
                {'padrao_tipo': 'TB', **{f'p{k}_obj': janela[k] for k in range(7)}})
    return resultados


def validate_and_score_triple_pattern(pattern: Dict[str, Any], df_historico: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Valida e pontua TT/TB."""
    tipo = pattern.get('padrao_tipo')
    if tipo not in ('TT', 'TB'):
        return None

    pivs = [pattern.get(f'p{k}_obj') for k in range(7)]
    if any(p is None for p in pivs):
        return None
    p0, p1, p2, p3, p4, p5, p6 = pivs
    precos = [float(p['preco']) for p in pivs]

    details = {key: False for key in Config.SCORE_WEIGHTS_TTB.keys()}

    # Estrutura
    if tipo == 'TT':
        rel_ok = (precos[1] > precos[0] and precos[1] > precos[2] and precos[3] > precos[2]
                  and precos[3] > precos[4] and precos[5] > precos[4] and precos[5] > precos[6])
    else:
        rel_ok = (precos[1] < precos[0] and precos[1] < precos[2] and precos[3] < precos[2]
                  and precos[3] < precos[4] and precos[5] < precos[4] and precos[5] < precos[6])

    details['valid_estrutura_picos_vales'] = rel_ok
    if not details['valid_estrutura_picos_vales']:
        return None

    # Contexto Extremo
    try:
        locs = pd.Series(range(len(df_historico)), index=df_historico.index)
        dist = [locs[pivs[i]['idx']] - locs[pivs[i-1]['idx']]
                for i in range(1, 7)]
        avg_dist = np.mean(dist) if dist else 0
        details['valid_contexto_extremos'] = is_head_extreme_past_only(
            df_historico, p1, avg_dist)
    except:
        details['valid_contexto_extremos'] = False
    if not details['valid_contexto_extremos']:
        return None

    # Simetria Extremos
    try:
        tol = Config.DTB_SYMMETRY_TOLERANCE_FACTOR
        if tipo == 'TT':
            alturas = [abs(precos[1]-precos[2]), abs(precos[3] -
                                                     precos[4]), abs(precos[5]-precos[6])]
            peaks = [precos[1], precos[3], precos[5]]
        else:
            alturas = [abs(precos[2]-precos[1]), abs(precos[4] -
                                                     precos[3]), abs(precos[6]-precos[5])]
            peaks = [precos[1], precos[3], precos[5]]  # Vales

        avg_h = np.mean(alturas)
        details['valid_simetria_extremos'] = avg_h > 0 and (
            max(peaks) - min(peaks)) <= tol * avg_h
    except:
        return None
    if not details['valid_simetria_extremos']:
        return None

    # Reteste Neckline
    neckline = np.mean([precos[2], precos[4]])
    atr = df_historico.get('ATR_14', pd.Series(dtype=float))
    atr_val = float(atr.loc[p6['idx']]) if p6['idx'] in atr.index else 0.0
    tol = (Config.NECKLINE_RETEST_ATR_MULTIPLIER *
           atr_val) if atr_val > 0 else float(neckline) * 0.010

    details['valid_neckline_retest_p6'] = abs(precos[6] - neckline) <= tol
    if not details['valid_neckline_retest_p6']:
        return None

    # Confirmações
    direction = 'bearish' if tipo == 'TT' else 'bullish'
    rsi_div, rsi_strong = assess_rsi_divergence_strength(
        df_historico, p1['idx'], p5['idx'], precos[1], precos[5], direction, df_historico['close'])
    if rsi_div:
        details['valid_divergencia_rsi'] = True
    if rsi_strong:
        details['valid_divergencia_rsi_strong'] = True

    if check_macd_divergence(df_historico, p1['idx'], p5['idx'], precos[1], precos[5], tipo):
        details['valid_divergencia_macd'] = True

    stoch = check_stochastic_confirmation(
        df_historico, p1['idx'], p5['idx'], precos[1], precos[5], direction)
    details.update(stoch)

    breakout = find_breakout_index(df_historico, neckline, p6['idx'], direction, getattr(
        Config, 'BREAKOUT_SEARCH_MAX_BARS', 60))
    if breakout:
        if check_breakout_volume(df_historico, breakout):
            details['valid_volume_breakout_neckline'] = True
        if detect_macd_signal_cross(df_historico, breakout, direction):
            details['valid_macd_signal_cross'] = True
    elif detect_macd_signal_cross(df_historico, p6['idx'], direction):
        details['valid_macd_signal_cross'] = True

    score = sum(Config.SCORE_WEIGHTS_TTB.get(rule, 0)
                for rule, passed in details.items() if passed)

    if score >= Config.MINIMUM_SCORE_TTB:
        base = {
            'padrao_tipo': tipo, 'score_total': score,
            'p0_idx': p0['idx'], 'p0_preco': precos[0],
            'p1_idx': p1['idx'], 'p1_preco': precos[1],
            'p2_idx': p2['idx'], 'p2_preco': precos[2],
            'p3_idx': p3['idx'], 'p3_preco': precos[3],
            'p4_idx': p4['idx'], 'p4_preco': precos[4],
            'p5_idx': p5['idx'], 'p5_preco': precos[5],
            'p6_idx': p6['idx'], 'p6_preco': precos[6],
        }
        base.update(details)
        return base
    return None
