"""
Módulo de Detecção e Validação de Padrões.
Identifica sequências de pivôs (ZigZag) e valida regras de OCO, Topo Duplo e Triplo.
Operando no modo "Live" (Pivô Fantasma no candle atual).
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

        logging.debug(msg)
        debug_dir = getattr(Config, 'DEBUG_DIR', 'logs')
        os.makedirs(debug_dir, exist_ok=True)
        sanitized = re.sub(r'\x1b\[[0-9;]*m', '', msg)

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

    altura_cabeca = abs(
        cabeca['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
    altura_ombro_esq = abs(
        ombro_esq['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
    altura_ombro_dir = abs(
        ombro_dir['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))

    details['valid_extremo_cabeca'] = (tipo_padrao == 'OCO' and cabeca['preco'] > ombro_esq['preco'] and cabeca['preco'] > ombro_dir['preco']) or \
                                      (tipo_padrao == 'OCOI' and cabeca['preco'] <
                                       ombro_esq['preco'] and cabeca['preco'] < ombro_dir['preco'])
    if not details['valid_extremo_cabeca']:
        return None

    details['valid_contexto_cabeca'] = is_head_extreme(
        df_historico, cabeca, avg_pivot_dist_bars)
    if not details['valid_contexto_cabeca']:
        return None

    details['valid_simetria_ombros'] = altura_cabeca > 0 and abs(
        altura_ombro_esq - altura_ombro_dir) <= altura_cabeca * Config.SHOULDER_SYMMETRY_TOLERANCE
    if not details['valid_simetria_ombros']:
        return None

    altura_media_ombros = np.mean([altura_ombro_esq, altura_ombro_dir])
    details['valid_neckline_plana'] = altura_media_ombros > 0 and abs(
        neckline1['preco'] - neckline2['preco']) <= altura_media_ombros * Config.NECKLINE_FLATNESS_TOLERANCE
    if not details['valid_neckline_plana']:
        return None

    if tipo_padrao == 'OCO':
        details['valid_base_tendencia'] = (p0['preco'] < neckline1['preco']) and (
            p0['preco'] < neckline2['preco'])
    else:
        details['valid_base_tendencia'] = (p0['preco'] > neckline1['preco']) and (
            p0['preco'] > neckline2['preco'])
    if not details['valid_base_tendencia']:
        return None

    neckline_price = np.mean([neckline1['preco'], neckline2['preco']])
    atr_series = df_historico.get('ATR_14', pd.Series(dtype=float))
    atr_val = float(atr_series.loc[p6['idx']]) if p6['idx'] in atr_series.index and not np.isnan(
        atr_series.loc[p6['idx']]) else (float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else 0.0)

    max_variation = Config.NECKLINE_RETEST_ATR_MULTIPLIER * \
        atr_val if atr_val > 0 else float(neckline_price) * 0.005

    details['valid_neckline_retest_p6'] = abs(
        p6['preco'] - neckline_price) <= max_variation
    if not details['valid_neckline_retest_p6']:
        return None

    direction = 'bearish' if tipo_padrao == 'OCO' else 'bullish'
    rsi_src = df_historico['high'] if tipo_padrao == 'OCO' else df_historico['low']

    rsi_div, rsi_strong = assess_rsi_divergence_strength(
        df_historico, p1['idx'], p3['idx'], p1['preco'], p3['preco'], direction, rsi_src)
    if rsi_div:
        details['valid_divergencia_rsi'] = True
    if rsi_strong:
        details['valid_divergencia_rsi_strong'] = True

    if check_macd_divergence(df_historico, p1['idx'], p3['idx'], p1['preco'], p3['preco'], tipo_padrao):
        details['valid_divergencia_macd'] = True

    stoch_flags = check_stochastic_confirmation(
        df_historico, p1['idx'], p3['idx'], p1['preco'], p3['preco'], direction)
    details.update(stoch_flags)

    breakout_idx = find_breakout_index(df_historico, neckline_price, p5['idx'], direction, getattr(
        Config, 'BREAKOUT_SEARCH_MAX_BARS', 60))

    if breakout_idx is not None and detect_macd_signal_cross(df_historico, breakout_idx, direction):
        details['valid_macd_signal_cross'] = True
    elif detect_macd_signal_cross(df_historico, p6['idx'], direction):
        details['valid_macd_signal_cross'] = True

    if breakout_idx is not None and check_breakout_volume(df_historico, breakout_idx):
        details['valid_volume_breakout_neckline'] = True

    if altura_ombro_esq > 0 and (altura_cabeca / altura_ombro_esq >= Config.HEAD_SIGNIFICANCE_RATIO):
        details['valid_proeminencia_cabeca'] = True
    if check_volume_profile(df_historico, pivots, p1['idx'], p3['idx'], p5['idx']):
        details['valid_perfil_volume'] = True
    if (tipo_padrao == 'OCO' and ombro_dir['preco'] < ombro_esq['preco']) or (tipo_padrao == 'OCOI' and ombro_dir['preco'] > ombro_esq['preco']):
        details['valid_ombro_direito_fraco'] = True

    score = sum(Config.SCORE_WEIGHTS_HNS.get(rule, 0)
                for rule, passed in details.items() if passed)

    if score >= Config.MINIMUM_SCORE_HNS:
        base_data = {
            'padrao_tipo': tipo_padrao, 'score_total': score, 'p0_idx': p0['idx'],
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
    """Gera janelas base de 6 pivôs e cria o 7º (p6) no preço atual."""
    padroes_encontrados = []
    n = len(pivots)
    if n < 6:
        return []

    try:
        locs = pd.Series(range(len(df_historico)), index=df_historico.index)
        distancias = [locs[pivots[i]['idx']] - locs[pivots[i-1]['idx']]
                      for i in range(1, n) if pivots[i]['idx'] in locs]
        avg_dist = np.mean(distancias) if distancias else 0
    except:
        avg_dist = 0

    start_index = max(0, n - 5 - Config.RECENT_PATTERNS_LOOKBACK_COUNT)

    last_idx = df_historico.index[-1]
    last_row = df_historico.iloc[-1]

    for i in range(start_index, n - 5):
        janela_base = pivots[i:i+6]

        # Ignora se o último pivô confirmado do ZigZag já for o candle atual (para evitar duplicação fantasma)
        if janela_base[-1]['idx'] == last_idx:
            continue

        tipo_padrao = None
        p6_fantasma = None

        if all(p['tipo'] == t for p, t in zip(janela_base, ['VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO'])):
            tipo_padrao = 'OCO'
            p6_fantasma = {'idx': last_idx,
                           'preco': last_row['low'], 'tipo': 'VALE'}
        elif all(p['tipo'] == t for p, t in zip(janela_base, ['PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE'])):
            tipo_padrao = 'OCOI'
            p6_fantasma = {'idx': last_idx,
                           'preco': last_row['high'], 'tipo': 'PICO'}

        if tipo_padrao and p6_fantasma:
            janela_completa = janela_base + [p6_fantasma]
            dados = validate_and_score_hns_pattern(
                *janela_completa, tipo_padrao, df_historico, pivots, avg_dist)
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

    if tipo_padrao == 'DT':
        estrutura_ok = (p1['tipo'] == 'PICO' and p2['tipo']
                        == 'VALE' and p3['tipo'] == 'PICO')
        precos_ok = (preco_p1 > preco_p0) and (
            preco_p1 > preco_p2 and preco_p3 > preco_p2)
    else:
        estrutura_ok = (p1['tipo'] == 'VALE' and p2['tipo']
                        == 'PICO' and p3['tipo'] == 'VALE')
        precos_ok = (preco_p1 < preco_p0) and (
            preco_p1 < preco_p2 and preco_p3 < preco_p2)

    details['valid_estrutura_picos_vales'] = estrutura_ok and precos_ok
    if not details['valid_estrutura_picos_vales']:
        return None

    details['valid_contexto_extremos'] = is_head_extreme(
        df_historico, p1, avg_pivot_dist_bars)
    if not details['valid_contexto_extremos']:
        return None

    min_sep = Config.DTB_TREND_MIN_DIFF_FACTOR * \
        max(1.0, abs(preco_p1 - preco_p2))
    if tipo_padrao == 'DT':
        details['valid_contexto_tendencia'] = (preco_p2 >= preco_p0 - min_sep)
    else:
        details['valid_contexto_tendencia'] = (preco_p2 <= preco_p0 + min_sep)
    if not details['valid_contexto_tendencia']:
        return None

    altura = abs(preco_p1 - preco_p2)
    diff_picos = abs(preco_p1 - preco_p3)
    details['valid_simetria_extremos'] = diff_picos <= Config.DTB_SYMMETRY_TOLERANCE_FACTOR * altura
    if not details['valid_simetria_extremos']:
        return None

    perna_anterior = abs(preco_p1 - preco_p0)
    profundidade = abs(preco_p1 - preco_p2)
    details['valid_profundidade_vale_pico'] = perna_anterior > 0 and profundidade >= Config.DTB_VALLEY_PEAK_DEPTH_RATIO * perna_anterior
    if not details['valid_profundidade_vale_pico']:
        return None

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
    """Gera janelas base de 4 pivôs e cria o 5º (p4) no preço atual."""
    padroes_encontrados = []
    n = len(pivots)
    if n < 4:
        return []

    try:
        locs = pd.Series(range(len(df_historico)), index=df_historico.index)
        distancias = [locs[pivots[i]['idx']] - locs[pivots[i-1]['idx']]
                      for i in range(1, n) if pivots[i]['idx'] in locs]
        avg_dist = np.mean(distancias) if distancias else 0
    except:
        avg_dist = 0

    start_index = max(0, n - 3 - Config.RECENT_PATTERNS_LOOKBACK_COUNT)

    last_idx = df_historico.index[-1]
    last_row = df_historico.iloc[-1]

    for i in range(start_index, n - 3):
        janela_base = pivots[i:i+4]

        if janela_base[-1]['idx'] == last_idx:
            continue

        tipo = None
        p4_fantasma = None

        if all(p['tipo'] == t for p, t in zip(janela_base, ['VALE', 'PICO', 'VALE', 'PICO'])):
            tipo = 'DT'
            p4_fantasma = {'idx': last_idx,
                           'preco': last_row['low'], 'tipo': 'VALE'}
        elif all(p['tipo'] == t for p, t in zip(janela_base, ['PICO', 'VALE', 'PICO', 'VALE'])):
            tipo = 'DB'
            p4_fantasma = {'idx': last_idx,
                           'preco': last_row['high'], 'tipo': 'PICO'}

        if tipo and p4_fantasma:
            janela_completa = janela_base + [p4_fantasma]
            dados = validate_and_score_double_pattern(
                *janela_completa, tipo, df_historico, avg_dist)
            if dados:
                padroes_encontrados.append(dados)

    return padroes_encontrados

# ==============================================================================
# 3. TRIPLE TOP / BOTTOM (TT/TB) - REFATORADO E PADRONIZADO
# ==============================================================================


def validate_and_score_triple_pattern(p0, p1, p2, p3, p4, p5, p6, tipo_padrao, df_historico, avg_pivot_dist_bars):
    """Valida e pontua TT/TB."""
    if tipo_padrao not in ('TT', 'TB'):
        return None

    pivs = [p0, p1, p2, p3, p4, p5, p6]
    precos = [float(p['preco']) for p in pivs]

    details = {key: False for key in Config.SCORE_WEIGHTS_TTB.keys()}

    if tipo_padrao == 'TT':
        rel_ok = (precos[1] > precos[0] and precos[1] > precos[2] and precos[3] > precos[2]
                  and precos[3] > precos[4] and precos[5] > precos[4] and precos[5] > precos[6])
    else:
        rel_ok = (precos[1] < precos[0] and precos[1] < precos[2] and precos[3] < precos[2]
                  and precos[3] < precos[4] and precos[5] < precos[4] and precos[5] < precos[6])

    details['valid_estrutura_picos_vales'] = rel_ok
    if not details['valid_estrutura_picos_vales']:
        return None

    # O avg_dist agora vem via argumento, igual ao HNS e DTB
    details['valid_contexto_extremos'] = is_head_extreme_past_only(
        df_historico, p1, avg_pivot_dist_bars)

    if not details['valid_contexto_extremos']:
        return None

    try:
        tol = Config.DTB_SYMMETRY_TOLERANCE_FACTOR
        if tipo_padrao == 'TT':
            alturas = [abs(precos[1]-precos[2]), abs(precos[3] -
                                                     precos[4]), abs(precos[5]-precos[6])]
            peaks = [precos[1], precos[3], precos[5]]
        else:
            alturas = [abs(precos[2]-precos[1]), abs(precos[4] -
                                                     precos[3]), abs(precos[6]-precos[5])]
            peaks = [precos[1], precos[3], precos[5]]

        avg_h = np.mean(alturas)
        details['valid_simetria_extremos'] = avg_h > 0 and (
            max(peaks) - min(peaks)) <= tol * avg_h
    except:
        return None
    if not details['valid_simetria_extremos']:
        return None

    neckline = np.mean([precos[2], precos[4]])
    atr = df_historico.get('ATR_14', pd.Series(dtype=float))
    atr_val = float(atr.loc[p6['idx']]) if p6['idx'] in atr.index else 0.0
    tol = (Config.NECKLINE_RETEST_ATR_MULTIPLIER *
           atr_val) if atr_val > 0 else float(neckline) * 0.010

    details['valid_neckline_retest_p6'] = abs(precos[6] - neckline) <= tol
    if not details['valid_neckline_retest_p6']:
        return None

    direction = 'bearish' if tipo_padrao == 'TT' else 'bullish'
    rsi_div, rsi_strong = assess_rsi_divergence_strength(
        df_historico, p1['idx'], p5['idx'], precos[1], precos[5], direction, df_historico['close'])
    if rsi_div:
        details['valid_divergencia_rsi'] = True
    if rsi_strong:
        details['valid_divergencia_rsi_strong'] = True

    if check_macd_divergence(df_historico, p1['idx'], p5['idx'], precos[1], precos[5], tipo_padrao):
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
            'padrao_tipo': tipo_padrao, 'score_total': score,
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


def identificar_padroes_ttb(pivots: List[Dict[str, Any]], df_historico: pd.DataFrame) -> List[Dict[str, Any]]:
    """Gera janelas base de 6 pivôs e cria o 7º (p6) no preço atual."""
    padroes_encontrados = []
    n = len(pivots)
    if n < 6:
        return padroes_encontrados

    try:
        locs = pd.Series(range(len(df_historico)), index=df_historico.index)
        distancias = [locs[pivots[i]['idx']] - locs[pivots[i-1]['idx']]
                      for i in range(1, n) if pivots[i]['idx'] in locs]
        avg_dist = np.mean(distancias) if distancias else 0
    except:
        avg_dist = 0

    start_index = max(0, n - 5 - Config.RECENT_PATTERNS_LOOKBACK_COUNT)
    last_idx = df_historico.index[-1]
    last_row = df_historico.iloc[-1]

    for i in range(start_index, n - 5):
        janela_base = pivots[i:i+6]

        if janela_base[-1]['idx'] == last_idx:
            continue

        tipo_padrao = None
        p6_fantasma = None
        tipos = [p['tipo'] for p in janela_base]

        if tipos == ['VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO']:
            tipo_padrao = 'TT'
            p6_fantasma = {'idx': last_idx,
                           'preco': last_row['low'], 'tipo': 'VALE'}
        elif tipos == ['PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE']:
            tipo_padrao = 'TB'
            p6_fantasma = {'idx': last_idx,
                           'preco': last_row['high'], 'tipo': 'PICO'}

        if tipo_padrao and p6_fantasma:
            janela_completa = janela_base + [p6_fantasma]

            # Agora chama a validação internamente, igual DTB e HNS
            dados = validate_and_score_triple_pattern(
                *janela_completa, tipo_padrao, df_historico, avg_dist)

            if dados:
                padroes_encontrados.append(dados)

    return padroes_encontrados
