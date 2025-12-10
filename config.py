import os


class Config:
    """Configurações globais do projeto."""

    TICKERS = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOGE-USD',
        'XRP-USD', 'LTC-USD', 'LINK-USD', 'MATIC-USD', 'DOT-USD'
    ]

    DATA_PERIOD = '2y'  # Período padrão (funciona bem para 1h e 1d)

    ZIGZAG_STRATEGIES = {
        'swing_short': {
            '15m': {'depth': 8,  'deviation': 2.0},
            '1h':  {'depth': 10, 'deviation': 2.8},
            '4h':  {'depth': 12, 'deviation': 4.0}
        },
        'scalping': {
            '5m':  {'depth': 5, 'deviation': 0.55},
            '15m': {'depth': 6, 'deviation': 0.75}
        }
    }

    # --- PESOS E REGRAS ---
    SCORE_WEIGHTS_HNS = {
        'valid_extremo_cabeca': 20, 'valid_contexto_cabeca': 15,
        'valid_simetria_ombros': 10, 'valid_neckline_plana': 5,
        'valid_base_tendencia': 5, 'valid_neckline_retest_p6': 15,
        'valid_divergencia_rsi': 10, 'valid_divergencia_macd': 10,
        'valid_volume_breakout_neckline': 10
    }
    MINIMUM_SCORE_HNS = 70

    SCORE_WEIGHTS_DTB = {
        'valid_estrutura_picos_vales': 20, 'valid_simetria_extremos': 10,
        'valid_profundidade_vale_pico': 15, 'valid_contexto_extremos': 5,
        'valid_contexto_tendencia': 5, 'valid_neckline_retest_p4': 15,
        'valid_divergencia_rsi': 8, 'valid_volume_breakout_neckline': 10
    }
    MINIMUM_SCORE_DTB = 70

    SCORE_WEIGHTS_TTB = SCORE_WEIGHTS_DTB.copy()
    MINIMUM_SCORE_TTB = 70

    # --- INDICADORES ---
    RSI_LENGTH = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9

    STOCH_K = 14
    STOCH_D = 3
    STOCH_SMOOTH_K = 3

    # --- PARÂMETROS TÉCNICOS ---
    HEAD_SIGNIFICANCE_RATIO = 1.1
    SHOULDER_SYMMETRY_TOLERANCE = 0.30
    NECKLINE_FLATNESS_TOLERANCE = 0.25
    NECKLINE_RETEST_ATR_MULTIPLIER = 5.0
    DTB_SYMMETRY_TOLERANCE_FACTOR = 0.35
    DTB_VALLEY_PEAK_DEPTH_RATIO = 0.1
    DTB_TREND_MIN_DIFF_FACTOR = 0.01

    # Validação de Contexto (Novos)
    HEAD_EXTREME_LOOKBACK_FACTOR = 1
    HEAD_EXTREME_LOOKBACK_MIN_BARS = 8

    # Quantos padrões recentes analisar (EVITA O ERRO QUE DEU AGORA)
    RECENT_PATTERNS_LOOKBACK_COUNT = 3

    # --- SISTEMA ---
    MAX_DOWNLOAD_TENTATIVAS = 3
    RETRY_DELAY_SEGUNDOS = 2
    OUTPUT_DIR = 'data/datasets'
    FINAL_CSV_PATH = os.path.join(OUTPUT_DIR, 'dataset_patterns.csv')
    DEBUG_DIR = 'logs'

    # Configs opcionais de validação avançada
    STOCH_DIVERGENCE_REQUIRES_OBOS = False
    MACD_SIGNAL_CROSS_LOOKBACK_BARS = 7
    MACD_CROSS_MAX_AGE_BARS = 3
    VOLUME_BREAKOUT_LOOKBACK_BARS = 20
    VOLUME_BREAKOUT_MULTIPLIER = 1.8
    BREAKOUT_SEARCH_MAX_BARS = 60
    STOCH_CROSS_LOOKBACK_BARS = 7
