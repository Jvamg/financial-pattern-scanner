import sys
import os
import argparse
import logging
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"--- DEBUG: Raiz do projeto configurada em: {project_root} ---")

try:
    from config import Config
    from src.data_loader import buscar_dados
    from src.indicators import calcular_indicadores, calcular_zigzag_oficial
    from src.patterns import (
        identificar_padroes_hns,
        identificar_padroes_double_top_bottom,
        identificar_padroes_ttb,
        validate_and_score_triple_pattern
    )
except ImportError as e:
    print("\nERRO CRÍTICO:")
    print(f"Não foi possível importar um módulo: {e}")
    print("Verifique se 'config.py' está na raiz e se a pasta 'src' tem o arquivo '__init__.py'.")
    sys.exit(1)


def setup_logging():
    debug_dir = getattr(Config, 'DEBUG_DIR', 'logs')
    os.makedirs(debug_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(
                debug_dir, 'run.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Gerador de Dataset de Padrões")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Lista de tickers (ex: BTC-USD)")
    parser.add_argument("--strategies", type=str,
                        default=None, help="Estratégias ZigZag")
    parser.add_argument("--intervals", type=str,
                        default=None, help="Intervalos")
    parser.add_argument("--period", type=str, default=None, help="Período")
    parser.add_argument("--output", type=str,
                        default=None, help="Caminho saída")
    parser.add_argument("--patterns", type=str, default="ALL",
                        help="Tipos: HNS, DTB, TTB, ALL")
    return parser.parse_args()


def main():
    setup_logging()
    args = _parse_cli_args()

    if args.period:
        Config.DATA_PERIOD = args.period

    selected_tickers = [t.strip() for t in args.tickers.split(
        ",")] if args.tickers else Config.TICKERS
    intervals_filter = {i.strip() for i in args.intervals.split(
        ",")} if args.intervals else None

    strategies_dict = Config.ZIGZAG_STRATEGIES
    if args.strategies:
        wanted = {s.strip() for s in args.strategies.split(",")}
        strategies_dict = {k: v for k,
                           v in strategies_dict.items() if k in wanted}

    final_csv_path = args.output if args.output else Config.FINAL_CSV_PATH
    wanted_patterns = {s.strip().upper() for s in args.patterns.split(",")}

    logging.info("--- INICIANDO GERADOR (Yahoo Finance) ---")
    os.makedirs(os.path.dirname(final_csv_path)
                or Config.OUTPUT_DIR, exist_ok=True)

    todos_padroes = []

    for strategy_name, intervals_config in strategies_dict.items():
        logging.info(f"===== ESTRATÉGIA: {strategy_name.upper()} =====")
        for interval, params in intervals_config.items():
            if intervals_filter and interval not in intervals_filter:
                continue

            for ticker in selected_tickers:
                logging.info(
                    f"Processando: {ticker} | {interval} | {strategy_name}")
                try:
                    # Pipeline
                    df = buscar_dados(ticker, Config.DATA_PERIOD, interval)
                    if df.empty:
                        logging.warning(f"Sem dados para {ticker}")
                        continue

                    df = calcular_indicadores(df)
                    pivots = calcular_zigzag_oficial(
                        df, params['depth'], params['deviation'])

                    if len(pivots) < 4:
                        continue

                    padroes_run = []

                    # Detecção
                    if 'ALL' in wanted_patterns or 'HNS' in wanted_patterns:
                        padroes_run.extend(identificar_padroes_hns(pivots, df))
                    if 'ALL' in wanted_patterns or 'DTB' in wanted_patterns:
                        padroes_run.extend(
                            identificar_padroes_double_top_bottom(pivots, df))
                    if 'ALL' in wanted_patterns or 'TTB' in wanted_patterns:
                        cand_ttb = identificar_padroes_ttb(pivots)
                        for cand in cand_ttb:
                            res = validate_and_score_triple_pattern(cand, df)
                            if res:
                                padroes_run.append(res)

                    if padroes_run:
                        logging.info(
                            f" > {len(padroes_run)} padrões encontrados.")
                        for p in padroes_run:
                            p.update({'strategy': strategy_name,
                                     'timeframe': interval, 'ticker': ticker})
                            todos_padroes.append(p)

                except Exception as e:
                    logging.error(f"Erro em {ticker}: {e}")

    if not todos_padroes:
        logging.warning("Nenhum padrão encontrado em toda a execução.")
        return

    # Salvamento
    df_final = pd.DataFrame(todos_padroes)

    # Tratamento de datas e deduplicação
    for col in ['cabeca_idx', 'p3_idx', 'p5_idx']:
        if col in df_final.columns:
            # Garante formato datetime naive para evitar erros de timezone
            df_final[col] = pd.to_datetime(
                df_final[col], errors='coerce').dt.tz_localize(None)

    # Cria chave única
    df_final['chave_idx'] = df_final.get('p3_idx', pd.NaT)
    if 'padrao_tipo' in df_final.columns:
        mask_hns = df_final['padrao_tipo'].isin(['OCO', 'OCOI'])
        if 'cabeca_idx' in df_final.columns:
            df_final.loc[mask_hns,
                         'chave_idx'] = df_final.loc[mask_hns, 'cabeca_idx']

        mask_ttb = df_final['padrao_tipo'].isin(['TT', 'TB'])
        if 'p5_idx' in df_final.columns:
            df_final.loc[mask_ttb,
                         'chave_idx'] = df_final.loc[mask_ttb, 'p5_idx']

    df_final.drop_duplicates(
        subset=['ticker', 'timeframe', 'padrao_tipo', 'chave_idx'], inplace=True)
    if 'chave_idx' in df_final.columns:
        df_final.drop(columns=['chave_idx'], inplace=True)

    # Ordenação
    cols = list(df_final.columns)
    prioridade = ['ticker', 'timeframe',
                  'strategy', 'padrao_tipo', 'score_total']
    ordered = [c for c in prioridade if c in cols] + \
        [c for c in cols if c not in prioridade]

    df_final = df_final[ordered]
    df_final.to_csv(final_csv_path, index=False,
                    date_format='%Y-%m-%d %H:%M:%S')
    logging.info(f"SUCESSO! Dataset salvo em: {final_csv_path}")


if __name__ == "__main__":
    main()
