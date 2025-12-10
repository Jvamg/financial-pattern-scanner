"""
Ferramenta Gráfica de Rotulagem (Labeler).
Permite visualizar padrões, aplicar zoom e classificar como Válido (1) ou Inválido (0).
Uso: python tools/labeler.py
"""
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas_ta as ta
import mplfinance as mpf
import numpy as np
import pandas as pd
from tkinter import messagebox
import tkinter as tk
import sys
import os

# ==============================================================================
# 1. CORREÇÃO DE PATH (Executa IMEDIATAMENTE)
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"--- DEBUG: Raiz do projeto configurada em: {project_root} ---")
# ==============================================================================


try:
    from config import Config
    from src.data_loader import buscar_dados_range, safe_to_naive_datetime
    from src.indicators import calcular_zigzag_oficial
except ImportError as e:
    print(f"\nERRO CRÍTICO: {e}")
    sys.exit(1)


class LabelingTool(tk.Tk):
    def __init__(self, arquivo_entrada: str, arquivo_saida: str):
        super().__init__()
        self.title("Ferramenta de Anotação Profissional (v2.1 - Final)")
        self.geometry("1350x950")

        self.arquivo_saida = arquivo_saida
        self.df_trabalho: Optional[pd.DataFrame] = None
        self.indice_atual: int = 0
        self.fig: Optional[plt.Figure] = None
        self.canvas_widget = None

        # Mapeamento para exibir regras na tela
        self.regras_map = {
            'valid_divergencia_rsi': 'Divergência RSI',
            'valid_divergencia_macd': 'Divergência MACD',
            'valid_neckline_retest_p6': 'Reteste Neckline (OCO)',
            'valid_neckline_retest_p4': 'Reteste Neckline (DT/DB)',
            'valid_volume_breakout_neckline': 'Volume Breakout',
            'valid_simetria_extremos': 'Simetria',
            'valid_profundidade_vale_pico': 'Profundidade',
        }
        self.max_rule_slots = 10

        if not self.setup_dataframe(arquivo_entrada, arquivo_saida):
            self.destroy()
            return

        self._setup_ui()
        self.bind('<Key>', self.on_key_press)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Inicia
        self.carregar_proximo_padrao()

    def setup_dataframe(self, arquivo_entrada: str, arquivo_saida: str) -> bool:
        try:
            # Prioriza arquivo de saída (para continuar de onde parou)
            if os.path.exists(arquivo_saida):
                print(f"Carregando progresso de: {arquivo_saida}")
                df = pd.read_csv(arquivo_saida)
            elif os.path.exists(arquivo_entrada):
                print(f"Iniciando novo trabalho com: {arquivo_entrada}")
                df = pd.read_csv(arquivo_entrada)
            else:
                messagebox.showerror(
                    "Erro", f"Arquivo não encontrado:\n{arquivo_entrada}")
                return False

            # Normalização básica
            rename_map = {'timeframe': 'intervalo',
                          'strategy': 'estrategia_zigzag'}
            df.rename(columns=rename_map, inplace=True)

            if 'padrao_tipo' in df.columns:
                df['tipo_padrao'] = df['padrao_tipo']

            # Lógica para determinar datas de visualização (Inicio -> Fim)
            is_hns = df['tipo_padrao'].isin(['OCO', 'OCOI'])

            def get_col(name):
                return df[name] if name in df.columns else pd.NaT

            # Início do padrão
            df['data_inicio'] = np.where(
                is_hns, get_col('ombro1_idx'), get_col('p0_idx'))

            # Fim do padrão (Tenta reteste, senão usa o último ponto estrutural)
            retest_col = np.where(is_hns, get_col(
                'retest_p6_idx'), get_col('p4_idx'))
            end_std_col = np.where(is_hns, get_col(
                'ombro2_idx'), get_col('p3_idx'))

            # Se reteste for NaT, usa o fim padrão
            df['data_fim'] = np.where(
                pd.notna(retest_col), retest_col, end_std_col)

            # Ponto de destaque (Cabeça ou P1)
            df['data_destaque'] = np.where(
                is_hns, get_col('cabeca_idx'), get_col('p1_idx'))

            # Conversão segura de datas
            for c in ['data_inicio', 'data_fim', 'data_destaque']:
                df[c] = safe_to_naive_datetime(df[c])

            if 'label_humano' not in df.columns:
                df['label_humano'] = np.nan

            self.df_trabalho = df
            return True

        except Exception as e:
            messagebox.showerror("Erro Setup", f"Falha ao preparar dados: {e}")
            return False

    def _setup_ui(self):
        m = tk.PanedWindow(self, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        m.pack(fill=tk.BOTH, expand=True)

        self.frame_grafico = tk.Frame(m, bg='white')
        m.add(self.frame_grafico, minsize=500)

        frame_controles = tk.Frame(m, height=180)
        frame_controles.pack_propagate(False)
        m.add(frame_controles, minsize=180)

        frame_controles.grid_columnconfigure(0, weight=1)
        frame_controles.grid_columnconfigure(1, weight=1)

        # Info Texto
        frame_info = tk.Frame(frame_controles)
        frame_info.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)

        self.info_label = tk.Label(frame_info, text="Inicializando...", font=(
            "Segoe UI", 11), justify=tk.LEFT, anchor="nw")
        self.info_label.pack(side=tk.TOP, anchor="w")

        lbl_instrucoes = tk.Label(frame_info, text="\nComandos:\n[A] Aprovar (1)  |  [R] Rejeitar (0)  |  [Q] Sair",
                                  font=("Segoe UI", 10, "bold"), fg="#333")
        lbl_instrucoes.pack(side=tk.BOTTOM, anchor="w")

        # Boletim
        frame_boletim = tk.LabelFrame(
            frame_controles, text=" Scorecard ", font=("Segoe UI", 9, "bold"))
        frame_boletim.grid(row=0, column=1, sticky="nsew", padx=10, pady=5)

        self.boletim_labels = []
        for i in range(self.max_rule_slots):
            r = i % 5
            c = (i // 5) * 2
            lbl_name = tk.Label(frame_boletim, text="", font=(
                "Consolas", 9), width=22, anchor="w")
            lbl_name.grid(row=r, column=c, sticky="w")
            lbl_val = tk.Label(frame_boletim, text="", font=(
                "Consolas", 9, "bold"), width=5, anchor="w")
            lbl_val.grid(row=r, column=c+1, sticky="w")
            self.boletim_labels.append((lbl_name, lbl_val))

        self.score_final_label = tk.Label(
            frame_boletim, text="SCORE: --", font=("Segoe UI", 12, "bold"), fg="blue")
        self.score_final_label.grid(row=5, column=0, columnspan=4, pady=5)

    def carregar_proximo_padrao(self):
        if self.df_trabalho is None:
            return

        pendentes = self.df_trabalho[self.df_trabalho['label_humano'].isna(
        )].index
        if pendentes.empty:
            if messagebox.askyesno("Concluído", "Todos os padrões foram rotulados! Deseja sair?"):
                self.on_closing()
            return

        self.indice_atual = pendentes[0]
        self.plotar_grafico()
        self.atualizar_painel_info()

    def plotar_grafico(self):
        if self.fig:
            plt.close(self.fig)
        for w in self.frame_grafico.winfo_children():
            w.destroy()

        try:
            row = self.df_trabalho.loc[self.indice_atual]
            ticker, intervalo = row['ticker'], row['intervalo']
            dt_inicio, dt_fim = row['data_inicio'], row['data_fim']

            # Configura Zoom: Baixa um pouco antes e depois para contexto
            buffer_dias = 20
            if 'm' in intervalo:
                buffer_dias = 3
            if 'd' in intervalo or 'wk' in intervalo:
                buffer_dias = 90

            dt_download_start = dt_inicio - pd.Timedelta(days=buffer_dias)
            dt_download_end = dt_fim + pd.Timedelta(days=buffer_dias/2)

            # 1. Download (Usando src)
            df_full = buscar_dados_range(
                ticker, dt_download_start, dt_download_end, intervalo)

            if df_full.empty:
                print(
                    f"Dados vazios para {ticker}. Marcando como erro (-1)...")
                self.marcar_e_avancar(-1)
                return

            # 2. Recalcula ZigZag visualmente (Garante alinhamento)
            estrategia = row.get('estrategia_zigzag')
            params = Config.ZIGZAG_STRATEGIES.get(estrategia, {}).get(
                intervalo, {'depth': 10, 'deviation': 5})
            pivots = calcular_zigzag_oficial(
                df_full, params['depth'], params['deviation'])

            # 3. Recorte Visual (Zoom)
            idx_start = df_full.index.get_indexer(
                [dt_inicio], method='nearest')[0]
            idx_end = df_full.index.get_indexer([dt_fim], method='nearest')[0]

            margin = 25  # Candles de margem
            view_start = max(0, idx_start - margin)
            view_end = min(len(df_full), idx_end + margin)

            df_view = df_full.iloc[view_start:view_end].copy()

            # 4. Plots Auxiliares
            zigzag_series = self._gerar_serie_zigzag(pivots, df_view)

            # Indicadores visuais
            df_view.ta.macd(fast=12, slow=26, signal=9, append=True)
            df_view['RSI'] = ta.rsi(df_view['close'], length=14)

            add_plots = [
                mpf.make_addplot(
                    zigzag_series, color='dodgerblue', width=1.5, panel=0),
                mpf.make_addplot(df_view['RSI'], panel=1,
                                 ylabel='RSI', color='purple'),
                mpf.make_addplot(df_view['MACDh_12_26_9'], type='bar',
                                 panel=2, ylabel='MACD', color='gray', alpha=0.5),
                mpf.make_addplot(df_view['MACD_12_26_9'],
                                 panel=2, color='blue'),
                mpf.make_addplot(df_view['MACDs_12_26_9'],
                                 panel=2, color='orange'),
            ]

            title_txt = f"{ticker} [{intervalo}] - {row['tipo_padrao']} (ID: {self.indice_atual})"

            self.fig, axlist = mpf.plot(
                df_view, type='candle', style='yahoo',
                addplot=add_plots, volume=True, volume_panel=3,
                panel_ratios=(4, 1, 1, 1), figsize=(12, 8),
                returnfig=True, title=title_txt
            )

            # Highlight (Região do Padrão)
            ax_main = axlist[0]
            try:
                x_start = df_view.index.get_indexer(
                    [dt_inicio], method='nearest')[0]
                x_end = df_view.index.get_indexer(
                    [dt_fim], method='nearest')[0]
                ax_main.axvspan(x_start, x_end, color='yellow', alpha=0.15)

                if pd.notna(row['data_destaque']):
                    x_head = df_view.index.get_indexer(
                        [row['data_destaque']], method='nearest')[0]
                    ax_main.axvline(x_head, linestyle='--',
                                    color='gray', alpha=0.5)
            except:
                pass

            self.canvas_widget = FigureCanvasTkAgg(
                self.fig, master=self.frame_grafico)
            self.canvas_widget.draw()
            self.canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        except Exception as e:
            print(f"Erro ao plotar: {e}")
            self.marcar_e_avancar(-1)

    def _gerar_serie_zigzag(self, pivots, df_view):
        if not pivots:
            return pd.Series([np.nan]*len(df_view), index=df_view.index)
        prices = pd.Series(np.nan, index=df_view.index)
        for p in pivots:
            if p['idx'] in df_view.index:
                prices[p['idx']] = p['preco']
        return prices.interpolate(method='linear', limit_direction='both')

    def atualizar_painel_info(self):
        row = self.df_trabalho.loc[self.indice_atual]

        try:
            d_ini = row['data_inicio'].strftime('%Y-%m-%d %H:%M')
            d_fim = row['data_fim'].strftime('%Y-%m-%d %H:%M')
        except:
            d_ini, d_fim = "N/A", "N/A"

        txt = (f"Ativo: {row['ticker']}\n"
               f"Timeframe: {row['intervalo']}\n"
               f"Padrão: {row['tipo_padrao']}\n"
               f"Estratégia: {row.get('estrategia_zigzag', 'N/A')}\n"
               f"Inicio: {d_ini}\n"
               f"Fim:    {d_fim}")
        self.info_label.config(text=txt)

        score = row.get('score_total', 0)
        self.score_final_label.config(
            text=f"SCORE: {score:.0f}", fg="green" if score > 80 else "orange")

        for lbl_n, lbl_v in self.boletim_labels:
            lbl_n.config(text="")
            lbl_v.config(text="")

        idx = 0
        for col_db, nome_visual in self.regras_map.items():
            if col_db in row and idx < self.max_rule_slots:
                val = bool(row[col_db])
                lbl_n, lbl_v = self.boletim_labels[idx]
                lbl_n.config(text=nome_visual)
                lbl_v.config(text="SIM" if val else "NÃO",
                             fg="green" if val else "red")
                idx += 1

    def on_key_press(self, event):
        key = event.keysym.lower()
        if key == 'a':
            self.marcar_e_avancar(1)
        elif key == 'r':
            self.marcar_e_avancar(0)
        elif key == 'q':
            self.on_closing()

    def marcar_e_avancar(self, label):
        if self.df_trabalho is not None:
            self.df_trabalho.loc[self.indice_atual, 'label_humano'] = label
            self.df_trabalho.to_csv(
                self.arquivo_saida, index=False, date_format='%Y-%m-%d %H:%M:%S')
            self.carregar_proximo_padrao()

    def on_closing(self):
        plt.close('all')
        self.destroy()
        print("Encerrando Labeler...")


if __name__ == '__main__':
    if not os.path.exists(Config.FINAL_CSV_PATH):
        messagebox.showerror("Erro",
                             f"Arquivo não encontrado:\n{Config.FINAL_CSV_PATH}\n\nRode o generator.py primeiro.")
    else:
        # Salva o arquivo rotulado na mesma pasta com sufixo _labeled
        arquivo_saida = Config.FINAL_CSV_PATH.replace('.csv', '_labeled.csv')
        app = LabelingTool(Config.FINAL_CSV_PATH, arquivo_saida)
        app.mainloop()
