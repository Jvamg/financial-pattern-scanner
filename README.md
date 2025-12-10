# Financial Pattern Scanner (Quantitative Analysis) 📈

Um sistema modular de análise quantitativa em Python para detectar, validar e rotular padrões gráficos clássicos (OCO, Topo Duplo, Topo Triplo) em mercados financeiros, focado em Criptomoedas.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-Stable-success)
![Data](https://img.shields.io/badge/Data-Yahoo%20Finance-purple)

## 📋 Sobre o Projeto

Este projeto automatiza a identificação de padrões técnicos utilizando dados históricos. Diferente de scripts simples, ele utiliza uma arquitetura profissional separada em:

1.  **Backend de Detecção (`generator`):** Algoritmos baseados em pivôs ZigZag e regras rígidas de geometria e indicadores técnicos.
2.  **Frontend de Visualização (`labeler`):** Interface gráfica para inspeção visual humana e criação de datasets rotulados ("Human-in-the-loop").

### Funcionalidades
* **Detecção Multicamada:** Identifica padrões OCO (Ombro-Cabeça-Ombro), Topos/Fundos Duplos e Triplos.
* **Validação Técnica:** Aplica regras de simetria, profundidade e confirmação via indicadores (RSI, MACD, Estocástico, Volume).
* **Dados Gratuitos:** Pipeline integrado ao **Yahoo Finance** com tratamento automático de *resampling* (ex: converte dados de 1h para 4h).
* **Interface de Rotulagem:** GUI em Tkinter/Matplotlib para validar os padrões encontrados visualmente.

## 🛠️ Estrutura do Projeto

```text
financial-pattern-scanner/
├── config.py           # Configurações globais (Ativos, Estratégias, Pesos)
├── requirements.txt    # Dependências do projeto
├── src/                # Núcleo de Lógica (Core)
│   ├── data_loader.py  # Conexão com Yahoo Finance e limpeza de dados
│   ├── indicators.py   # Cálculos matemáticos (ZigZag, RSI, MACD)
│   └── patterns.py     # Lógica de geometria e regras de detecção
├── tools/              # Ferramentas Executáveis
│   ├── generator.py    # Script que baixa dados e gera o dataset de candidatos
│   └── labeler.py      # GUI para visualizar e validar os padrões
└── data/               # Armazenamento de CSVs (Datasets)
```
## 🚀 Como Executar
1. **Instalação:**\n
Clone o repositório e instale as dependências:

```bash

git clone [https://github.com/Jvamg/financial-pattern-scanner.git](https://github.com/Jvamg/financial-pattern-scanner.git)
cd financial-pattern-scanner
pip install -r requirements.txt
```
Nota: Se tiver problemas com o pandas-ta, instale a versão de desenvolvimento:

```bash
pip install git+https://github.com/twopirllc/pandas-ta.git@development
```

2. **Gerar Padrões (`Scanner`):**\n
Execute o gerador para escanear os ativos configurados. Ele baixará os dados e aplicará as regras matemáticas.

```bash
# Exemplo: Escanear BTC e ETH usando a estratégia de Swing Trade
python tools/generator.py --tickers BTC-USD,ETH-USD --strategies swing_short
```
Isso criará um arquivo CSV em data/datasets/dataset_patterns.csv.

3. **Visualizar e Rotular (`Labeler`):**\n
Abra a interface gráfica para validar os padrões encontrados pelo robô:

```bash

python tools/labeler.py
```

Comandos na Interface:

[A] ou [Seta Direita]: Aprovar Padrão (Válido)

[R]: Rejeitar Padrão (Inválido)

[Q]: Sair e Salvar

## ⚙️ Calibração e Configuração
Você pode ajustar a sensibilidade do robô no arquivo config.py:

ZIGZAG_STRATEGIES: Ajuste a profundidade (depth) e desvio (deviation) para pegar movimentos maiores ou menores.

SCORE_WEIGHTS: Defina o peso de cada regra (ex: simetria vale 10 pts, RSI vale 15 pts).

TOLERANCES: Ajuste a rigidez geométrica (ex: o quão alinhados os ombros precisam estar).

## ⚠️ Disclaimer
Esta ferramenta é apenas para fins educacionais e de pesquisa quantitativa. Não constitui recomendação de investimento.