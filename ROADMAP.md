# SQA-ML v2.0 — Roadmap e histórico técnico

## Status geral

| Área | Situação |
|---|---|
| Execução real | LIVE desativado por trava explícita |
| Ambiente padrão | PAPER, com estado persistido |
| Validação temporal | Purged Walk-Forward com embargo e `t1` por evento |
| Seleção atual | 13 features aprovadas por MDA médio positivo |
| Modelo ETH reportado | OOS ROC AUC 0,5334; calibração de 24h reprovada no holdout |
| Retreino automático | Staging + aprovação de holdout antes de promover modelo |
| Risco | Kelly fracionado, limite diário/global, exposição e circuito de volatilidade |

**Conclusão de pesquisa:** ainda não há edge comprovado para capital real. A análise de barreira temporal identificou 12h como melhor candidata em um teste: PF OOS 1,2897, retorno holdout +0,34% e PF holdout 1,1173. Como o holdout não alcança PF 1,20, o resultado não satisfaz o critério robusto; continua não recomendado operar.

## Fases da auditoria e estabilização

| Fase | Entregas registradas |
|---|---|
| Fase 1 — Ingestão e segurança | Coleta OHLCV resiliente, modo LIVE bloqueado durante auditoria, paridade de features em janela de inferência e persistência atômica de PAPER com recuperação de backup. |
| Fase 2 — Execução | Correção do caixa/PnL SHORT no PaperTrader, execução PAPER sem inicializar CCXT, resolução dinâmica do `.env` e estados explícitos de ordem. |
| Fase 3 — Simulação | Engine contínuo barra a barra, barreiras coerentes com rotulagem, custos, SL/TP, gaps e barreira temporal. |
| Fase 4 — Validação | Purged Walk-Forward, remoção de warmup, validação de paridade e calibração em previsões OOS com holdout temporal separado. |

## Etapas de arquitetura v2.0

| Etapa | Entregas | Resultado / salvaguarda |
|---|---|---|
| 1 — FSM de ordens | `src/execution/fsm.py`, transições válidas/terminais, histórico de estado e integração PAPER para ordens pendentes, preenchidas e rejeitadas. Testes em `tests/test_fsm.py`. | Transições inválidas são rejeitadas; LIVE/CCXT está desativado. |
| 2 — PSI Data Drift | `src/validation/drift.py`, PSI por feature e proteção no runner para drift severo ou referência indisponível. Testes em `tests/test_drift.py`. | PSI > 0,25 pausa novas entradas; a proteção não encerra posições abertas. |
| 3 — Expansão de fatores alpha | Pipeline com 32 features derivadas; fatores de momentum, volatilidade, volume e momentos estatísticos. Primeiro rebuild e treino OOS. | OOS AUC 0,5105; candidato de threshold falhou no holdout, sem autorização para operar. |
| 4 — Seleção MDA e t-barrier | `src/features/selection.py`, `scripts/select_features.py`, 13 features positivas em `config/selected_features.json`; `scripts/test_barrier_sensitivity.py` compara 12h/24h/48h. | MDA: OOS AUC ETH 0,5334; threshold 0,50 teve holdout PF 0,6032 e retorno -12,07%. A janela 12h teve PF OOS 1,2897, mas PF holdout 1,1173 (abaixo do alvo robusto 1,20). |
| 5 — Risco, autotrain e documentação | `src/risk/portfolio.py` integrado ao runner/engine/PaperTrader; `scripts/autotrain.py` treina e calibra em staging; README e ROADMAP v2.0. | Defaults: Kelly 0,25, DD diário 3%, DD global 15%, exposição 20%, bloqueio ATR/preço ≥ 10%. Lock impede novas entradas até aprovação da calibração/holdout; lock de execução é liberado pelo kernel em encerramentos abruptos. |

As fases de auditoria e estabilização (ingestão, persistência, engine contínuo e validação) estão resumidas separadamente acima; as cinco etapas desta tabela registram a evolução de arquitetura e pesquisa da versão 2.0.

## Arquitetura atual de risco

1. **Sinal primário:** squeeze (`vol_ratio_6_24`) e rompimento de VWAP determinam lado; o modelo de meta-labeling estima probabilidade de sucesso.
2. **Filtro de regime:** PSI verifica features live contra referência de treino. PSI severo ou referência ausente ativa o modo protetor.
3. **Dimensionamento:** Kelly cheio \(f^*=(pb-(1-p))/b\), multiplicado pela fração de segurança e por um escalar inverso de ATR/preço. O notional é limitado à exposição aberta restante. Backtest e calibração OOS usam o mesmo dimensionamento, custos e circuit breakers, com base nas probabilidades OOS.
4. **Circuit breakers:** limites diário/global, exposição total e volatilidade bloqueiam criação de novas ordens. A FSM mantém ciclo de vida e histórico de cada ordem.
5. **Execução:** PAPER é o único modo permitido pelo runner e pelo engine; CCXT não é usado para execução de ordens. O engine respeita os limites antes de encaminhar entradas ao PaperTrader.
6. **Retreino:** ingestão, features, dataset, MDA, treino e calibração são encadeados pelo autotrain. Artefatos candidatos ficam em staging; o lock só é removido após promoção aprovada.

O engine e o runner recusam qualquer modo diferente de PAPER; o caminho CCXT de execução LIVE foi removido. Feature selection exclui o intervalo temporal final reservado ao holdout. PSI compara as features live com a matriz bruta de features de treino, em vez do subconjunto condicionado a eventos rotulados.

## Critérios de promoção do modelo

O autotrain exige um threshold que alcance PF de calibração ≥ 1,0 com ao menos 20 trades. Além disso, o holdout temporal deve ter PF > 1,0 e retorno positivo. Se qualquer etapa falhar, não substitui modelo/metadata/features selecionadas de produção e deixa `models/saved/autotrain_safety_lock.json` ativo.

Esses critérios operacionais são um gate mínimo, não evidência de robustez estatística. A aprovação futura para LIVE exige auditoria específica, validação fora da amostra adicional, custos/slippage realistas, estabilidade por regime e autorização explícita; a trava LIVE permanece vigente.
