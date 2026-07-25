# Datathon 7-MLET — Experimentação Adaptativa em Ofertas Financeiras

**Turma:** MLET7 | **Fase:** 05 — Datathon

## Problema

Uma instituição financeira digital precisa decidir qual oferta apresentar para cada cliente elegível em diferentes canais. Regras fixas e testes A/B desperdiçam tráfego e demoram para reagir a mudanças. Esta plataforma aprende com as interações usando **Multi-Armed Bandits** (Thompson Sampling + Nilos-UCB), equilibrando exploração e explotação continuamente até concentrar o tráfego no braço de maior conversão.

> **Escopo do bandit:** a versão atual é um bandit **global** (não-contextual) — aprende o melhor braço no agregado, sem condicionar a decisão ao segmento/canal do cliente. O contexto é recebido e registrado para auditoria; torná-lo contextual é a evolução natural (ver *Limitações*).

## Dataset

**Base:** Bank Marketing (Moro et al., 2014) — Kaggle  
**Link:** https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing  
**Arquivo:** `bank-additional-full.csv` (41.188 clientes, 20 features + target)  
**Coluna descartada:** `duration` (vazamento temporal — só conhecida após o contato)

## Braços (arms)

| Arm | Descrição |
|---|---|
| `sem_oferta` | Nenhuma oferta ativa |
| `deposito_prazo_basico` | Pitch de depósito a prazo básico |
| `deposito_prazo_premium` | Depósito a prazo com taxa diferenciada |
| `educacao_financeira` | Conteúdo educativo (abordagem indireta) |

## Quickstart

```bash
# 1. Instalar
pip install -e ".[dev]"
cp .env.example .env

# 2. Preparar dados (coloque bank-additional-full.csv em data/kaggle/)
make prepare

# 3. Simular e registrar no MLflow
make simulate

# 4. Servir API
make serve                    # http://localhost:8000/docs

# 5. Testar
make test
```

## Exemplo de uso da API

```bash
# Pedir decisão
curl -X POST http://localhost:8000/decide \
  -H "Content-Type: application/json" \
  -d '{"subject_key":"c001","channel":"app","segment":"recorrente","context":{"age_group":"35-50","job_group":"management"},"mode":"thompson_sampling"}'

# Registrar recompensa
curl -X POST http://localhost:8000/reward \
  -d '{"decision_id":"<id>","reward":1.0}'

# Status dos braços
curl http://localhost:8000/status
```

## Baseline vs. Adaptativo (Etapa 3)

Comparação por *replay* dos mesmos eventos (notebook, seção 7). O **Thompson Sampling** supera o **Baseline** em recompensa média e converge mais rápido para o braço ótimo (`deposito_prazo_premium`). O Baseline guloso, sem exploração e com recompensas esparsas, pode travar em um braço subótimo.

## 5 Casos de Teste (Etapa 4 — Golden Set simplificado)

Cinco clientes de exemplo, com Baseline e Thompson treinados em 500 eventos (seed fixa — reprodutível na seção 9 do notebook):

| Cliente | Perfil | Baseline | Thompson | Faz sentido? |
|---|---|---|---|---|
| Cliente 1 | recorrente / app | `sem_oferta` | `deposito_prazo_premium` | ✅ Thompson acerta o braço ótimo; baseline travou |
| Cliente 2 | novo / web | `sem_oferta` | `deposito_prazo_premium` | ✅ idem |
| Cliente 3 | reativado / telemarketing | `sem_oferta` | `deposito_prazo_premium` | ✅ idem |
| Cliente 4 | recorrente / app (só `premium` disponível) | `deposito_prazo_premium` | `deposito_prazo_premium` | ✅ respeita o único braço elegível |
| Cliente 5 | novo / app | `sem_oferta` | `deposito_prazo_premium` | ✅ contraste baseline × adaptativo |

**Leitura:** o Thompson recomenda consistentemente o braço de maior conversão simulada, enquanto o Baseline guloso ficou preso em `sem_oferta` — demonstrando na prática o ganho da política adaptativa. Como o bandit é global, a recomendação do Thompson não varia com o perfil (ver *Limitações*).

## MLflow (Etapa 7)

Os parâmetros e métricas da Etapa 3 são registrados no MLflow (seção 7.1 do notebook e `make simulate`):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

## Arquitetura-alvo em Nuvem — AWS (Etapa 6)

Para colocar o projeto no ar na **AWS**, a **API FastAPI** rodaria em um contêiner gerenciado no **AWS App Runner** (ou **Amazon ECS/Fargate**), atrás do **Amazon API Gateway** para autenticação e *rate limiting*, escalando sob demanda — inclusive a zero quando ocioso, o que elimina custo no período de experimentação. Os **logs de decisão** (auditáveis) e o dataset processado ficariam no **Amazon S3**, e os segredos/config no **AWS Secrets Manager**.

O rastreamento de experimentos e o versionamento das políticas usariam o **Amazon SageMaker** com **MLflow** (tracking + Model Registry). A observabilidade (latência, taxa de erro e queda de recompensa média) ficaria no **Amazon CloudWatch**, com alarmes. O retreino rodaria em pipeline agendado (SageMaker Pipelines / EventBridge), mas a **promoção de uma nova política exige aprovação humana** — não há promoção automática para produção.

| Componente | Serviço AWS |
|---|---|
| API (contêiner) | AWS App Runner ou ECS/Fargate |
| Gateway / autenticação | Amazon API Gateway |
| Logs de decisão + dataset | Amazon S3 |
| Tracking de experimentos (MLflow) | Amazon SageMaker |
| Observabilidade / alarmes | Amazon CloudWatch |
| Segredos e configuração | AWS Secrets Manager |

## Estrutura

```
datathon-7mlet-grupo/
├── notebooks/
│   └── 01-eda-e-baseline.ipynb  # EDA, baseline vs adaptativo, MLflow, 5 casos
├── src/datathon_offerexp/
│   ├── contracts.py          # Dataclasses: SyntheticOfferEvent, DecisionLog
│   ├── policies.py           # Baseline, ThompsonSampling, NilosUCB
│   ├── evaluation.py         # Métricas offline: regret, fairness
│   ├── decision_log.py       # Log auditável (SQLite)
│   └── app.py                # API FastAPI (Etapa 5)
├── src/tests/                # pytest ≥60% cobertura
├── scripts/                  # prepare_data (Etapa 2), run_simulation (MLflow)
├── data/
│   ├── kaggle/               # Dataset original (não versionado no Git)
│   ├── processed/            # Gerado por prepare_data.py
│   └── golden_set/           # 5 casos de avaliação (Etapa 4)
└── docs/roteiro-video.md     # Roteiro do vídeo (Etapa 8)
```

## Limitações

- **Bandit global (não-contextual):** a decisão usa apenas as estatísticas agregadas dos braços; o segmento/canal do cliente é registrado, mas não altera a escolha. Evolução natural: bandit contextual (LinUCB / Thompson contextual).
- Recompensas simuladas (não dados reais de conversão)
- Política não persiste entre reinicializações da API (usar MLflow para reload)
- Arquitetura de nuvem é plano de deploy, não implantação ativa
