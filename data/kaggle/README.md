# Dataset — Bank Marketing

## Fonte

| Campo | Valor |
|---|---|
| **Nome** | Bank Marketing (with social/economic context) |
| **Autores** | Moro, S.; Cortez, P.; Rita, P. (2014) |
| **Link Kaggle** | https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing |
| **Link UCI** | http://archive.ics.uci.edu/ml/datasets/Bank+Marketing |
| **Arquivo usado** | `bank-additional-full.csv` |
| **Instâncias** | 41.188 |
| **Features** | 20 + target |
| **Licença** | CC BY 4.0 (uso educacional) |
| **Versão** | Maio/2008 – Novembro/2010 |

## Como baixar

```bash
# Via Kaggle CLI
kaggle datasets download henriqueyamahata/bank-marketing
unzip bank-marketing.zip -d data/kaggle/

# Ou copiar manualmente o arquivo bank-additional-full.csv para esta pasta
```

## Colunas utilizadas

| Coluna | Tipo | Uso |
|---|---|---|
| age | numérico | age_group (feature de contexto) |
| job | categórico | job_group (feature de contexto) |
| marital | categórico | feature de contexto |
| education | categórico | feature de contexto |
| default | categórico | elegibilidade (sem_oferta se default=yes) |
| housing | categórico | feature de contexto |
| loan | categórico | feature de contexto |
| contact | categórico | mapeado para channel |
| month | categórico | sazonalidade |
| day_of_week | categórico | sazonalidade |
| campaign | numérico | frequência de contato |
| pdays | numérico | recência |
| previous | numérico | previous_contact flag |
| poutcome | categórico | mapeado para segment |
| emp.var.rate | numérico | indicador econômico |
| cons.price.idx | numérico | indicador econômico |
| cons.conf.idx | numérico | indicador econômico |
| euribor3m | numérico | indicador econômico |
| nr.employed | numérico | indicador econômico |
| **y** | binário | **target: conversão simulada como recompensa** |

## Colunas descartadas

| Coluna | Motivo |
|---|---|
| **duration** | **Vazamento temporal**: só conhecida após o contato encerrar. Descarte obrigatório para modelo realista. |

## Limitações

- Dataset de telemarketing bancário português (2008–2010) — contexto econômico diferente do atual
- Taxa de conversão real: ~11% (desbalanceamento)
- `duration` altamente correlacionada com `y` mas inutilizável em produção
- Não representa dados reais de clientes — uso exclusivamente educacional
