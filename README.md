# Molecule Lab Backend

Backend FastAPI do Molecule Lab, projeto educacional de simulação molecular para o Ciência Aberta no CNPEM, desenvolvido no contexto da ILUM Escola de Ciência.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![RDKit](https://img.shields.io/badge/RDKit-2024.3+-336791)
![Pytest](https://img.shields.io/badge/Pytest-8+-0A9EDC?logo=pytest&logoColor=white)

---

## Visão Geral

O serviço recebe uma molécula desenhada no frontend como grafo, valida a estrutura, converte para RDKit, calcula propriedades estáticas e executa uma simulação qualitativa de dinâmica molecular. O progresso e o resultado da simulação são enviados para o frontend por Server-Sent Events (SSE).

Este repositório é a contraparte de API do projeto `molecule-lab-frontend`.

## Funcionalidades

- Validação de grafos moleculares enviados pelo frontend.
- Conversão de grafo para molécula RDKit e SMILES canônico.
- Cálculo de fórmula molecular, massa molecular e propriedades estáticas.
- Simulação qualitativa BDE/ranking com parâmetros calibrados por Optuna.
- Streaming de progresso por SSE com eventos de metadata, progresso, cache e resultado.
- Cache em memória para resultados determinísticos repetidos por molécula, preset, seed e versão dos dados BDE/Optuna.
- Testes para química, simulação e fluxo HTTP/SSE.

## Escopo Científico

Esta é uma simulação educacional para explorar tendências qualitativas. O modelo integrado usa geometria inicial ETKDG/UFF, ligações harmônicas, repulsão WCA, integrador Velocity Verlet com termostato de Berendsen, rampa de temperatura e detecção de quebra calibrada por tabela BDE e parâmetros Optuna.

Os resultados não devem ser usados como previsão experimental de alta fidelidade. O objetivo é apoiar interação didática, visualização de possíveis quebras e comparação qualitativa entre moléculas pequenas.

## Requisitos

- Python 3.11+
- `pip`
- Dependências declaradas em `pyproject.toml`

## Configuração

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

Instale o pacote em modo editável com dependências de desenvolvimento:

```bash
pip install -e ".[dev]"
```

## Execução

Inicie o servidor local:

```bash
uvicorn main:app --reload --port 8000
```

A API ficará disponível em:

```text
http://localhost:8000
```

Verifique a saúde do serviço:

```bash
python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/api/health').read().decode())"
```

## Contrato da API

### Healthcheck

```http
GET /api/health
```

Resposta:

```json
{
  "status": "ok",
  "service": "Molecule Lab API"
}
```

### Criar Simulação

```http
POST /api/simulations
```

Corpo:

```json
{
  "preset": "fast",
  "seed": 42,
  "graph": {
    "atoms": [
      { "id": "c1", "symbol": "C", "x": 0, "y": 0 }
    ],
    "bonds": []
  }
}
```

Campos:

- `graph.atoms`: lista obrigatória com ao menos um átomo.
- `graph.bonds`: lista opcional de ligações.
- `preset`: `fast`, `balanced` ou `debug`. O padrão é `fast`.
- `seed`: opcional, usado para reproduzir a simulação.

Resposta:

```json
{
  "simulation_id": "abc123",
  "preset": "fast",
  "molecule": {
    "smiles": "C",
    "formula": "CH4",
    "molecular_weight": 16.043,
    "properties": {
      "num_atoms": 1,
      "num_bonds": 0,
      "num_rings": 0,
      "is_aromatic": false
    }
  },
  "events_url": "/api/simulations/abc123/events"
}
```

### Acompanhar Simulação

```http
GET /api/simulations/{simulation_id}/events
```

O endpoint retorna `text/event-stream` com os seguintes eventos:

- `metadata`: dados da simulação e da molécula.
- `progress`: passo atual, progresso, temperatura e energia.
- `cache_hit`: resultado recuperado do cache em memória para a mesma molécula, preset, seed e versão dos dados BDE/Optuna.
- `result`: resultado final `stable` ou `break`.
- `error`: erro estruturado com `code` e `message`.

## Presets

| Preset | Uso |
|---|---|
| `fast` | Padrão para interação fluida no frontend. |
| `balanced` | Mais passos para uma simulação menos curta. |
| `debug` | Determinístico e rápido para testes. |

## Testes

```bash
pytest
```

Os testes cobrem validação do grafo, conversão para SMILES, propriedades RDKit, topologia, detecção controlada de quebra, cache/determinismo e fluxo SSE da API.

## Estrutura

```text
.
├── main.py                    # Entrypoint ASGI para Uvicorn
├── pyproject.toml             # Metadados e dependências Python
├── src/
│   └── molecule_lab/
│       ├── api/               # App FastAPI, rotas, schemas e registry
│       ├── chem/              # Validação, conversão e propriedades RDKit
│       ├── core/              # Configuração, erros e logging
│       └── simulation/        # Parâmetros, topologia, forças e engine
└── tests/                     # Testes automatizados
```

## Integração com o Frontend

Para uso local com `molecule-lab-frontend`, mantenha este backend rodando em `http://localhost:8000` e configure o frontend com:

```env
VITE_API_BASE_URL=http://localhost:8000
```

## Licença

Projeto de uso educacional e institucional da ILUM Escola de Ciência.
