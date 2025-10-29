"""
ner_cluster_pipeline.py
=======================

Pipeline único para *extrair NER, **normalizar* e *agrupar operações* a partir
de metadados de vídeos (título, descrição e opcionalmente transcrição). Mantém a
*lógica original de agrupamento* (janelas temporais +/- 15 dias, criação de sufixos
-2, -3… quando necessário), adiciona *cache* para a etapa cara (NER), melhora a
*experiência no terminal* (console limpo + barra de progresso) e salva **logs
detalhados em arquivo**.

----------------------------------------------------------------------
OBJETIVO
----------------------------------------------------------------------
1) Identificar entidades (LOC, PER, EVENT, DATE, LAW e regra spaCy "OPERACAO").
2) Saneá-las (apenas limpeza de ruído, sem alterar a lógica decisória).
3) Atribuir um rótulo de operação por linha, reaproveitando janelas temporais.
4) Escrever um TSV/CSV final com todas as colunas originais + operation_ner.

----------------------------------------------------------------------
FUNCIONAMENTO (visão geral)
----------------------------------------------------------------------
- Lê o arquivo de entrada.
- Inicializa modelos:
  - Hugging Face: Babelscape/wikineural-multilingual-ner
  - spaCy: pt_core_news_lg + EntityRuler (regra "OPERACAO")
- Extrai NER de titulo e descricao (e transcribedText, se existir).
- Salva um *cache* .cache/ner_results_<year-tag>.tsv (reutilizado com --use-cache).
- *Normaliza* colunas NER (minúsculas, remoção de acentos/pontuação; apenas saneamento).
- *Agrupa* por janelas de tempo (+/- 15 dias) e atribui operation_ner.
- Escreve a saída e imprime um *sumário* amigável no console.

----------------------------------------------------------------------
ENTRADA (arquivo CSV/TSV)
----------------------------------------------------------------------
Colunas mínimas esperadas:
- id_video (string)
- titulo (string)
- descricao (string)
- data_postagem (datetime)

Colunas opcionais úteis:
- operation (para filtros auxiliares já existentes no seu fluxo)
- transcribedText (string, se quiser aplicar NER em transcrições)

Separador de entrada/saída configurável por --sep (default: \t).

----------------------------------------------------------------------
SAÍDA (arquivo CSV/TSV final)
----------------------------------------------------------------------
- Todas as colunas originais da entrada.
- + operation_ner (string) — rótulo atribuído pelo agrupamento.
- Se você usar o cache como etapa de trabalho, o arquivo de cache conterá também:
  - ner_titulo, ner_descricao, (ner_transcribedText, se existir)
  - all_ners (coluna intermediária usada na identificação de genéricos)

----------------------------------------------------------------------
PARAMETROS CLI
----------------------------------------------------------------------
--input <str>                : Caminho do CSV/TSV de entrada.
--year-tag <str>             : Rótulo para nome do cache (ex.: "2022").
--final-output <str>         : Caminho do arquivo final (TSV/CSV).
--cache-dir <str>            : Diretório de cache (default: ".cache").
--use-cache                  : Reutiliza cache se compatível.
--force                      : Ignora cache e refaz a extração NER.
--sep <str>                  : Separador I/O (default: "\t").
--log-level {DEBUG..}        : Nível de logs (arquivo). Console é limpo por padrão.
--check-setup                : Apenas valida ambiente/modelos e encerra.
--summary-head-percent <int> : Percentual (1–99) usado no sumário (default: 20).
--version                    : Mostra versão do script.


----------------------------------------------------------------------
AMBIENTE E INSTALAÇÃO
----------------------------------------------------------------------

1- *Crie e ative um ambiente virtual (venv)*

Windows (PowerShell):
---------------------
python -m venv venv
.\venv\Scripts\Activate.ps1

Linux/macOS (bash/zsh):
-----------------------
python3 -m venv venv
source venv/bin/activate

2- *Instale as dependências*
------------------------------
Com o ambiente ativo, rode:
pip install -r requirements.txt

> O arquivo requirements.txt contém todas as bibliotecas necessárias
> (pandas, torch, transformers, spacy, tqdm, etc.)

3- *Baixe e verifique o modelo spaCy*
----------------------------------------
python -m spacy download pt_core_news_lg
ou
pip install "https://github.com/explosion/spacy-models/releases/download/pt_core_news_lg-3.8.0/pt_core_news_lg-3.8.0-py3-none-any.whl"   

Depois, confirme se está tudo certo:
python -c "import spacy; nlp = spacy.load('pt_core_news_lg'); print('Modelo OK:', nlp.meta.get('lang'), nlp.meta.get('name'))"

> Saída esperada:
> Modelo OK: pt core_news_lg

4- *(Opcional) GPU / CUDA*
-----------------------------
Se tiver GPU NVIDIA e quiser acelerar a extração NER:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Verifique se o PyTorch detecta sua GPU:
python -c "import torch; print(torch.cuda.is_available())"
# True → GPU pronta
# False → será usada CPU normalmente

----------------------------------------------------------------------
EXEMPLOS DE USO
----------------------------------------------------------------------
PowerShell (Windows)
--------------------
# Rodando com cache (recomendado após a 1ª execução)
python .\\ner_cluster_pipeline.py `
  --input ".\\dummy.csv" `
  --year-tag "YEAR" `
  --final-output ".\\out.tsv" `
  --use-cache --log-level INFO

# Forçando refazer a NER (ignora cache)
python .\\ner_cluster_pipeline.py `
  --input ".\\dummy.csv" `
  --year-tag "YEAR" `
  --final-output ".\\out.tsv" `
  --force --log-level INFO

# Apenas verificar ambiente e modelos
python .\\ner_cluster_pipeline.py `
  --input ".\\dummy.csv" `
  --year-tag "YEAR" `
  --final-output ".\\out.tsv" `
  --check-setup

Bash (Linux/macOS)
------------------
python ./ner_cluster_pipeline.py \
  --input "./dummy.csv" \
  --year-tag "YEAR" \
  --final-output "./out.tsv" \
  --use-cache --log-level INFO

----------------------------------------------------------------------
PROGRESSO E LOGS
----------------------------------------------------------------------
- *Console: mensagens curtas de fase + **barra de progresso* (tqdm) na extração NER.
- *Arquivo de log*: .cache/logs_<year-tag>.log com detalhes (nível do --log-level).
- Transformers com verbosidade reduzida (apenas erros).

----------------------------------------------------------------------
CÓDIGOS DE SAÍDA (exit codes)
----------------------------------------------------------------------
0  : Sucesso
2  : Validação de entrada falhou
3  : --check-setup falhou (ambiente/modelos)
4  : Erro ao ler arquivo completo
5  : Falha ao inicializar modelos
6  : Erro na normalização / DataFrame vazio
130: Interrupção pelo usuário (Ctrl+C)

----------------------------------------------------------------------
REQUISITOS
----------------------------------------------------------------------
Python 3.10+ (recomendado)
pip install:
  - pandas
  - torch (CPU ou GPU, conforme seu ambiente)
  - transformers
  - spacy
  - tqdm   (opcional, para barra de progresso)

Modelos:
  - spaCy pt:  pt_core_news_lg
    * Se o download via python -m spacy download pt_core_news_lg falhar,
      instale via wheel oficial da Explosion (compatível com sua versão do spaCy).

GPU (opcional):
  - Se CUDA estiver disponível, a pipeline HF usará GPU automaticamente.
----------------------------------------------------------------------
SOLUÇÃO DE PROBLEMAS
----------------------------------------------------------------------
- Erro spaCy ao carregar pt_core_news_lg:
  * Instale via: python -m spacy download pt_core_news_lg
  * ou via wheel: pip install https://github.com/explosion/spacy-models/.../pt_core_news_lg-<ver>-py3-none-any.whl
- Barra de progresso não aparece:
  * Garanta pip install tqdm. Sem tqdm, o script roda normalmente, só sem a barra.
- Mensagens excessivas no console:
  * O console é intencionalmente limpo; detalhes estão no arquivo .cache/logs_<year-tag>.log.

----------------------------------------------------------------------
LICENÇA
----------------------------------------------------------------------
Este script utiliza modelos de terceiros (Hugging Face / spaCy) que possuem
licenças próprias. Verifique as licenças dos modelos antes de uso comercial.

"""

from __future__ import annotations

import warnings
import argparse
import datetime
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union, cast, TypedDict

import pandas as pd
import torch
import unicodedata
import re

import spacy
from spacy.pipeline import EntityRuler

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from transformers.pipelines import TokenClassificationPipeline
from transformers.utils import logging as hf_logging

# Silencia ruído do Transformers no console
hf_logging.set_verbosity_error()

# Silencia aviso de Period com timezone
warnings.filterwarnings(
    "ignore",
    message="Converting to PeriodArray/Index representation will drop timezone information.",
    category=UserWarning
)

# Barra de progresso (opcional)
try:
    from tqdm import tqdm  # type: ignore[import-untyped]
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

version = "0.3.3-fase4-ux-logic-original"


# =========================
# Logging / CLI utilities
# =========================

def setup_logging(level: str, log_file: str) -> Tuple[logging.Logger, logging.Logger]:
    """
    Retorna (APP_LOG, UX_LOG):
      - APP_LOG: use .debug/.info em TODO o pipeline (vai para arquivo).
      - UX_LOG : use só para mensagens amigáveis ao usuário no console.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)  # capturar tudo e rotear conforme handler

    # arquivo (detalhado)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    # console (limpo)
    ch = logging.StreamHandler(sys.stdout)

    class OnlyUX(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.name == "UX"

    ch.addFilter(OnlyUX())
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)

    app_log = logging.getLogger("APP")
    ux_log = logging.getLogger("UX")

    ux_log.info(f"ner_cluster {version}")
    ux_log.info(f"[logs completos] {log_file}")

    return app_log, ux_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ner_cluster_pipeline",
        description="Pipeline único para extrair NER, normalizar e agrupar operações a partir de vídeos.",
    )
    parser.add_argument("--input", required=True, help="CSV/TSV de entrada.")
    parser.add_argument("--year-tag", required=True, help="Rótulo para cache.")
    parser.add_argument("--final-output", required=True, help="Arquivo final TSV/CSV.")
    parser.add_argument("--cache-dir", default=".cache", help="Diretório de cache.")
    parser.add_argument("--use-cache", action="store_true", help="Reutiliza cache se compatível.")
    parser.add_argument("--force", action="store_true", help="Refaz tudo ignorando cache.")
    parser.add_argument("--sep", default="\t", help=r"Separador de I/O (default: '\t').")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Nível de logs (arquivo recebe detalhado; console fica clean).",
    )
    parser.add_argument("--check-setup", action="store_true", help="Valida setup e sai.")
    parser.add_argument(
        "--summary-head-percent",
        type=int,
        default=20,
        help="Percentual para métrica 'operações nos primeiros X%% dos dados' (default: 20).",
    )
    parser.add_argument("--version", action="version", version=version)
    return parser.parse_args()


def print_params_clean(args: argparse.Namespace, log_file: str) -> None:
    print("=== Execução ===")
    print(f"• Entrada     : {args.input}")
    print(f"• Saída final : {args.final_output}")
    print(f"• Cache dir   : {args.cache_dir}")
    print(f"• Year tag    : {args.year_tag}")
    print(f"• Sep I/O     : {'\\t' if args.sep == '\\t' else args.sep}")
    print(f"• Logs        : {log_file}")
    if args.use_cache:
        print("• Cache       : reutilizar se disponível")
    if args.force:
        print("• Forçar      : refazer mesmo com cache")
    if args.check_setup:
        print("• Modo        : check-setup\n")


def quick_validate_input(input_path: str, sep: str) -> bool:
    try:
        sample = pd.read_csv(input_path, sep=sep, nrows=50)
        cols = set(sample.columns.tolist())
        expected_min = {"id_video", "titulo", "descricao"}
        missing = expected_min - cols
        if missing:
            logging.warning(f"[check] colunas mínimas ausentes na amostra: {missing}")
        logging.info(f"[check] amostra ok: {len(sample)} linhas | colunas: {list(sample.columns)}")
        return True
    except Exception as e:
        logging.error(f"[check] falha ao ler entrada: {e}")
        return False


# =========================
# Modelos (HF e spaCy)
# =========================

class SpacyTokenPattern(TypedDict, total=False):
    LOWER: Union[str, Dict[str, Any]]

def init_models() -> Tuple[TokenClassificationPipeline, spacy.Language, EntityRuler]:
    logging.info("[1/4] Inicializando modelos…")
    # HF
    tokenizer_hf = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
    model_hf = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
    device: int | str = 0 if torch.cuda.is_available() else -1
    logging.info(f" - Device: {'GPU' if device == 0 else 'CPU'}")

    nlp_hf = cast(
        TokenClassificationPipeline,
        pipeline(
            task="token-classification",
            model=model_hf,
            tokenizer=tokenizer_hf,
            aggregation_strategy="simple",
            device=device,
        ),
    )

    # spaCy + EntityRuler
    try:
        nlp_spacy = spacy.load("pt_core_news_lg")
    except Exception as e:
        logging.error("Modelo spaCy 'pt_core_news_lg' não encontrado. Instale-o e tente novamente.")
        raise e

    if "entity_ruler" not in nlp_spacy.pipe_names:
        ruler = cast(EntityRuler, nlp_spacy.add_pipe("entity_ruler", before="ner"))
    else:
        ruler = cast(EntityRuler, nlp_spacy.get_pipe("entity_ruler"))

    # Alias de dict separado do TypedDict para não conflitar com Pylance/mypy
    SpacyEntityPatternDict = Dict[str, Union[str, List[Dict[str, Any]]]]

    patterns: List[SpacyEntityPatternDict] = [
        {
            "label": "OPERACAO",
            "pattern": [
                {"LOWER": "operação"},
                {"LOWER": {"NOT_IN": ["policial"]}},
            ],
        }
    ]
    ruler.add_patterns(patterns)

    logging.info("[1/4] Modelos prontos.")
    return nlp_hf, nlp_spacy, ruler


# =========================
# NER + Normalização
# =========================

def extract_entities(text: Any, nlp_hf: TokenClassificationPipeline, nlp_spacy: spacy.Language) -> str:
    if pd.isna(text):
        return ""
    text_str = str(text)

    ner_results = nlp_hf(text_str)
    categories_of_interest = {"LOC", "PER", "EVENT", "DATE", "LAW", "OPERACAO"}
    entities: List[str] = []

    for result in ner_results:
        grp = result.get("entity_group", "")
        if grp in categories_of_interest:
            entities.append(f"{result['word']}:{grp}")

    doc_spacy = nlp_spacy(text_str)
    for ent in doc_spacy.ents:
        if ent.label_ == "OPERACAO":
            entities.append(f"{ent.text}:{ent.label_}")

    return ", ".join(sorted(set(entities)))


def _progress_iterable(iterable, desc: str):
    if _HAS_TQDM:
        return tqdm(iterable, desc=desc, unit="lin", dynamic_ncols=True, leave=False)
    return iterable


def process_videos_and_extract_entities(
    df: pd.DataFrame,
    year_tag: str,
    cache_dir: str,
    use_cache: bool,
    force: bool,
    sep: str,
    nlp_hf: TokenClassificationPipeline,
    nlp_spacy: spacy.Language,
) -> Tuple[pd.DataFrame, str]:
    """
    Extrai entidades nas colunas 'titulo' e 'descricao' (e 'transcribedText' se existir).
    Salva cache em disco para reutilização (etapa cara).
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"ner_results_{year_tag}.tsv")

    if use_cache and not force and os.path.exists(cache_path):
        print("[2/4] Extração NER: usando cache existente")
        logging.info(f"[NER] usando cache: {cache_path}")
        cached = pd.read_csv(cache_path, sep="\t")
        return cached, cache_path

    print("[2/4] Extração NER: em progresso…")
    logging.info("[NER] iniciando extração (sem cache)")

    if "titulo" not in df.columns or "descricao" not in df.columns:
        raise ValueError("As colunas 'titulo' e 'descricao' são obrigatórias para extração NER.")

    df_proc = df.copy()

    # Barra de progresso linha a linha (equivalente ao apply; apenas UX)
    titles = df_proc["titulo"].tolist()
    descrs = df_proc["descricao"].tolist()

    ner_tit: List[str] = []
    for t in _progress_iterable(titles, desc="NER títulos"):
        ner_tit.append(extract_entities(t, nlp_hf, nlp_spacy))
    df_proc["ner_titulo"] = ner_tit

    ner_desc: List[str] = []
    for d in _progress_iterable(descrs, desc="NER descrições"):
        ner_desc.append(extract_entities(d, nlp_hf, nlp_spacy))
    df_proc["ner_descricao"] = ner_desc

    if "transcribedText" in df_proc.columns:
        logging.info("[NER] aplicando também em 'transcribedText'")
        trans_list = df_proc["transcribedText"].fillna("").tolist()
        ner_trs: List[str] = []
        for t in _progress_iterable(trans_list, desc="NER transcrições"):
            ner_trs.append(extract_entities(t, nlp_hf, nlp_spacy))
        df_proc["ner_transcribedText"] = ner_trs
    else:
        logging.info("[NER] coluna 'transcribedText' não encontrada (pulando)")

    # salva cache (com todas as colunas + colunas ner_*)
    df_proc.to_csv(cache_path, index=False, sep="\t")
    logging.info(f"[NER] concluída; cache salvo em {cache_path}")
    return df_proc, cache_path


# ==========
# Fase 4: utilidades anti-ruído (apenas saneamento; não altera a decisão)
# ==========

_VALID_LABELS: Set[str] = {"loc", "per", "event", "date", "law", "operacao"}
_ENTITY_RE = re.compile(r"^([a-z0-9][a-z0-9\s\-]{2,}):([a-z]+)$")

def _is_valid_entity_token(token: str) -> bool:
    token = token.strip()
    if not token:
        return False
    m = _ENTITY_RE.match(token)
    if not m:
        return False
    left, label = m.groups()
    return label in _VALID_LABELS and left.strip() != ""

def _clean_entity_field(s: str) -> str:
    """
    Recebe string "e1, e2, e3" e retorna somente entidades válidas,
    deduplicadas e ordenadas (apenas saneamento; não altera decisão posterior).
    """
    if not isinstance(s, str) or not s:
        return ""
    parts = [p.strip() for p in s.split(",")]
    kept = {p for p in parts if _is_valid_entity_token(p)}
    if not kept:
        return ""
    return ", ".join(sorted(kept))


# =========================
# Normalização pós-cache
# =========================

def load_data(file_path: str) -> pd.DataFrame:
    """
    Carrega e normaliza as colunas NER que existirem
    ('ner_titulo', 'ner_descricao', e opcionalmente 'ner_transcribedText').
    """
    try:
        df = pd.read_csv(file_path, sep="\t", parse_dates=["data_postagem"])
        logging.info(f"[norm] carregado de {file_path}; colunas: {list(df.columns)}")

        ner_cols_to_process: List[str] = []
        if "ner_titulo" in df.columns:
            ner_cols_to_process.append("ner_titulo")
        if "ner_descricao" in df.columns:
            ner_cols_to_process.append("ner_descricao")
        if "ner_transcribedText" in df.columns:
            logging.info("[norm] incluindo 'ner_transcribedText'")
            ner_cols_to_process.append("ner_transcribedText")

        def remove_accents(input_str: Any) -> str:
            if pd.isna(input_str):
                return ""
            nfkd_form = unicodedata.normalize("NFD", str(input_str))
            return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

        for col in ner_cols_to_process:
            df[col] = df[col].fillna("")
            df[col] = df[col].str.lower()
            df[col] = df[col].apply(remove_accents)
            df[col] = df[col].str.replace(r"[^a-z0-9\s,:]", "", regex=True)
            df[col] = df[col].apply(_clean_entity_field)

        logging.info("[norm] concluída")
        return df
    except Exception as e:
        logging.error(f"[norm] erro ao carregar/normalizar: {e}")
        return pd.DataFrame()


# =========================
# Lógica de janelas e contagens (LÓGICA ORIGINAL)
# =========================

def get_time_window(df: pd.DataFrame, base_date: pd.Timestamp, days: int = 15) -> pd.DataFrame:
    start_date = base_date - datetime.timedelta(days=days)
    end_date = base_date + datetime.timedelta(days=days)
    return df[(df['data_postagem'] >= start_date) & (df['data_postagem'] <= end_date)]


def count_entities_in_window(df: pd.DataFrame) -> Dict[str, int]:
    """
    Conta a frequência de todas as entidades na janela de tempo. Inclui as
    entidades da transcrição se a coluna existir.
    """
    series_to_count = df['ner_titulo'].str.cat(df['ner_descricao'], sep=', ')
    if 'ner_transcribedText' in df.columns:
        series_to_count = series_to_count.str.cat(df['ner_transcribedText'].fillna(''), sep=', ')

    entity_counts: Dict[str, int] = {}
    for entities in series_to_count.str.split(', '):
        for entity in entities:
            entity = entity.strip()
            if entity:
                entity_counts[entity] = entity_counts.get(entity, 0) + 1
    return entity_counts


def assign_operations(df: pd.DataFrame, generic_entities: Set[str]) -> pd.DataFrame:
    operations: List[str] = []
    operation_ids: Dict[str, str] = {}
    operation_ners: Dict[str, str] = {}
    last_date_used: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    operation_counters: Dict[str, int] = {}
    unique_id: int = 1

    print("[3/4] Agrupando operações…")
    for index, row in df.iterrows():
        logging.info(f"--- [Linha {index}, Data: {row['data_postagem'].date()}] ---")

        window_df = get_time_window(df, row['data_postagem'])
        entity_counts = count_entities_in_window(window_df)

        current_entities = set(row['ner_titulo'].split(', ') + row['ner_descricao'].split(', '))
        if 'ner_transcribedText' in row and pd.notna(row['ner_transcribedText']):
            current_entities.update(row['ner_transcribedText'].split(', '))
        current_entities.discard('')

        sorted_entities = sorted(current_entities, key=lambda x: entity_counts.get(x, 0), reverse=True)
        sorted_entities = [ent for ent in sorted_entities if ent not in generic_entities]
        operation = 'Unique Operation'

        if sorted_entities and entity_counts.get(sorted_entities[0], 0) > 0:
            operation = sorted_entities[0]
            # Mantido o log original com "Operação Real" caso exista coluna 'operation'
            real_op = row['operation'] if 'operation' in row and pd.notna(row['operation']) else ''
            logging.info(f"Entidade candidata: '{operation}' (Contagem na janela: {entity_counts.get(operation, 0)}). Operação Real: {real_op}")

            if operation in last_date_used:
                logging.info(f"'{operation}' JÁ EXISTE como operação base. Verificando proximidade da data...")
                if row['data_postagem'] < last_date_used[operation][0] - datetime.timedelta(days=15) or \
                   row['data_postagem'] > last_date_used[operation][1] + datetime.timedelta(days=15):
                    logging.warning(f"Data distante da operação base '{operation}'. Iniciando lógica de sufixos.")

                    base_ner = operation
                    found_fit = False

                    suffixed_ops = sorted([op for op in last_date_used.keys() if op.startswith(f"{base_ner}-")])
                    logging.info(f"Sufixos existentes para '{base_ner}': {suffixed_ops}")

                    for op_name in suffixed_ops:
                        start_date, end_date = last_date_used[op_name]
                        if start_date - datetime.timedelta(days=15) <= row['data_postagem'] <= end_date + datetime.timedelta(days=15):
                            logging.info(f"DECISÃO: Encontrado encaixe! Reutilizando e estendendo a operação com sufixo '{op_name}'.")
                            operation = op_name
                            last_date_used[op_name] = (min(start_date, row['data_postagem']), max(end_date, row['data_postagem']))
                            found_fit = True
                            break

                    if not found_fit:
                        current_count = operation_counters.get(base_ner, 1)
                        new_count = current_count + 1
                        operation_counters[base_ner] = new_count

                        operation = f"{base_ner}-{new_count}"
                        logging.warning(f"DECISÃO: Nenhum encaixe encontrado. Criando novo sufixo -> '{operation}'.")

                        operation_ids[operation] = f"OP{unique_id}"
                        operation_ners[operation] = base_ner
                        last_date_used[operation] = (row['data_postagem'], row['data_postagem'])
                        unique_id += 1
                else:
                    logging.info(f"DECISÃO: Data próxima da operação base. Estendendo janela de '{operation}'.")
                    last_date_used[operation] = (
                        min(last_date_used[operation][0], row['data_postagem']),
                        max(last_date_used[operation][1], row['data_postagem']),
                    )
            else:
                logging.info(f"DECISÃO: É a PRIMEIRA VEZ para a operação '{operation}'. Iniciando contador e registro.")
                operation_counters[operation] = 1
                operation_ids[operation] = f"OP{unique_id}"
                operation_ners[operation] = sorted_entities[0]
                last_date_used[operation] = (row['data_postagem'], row['data_postagem'])
                unique_id += 1
        else:
            operation = f"OP{unique_id}"
            logging.warning(f"DECISÃO: Nenhuma entidade relevante. Criando ID genérico -> '{operation}'.")
            operation_ids[operation] = operation
            operation_ners[operation] = 'None' if not sorted_entities else ', '.join(sorted_entities)
            last_date_used[operation] = (row['data_postagem'], row['data_postagem'])
            unique_id += 1

        operations.append(operation)
        logging.info(f"-> Atribuído à linha {index}: '{operation}'\n")

    df['operation_ner'] = operations
    logging.info("Operações atribuídas com base nas entidades mais frequentes")
    return df


def get_generic_entities_unsupervised(df: pd.DataFrame, threshold: int = 5) -> Set[str]:
    logging.info(f"[gen] identificando entidades genéricas (meses distintos > {threshold})")
    ner_cols_to_process = ["ner_titulo", "ner_descricao"]
    if "ner_transcribedText" in df.columns:
        ner_cols_to_process.append("ner_transcribedText")

    df = df.copy()
    df["all_ners"] = df[ner_cols_to_process].fillna("").agg(", ".join, axis=1)

    ner_date_df = (
        df[["data_postagem", "all_ners"]]
        .assign(all_ners=df["all_ners"].str.split(", "))
        .explode("all_ners")
        .rename(columns={"all_ners": "entity"})
    )
    ner_date_df["entity"] = ner_date_df["entity"].str.strip()
    ner_date_df.dropna(subset=["entity"], inplace=True)
    ner_date_df = ner_date_df[ner_date_df["entity"] != ""]

    ner_date_df["year_month"] = ner_date_df["data_postagem"].dt.to_period("M")
    temporal_freq = ner_date_df.groupby("entity")["year_month"].nunique()
    generic_entities = set(temporal_freq[temporal_freq > threshold].index)

    logging.info(f"[gen] {len(generic_entities)} entidades genéricas encontradas")
    return generic_entities


# =========================
# Sumário final (console)
# =========================

def print_final_summary(final_df: pd.DataFrame, args: argparse.Namespace, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> None:
    print("\n[4/4] Sumário")
    total_videos = len(final_df)
    ops_series = final_df.get("operation_ner", pd.Series([], dtype=str)).fillna("")
    ops_distintas = ops_series.replace("", pd.NA).dropna().nunique()
    vazios = int((ops_series == "").sum())
    genericos = int(ops_series.str.match(r"^OP\d+$").sum())

    period_str = "n/d"
    if "data_postagem" in final_df.columns:
        dmin = pd.to_datetime(final_df["data_postagem"], errors="coerce").min()
        dmax = pd.to_datetime(final_df["data_postagem"], errors="coerce").max()
        if pd.notna(dmin) and pd.notna(dmax):
            delta = (dmax - dmin).days
            period_str = f"{dmin.date()} — {dmax.date()} ({delta} dias)"

    head_pct = max(1, min(99, args.summary_head_percent))
    head_n = max(1, int(total_videos * head_pct / 100))
    df_ord = final_df.copy()
    if "data_postagem" in df_ord.columns:
        df_ord = df_ord.sort_values("data_postagem", kind="stable")
    head_ops = df_ord.head(head_n)["operation_ner"].replace("", pd.NA).dropna().nunique()

    top_ops = (
        ops_series[ops_series != ""]
        .value_counts()
        .head(5)
        .to_dict()
    )

    elapsed = (end_ts - start_ts).total_seconds()

    print("────────────────────────────────────────")
    print(f"Vídeos processados           : {total_videos}")
    print(f"Operações únicas             : {ops_distintas}")
    print(f"Sem entidade (vazio)         : {vazios}")
    print(f"IDs genéricos (OP###)        : {genericos}")
    print(f"Período de datas             : {period_str}")
    print(f"Primeiros {head_pct}% dos dados: {head_ops} operações distintas")
    if top_ops:
        print("Top 5 operações              :")
        for k, v in top_ops.items():
            print(f"  - {k} -> {v}")
    print(f"Tempo total                  : {elapsed:.2f}s")
    print("────────────────────────────────────────\n")


# =========================
# Orquestração
# =========================

def main() -> int:
    args = parse_args()
    log_path = os.path.join(args.cache_dir, f"logs_{args.year_tag}.log")
    os.makedirs(args.cache_dir, exist_ok=True)
    APP, UX = setup_logging(args.log_level, log_path)

    # Cabeçalho amigável
    print_params_clean(args, log_path)

    # Check setup?
    if args.check_setup:
        print("[1/4] Checando ambiente e modelos…")
        ok = quick_validate_input(args.input, args.sep)
        if not ok:
            return 2
        try:
            init_models()
            print("✓ Ambiente OK. Modelos carregam corretamente.")
            return 0
        except Exception:
            print("✗ Falha na checagem de setup. Veja o arquivo de logs para detalhes.")
            return 3

    # Validação rápida do arquivo
    print("[1/4] Validando arquivo de entrada…")
    ok = quick_validate_input(args.input, args.sep)
    if not ok:
        print("✗ Entrada inválida (detalhes no log).")
        return 2

    # Carrega input original
    try:
        df_input = pd.read_csv(args.input, sep=args.sep)
    except Exception as e:
        logging.error(f"[io] erro ao ler entrada completa: {e}")
        print("✗ Falha ao ler o arquivo de entrada.")
        return 4

    # Converte data_postagem se existir
    if "data_postagem" in df_input.columns:
        try:
            df_input["data_postagem"] = pd.to_datetime(df_input["data_postagem"], errors="coerce")
        except Exception as e:
            logging.warning(f"[io] não foi possível converter 'data_postagem': {e}")

    # Inicializa modelos
    try:
        nlp_hf, nlp_spacy, _ = init_models()
    except Exception:
        print("✗ Falha ao inicializar modelos. Consulte o arquivo de logs.")
        return 5

    start_ts = pd.Timestamp.now()

    # Etapa cara: extração NER + cache
    df_ner, cache_path = process_videos_and_extract_entities(
        df=df_input,
        year_tag=args.year_tag,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        force=args.force,
        sep=args.sep,
        nlp_hf=nlp_hf,
        nlp_spacy=nlp_spacy,
    )

    # Normalização das colunas ner_* (lendo do cache para manter fluxo simples)
    df_norm = load_data(cache_path)
    if df_norm.empty:
        print("✗ Erro na normalização (detalhes no arquivo de logs).")
        return 6

    # >>> NÃO filtramos por 'operation' antes de agrupar (mantém OPx para todas as linhas)
    df_work = df_norm.copy()

    # Identifica entidades genéricas
    generic_entities = get_generic_entities_unsupervised(df_work, threshold=5)

    # Agrupamento / atribuição de operações em TODO o df_norm
    df_assigned = assign_operations(df_work, generic_entities)

    # Construção do resultado final (inclui ner_titulo, ner_descricao e all_ners)
    print("[4/4] Escrevendo resultados…")

    # Gera all_ners no df_norm (baseado nas colunas ner_* que existirem)
    ner_cols = []
    if "ner_titulo" in df_norm.columns: ner_cols.append("ner_titulo")
    if "ner_descricao" in df_norm.columns: ner_cols.append("ner_descricao")
    if "ner_transcribedText" in df_norm.columns: ner_cols.append("ner_transcribedText")

    def _join_ners(row: pd.Series) -> str:
        vals = [str(row[c]) if pd.notna(row[c]) else "" for c in ner_cols]
        return ", ".join(vals)

    if ner_cols:
        df_norm["all_ners"] = df_norm.apply(_join_ners, axis=1)
    else:
        df_norm["all_ners"] = ""

    # Monta saída final com:
    # - todas as colunas do arquivo de entrada
    # - + ner_titulo, ner_descricao, all_ners, operation_ner
    final_df = df_input.copy()

    # Injeta ner_titulo / ner_descricao (se existirem)
    if "ner_titulo" in df_norm.columns:
        final_df["ner_titulo"] = df_norm["ner_titulo"].reindex(final_df.index).fillna("")
    else:
        final_df["ner_titulo"] = ""
    if "ner_descricao" in df_norm.columns:
        final_df["ner_descricao"] = df_norm["ner_descricao"].reindex(final_df.index).fillna("")
    else:
        final_df["ner_descricao"] = ""

    # Injeta all_ners
    final_df["all_ners"] = df_norm["all_ners"].reindex(final_df.index).fillna("")

    # Injeta operation_ner (preenchida para TODAS as linhas, inclusive com OPx)
    if "operation_ner" in df_assigned.columns:
        final_df["operation_ner"] = df_assigned["operation_ner"].reindex(final_df.index).fillna("")
    else:
        final_df["operation_ner"] = ""

    # Garante diretório e salva
    Path(args.final_output).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.final_output, index=False, sep=args.sep)
    end_ts = pd.Timestamp.now()

    print(f"✓ Arquivo salvo em: {args.final_output}")
    print_final_summary(final_df, args, start_ts, end_ts)
    logging.info("[done] pipeline concluída com sucesso")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.error("Execução interrompida pelo usuário (Ctrl+C).")
        sys.exit(130)