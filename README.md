# Identificação de Eventos de Violência Urbana

Este projeto tem como objetivo a identificação e agrupamento de eventos de violência urbana a partir de metadados de vídeos (títulos, descrições e transcrições), utilizando técnicas de Reconhecimento de Entidades Nomeadas (NER) e heurísticas temporais.

A solução foi desenvolvida como parte da pesquisa apresentada no artigo **"Uso de Características Temporais e Semânticas para Detectar Eventos em Vídeos de Violência Urbana"**, publicado no **WebMedia 2025**.

## 📖 Contexto e Motivação

A detecção de eventos em vídeos de violência urbana é um desafio complexo devido à natureza não estruturada dos dados e à necessidade de correlacionar informações espaciais, temporais e semânticas. Este repositório implementa um pipeline que combina modelos de Processamento de Linguagem Natural (PLN) de última geração para extrair entidades relevantes (como locais, datas, leis e tipos de operação) e agrupa vídeos relacionados a um mesmo evento real através de janelas temporais e similaridade semântica.

## 🎯 Objetivo do Código

O componente principal (`ner_cluster_pipeline.py`) executa as seguintes etapas:

1.  **Extração NER**: Utiliza modelos do Hugging Face (`wikineural-multilingual-ner`) e spaCy (`pt_core_news_lg`) para identificar entidades de interesse.
2.  **Normalização**: Saneamento de texto (remoção de acentos, pontuação e ruídos) para garantir consistência.
3.  **Agrupamento (Clustering)**: Aplica uma lógica de janelas temporais (±15 dias) para atribuir rótulos de "operação" a vídeos que compartilham entidades frequentes em um mesmo período.
4.  **Eficiência**: Implementação de sistema de *cache* para etapas de processamento custosas e suporte a aceleração por GPU (CUDA).

## 🚀 Como Iniciar

### Pré-requisitos
- Python 3.10 ou superior
- (Opcional) GPU NVIDIA com suporte a CUDA para maior performance

### Instalação

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/LABPAAD/urban_events_identification.git
   cd urban_events_identification
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r ner_cluster/temporal_heuristic/requirements.txt
   ```

4. **Baixe o modelo do spaCy:**
   ```bash
   python -m spacy download pt_core_news_lg
   ```

### Execução

Para rodar o pipeline principal:

```bash
python ner_cluster/temporal_heuristic/ner_cluster_pipeline.py \
  --input "videos_operations_combined.csv" \
  --year-tag "2025" \
  --final-output "resultado_final.tsv" \
  --use-cache
```

## 📄 Publicação e Citação

Se este trabalho for útil para sua pesquisa, por favor cite nosso artigo:

> ROCHA, Saul Sousa da et al. **Uso de Características Temporais e Semânticas para Detectar Eventos em Vídeos de Violência Urbana.** In: BRAZILIAN SYMPOSIUM ON MULTIMEDIA AND THE WEB (WEBMEDIA), 31., 2025, Rio de Janeiro. Anais [...]. Porto Alegre: Sociedade Brasileira de Computação, 2025. p. 464-472.

**Link para publicação:** [https://doi.org/10.5753/webmedia.2025.16150](https://doi.org/10.5753/webmedia.2025.16150)
**DOI:** `10.5753/webmedia.2025.16150`

## ⚖️ Licença

Este projeto está sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
