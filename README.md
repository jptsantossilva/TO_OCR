# Flexible Packaging OCR

MVP para ler texto à volta de códigos de barras em embalagens flexíveis, com dois modos:

- `Batch Upload`: processa um conjunto de imagens e devolve os resultados por ficheiro.
- `Image Library`: guarda imagens numa pasta local, permite ajustar filtros em tempo real, guardar settings e correr OCR batch com esses filtros.
- `Webcam Realtime`: analisa frames da webcam e faz overlay do texto lido.
- `ocr`: modo único atual, focado em OCR geral na imagem completa.
- `Comparar todos os backends`: corre `tesseract`, `easyocr`, `paddleocr` e `deepseek_ocr` na mesma imagem.
- `Mostrar filtros de diagnóstico`: mostra a ROI e as variantes filtradas aplicadas antes da inferência OCR.
- O diagnóstico de filtros também corre OCR por variante e destaca automaticamente o filtro com maior similaridade.

## Arquitetura

O fluxo implementado é:

1. Tentar ler o código de barras com `pyzbar`.
2. Se a leitura falhar, estimar a região do código com OpenCV.
3. Aplicar pré-processamento para OCR.
4. Correr OCR com `tesseract`, `easyocr`, `paddleocr` ou `deepseek_ocr`.
5. Validar o texto com tokens obrigatórios, expressão regular e string esperada.

## Instalação

Criar e ativar um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Executar a aplicação:

```bash
streamlit run app.py
```

Dependências de sistema no Ubuntu:

```bash
sudo apt update
sudo apt install tesseract-ocr libzbar0
```

Dependências opcionais para `deepseek_ocr`:

```bash
python -m pip install -r requirements-deepseek.txt
```

Se o ambiente já tiver `torchvision` de outra versão, remove o conflito antes:

```bash
python -m pip uninstall -y torchvision
python -m pip install -r requirements-deepseek.txt
```

Dependências opcionais para `paddleocr`:

```bash
python -m pip install -r requirements-paddle.txt
```

## Notas práticas

- `tesseract` tende a ser a opção mais estável para CPU em imagens de embalagem.
- `easyocr` continua disponível para comparação.
- `paddleocr` foi integrado como backend opcional para OCR em imagem.
- `deepseek_ocr` foi integrado como backend opcional/experimental para comparação de resultados.
- O backend `deepseek_ocr` tem seletor de `device`; por defeito usa `cpu` para evitar erro de VRAM.
- Se aparecer `cannot import name 'LlamaFlashAttention2'`, o problema é incompatibilidade de `transformers`, não falta de download do modelo.
- As versões testadas para este backend são `torch==2.6.0`, `transformers==4.46.3` e `tokenizers==0.20.3`.
- Se surgir conflito com `torchvision`, usa `torchvision==0.21.0` com `torch==2.6.0`.
- Em CPU, a app usa por defeito a revisão `refs/pr/6` do modelo para evitar chamadas forçadas a CUDA no `infer()`.
- Pode forçar outra revisão com `DEEPSEEK_OCR_REVISION`, por exemplo: `export DEEPSEEK_OCR_REVISION=main`.
- No backend `deepseek_ocr`, a UI mostra passos de execução e há timeout de 300s para evitar spinner indefinido.
- A primeira inferência com `easyocr` pode demorar vários segundos porque o modelo é carregado em memória; as seguintes devem ser mais rápidas.
- `pytesseract` requer o binário `tesseract-ocr` instalado no sistema.
- `pyzbar` pode exigir a biblioteca nativa `zbar` no sistema operativo.
- `deepseek_ocr` não entra nas dependências pesadas padrão do projeto, mas o ambiente Python precisa de `matplotlib`, `torch`, `transformers` e dos pesos grandes do modelo.
- O modo webcam usa amostragem de frames para não bloquear CPU.
- O modo de verificação por `Texto esperado` ajuda a estabilizar resultados em cenários industriais.
- Para produção, o próximo passo natural é recolher imagens reais do vosso packaging e ajustar:
  - pré-processamento por condição de luz e foco
  - pré-processamento por tipo de impressão
  - regras de validação por SKU

## Estrutura

- `app.py`: interface Streamlit.
- `ocr_pipeline.py`: pré-processamento, OCR e validação.
- `requirements.txt`: dependências Python.
- `requirements-paddle.txt`: dependências opcionais para backend `paddleocr`.
- `image_library/`: pasta criada pela app para guardar imagens e settings de filtros.

Filtros disponíveis no editor da biblioteca:

- `Exposure`
- `Saturation`
- `Contrast`
- `Brightness`
- `Sharpness`
- `Black/White threshold`
- `Grayscale`
