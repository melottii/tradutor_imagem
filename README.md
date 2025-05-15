# Tradutor de Imagem

Este projeto é um sistema de processamento de imagens que realiza a detecção, extração, classificação e tradução de textos presentes em imagens. O sistema é capaz de identificar textos em inglês, japonês e chinês, traduzindo-os para português brasileiro e substituindo o texto original na imagem.

## Funcionalidades

1. **Detecção de Texto**
   - Utiliza o modelo doctr para detectar e extrair texto de imagens
   - Identifica a posição exata de cada texto na imagem
   - Agrupa textos próximos em balões de fala

2. **Classificação de Idioma**
   - Identifica o idioma do texto detectado
   - Suporta inglês, japonês e chinês
   - Utiliza o modelo XLM-RoBERTa para classificação

3. **Tradução**
   - Traduz o texto para português brasileiro
   - Utiliza a API do Google Translate
   - Implementa tratamento especial para nomes próprios e expressões

4. **Substituição de Texto**
   - Mascara o texto original com uma caixa
   - Insere o texto traduzido na posição original
   - Mantém a formatação e estilo do texto original

## Requisitos

```bash
python-doctr==0.7.0
torch==2.1.0+cpu
torchvision==0.16.0+cpu
opencv-python==4.8.1.78
numpy>=1.21.0
Pillow==10.0.0
transformers==4.35.0
googletrans==3.1.0a0
```

## Instalação

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITÓRIO]
cd tradutor_imagem
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Estrutura do Projeto

```
tradutor_imagem/
├── resources/             # Pasta para imagens de entrada
├── output/                # Pasta para resultados
├── main.py                # Script principal
├── text_treatment.py      # Classes de processamento de texto
└── requirements.txt       # Dependências do projeto
```

## Classes Principais

### TextDetector
- Responsável pela detecção e extração de texto
- Utiliza o modelo doctr para OCR
- Agrupa textos por proximidade espacial

### TextClassifier
- Classifica o idioma do texto detectado
- Utiliza o modelo XLM-RoBERTa
- Suporta múltiplos idiomas

### TextTranslator
- Realiza a tradução do texto
- Implementa tratamento especial para nomes próprios
- Utiliza a API do Google Translate

## Uso

1. Coloque a imagem a ser processada na pasta `resources/`

2. Execute o script principal:
```bash
python main.py
```

3. Os resultados serão salvos na pasta `output/`:
   - `resultados.json`: Textos detectados e suas coordenadas
   - `imagem_com_deteccoes.jpeg`: Imagem com os textos traduzidos

## Exemplo de Saída

```
Textos detectados por grupo:
Grupo 1: STAIRWAY PALACE
Grupo 2: THE CHALLENGER HAS ARRIVED!

Classificação de idioma por grupo:
Grupo 1:
  en: 99.99%

Grupo 2:
  en: 99.99%

Texto original e tradução:
Grupo 1:
Original: STAIRWAY PALACE
Tradução: PALÁCIO DA ESCADARIA

Grupo 2:
Original: THE CHALLENGER HAS ARRIVED!
Tradução: O DESAFIANTE CHEGOU!
```

## Limitações

1. A detecção de texto pode não ser precisa em imagens com:
   - Baixa resolução
   - Texto muito pequeno
   - Texto com fontes não convencionais

2. A tradução pode não ser perfeita para:
   - Expressões idiomáticas
   - Nomes próprios
   - Textos com contexto específico
