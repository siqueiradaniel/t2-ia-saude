# Pipeline de Machine Learning para Classificação de Imagens de Mamografia

Este projeto implementa um pipeline de Machine Learning metodologicamente rigoroso e replicável para a predição de patologias em imagens de mamografia do dataset **CBIS-DDSM**.  
O pipeline foi desenhado para tratar explicitamente desafios como o desbalanceamento de classes e o vazamento de dados (data leakage), buscando resultados realistas e confiáveis.

## Instalação

1.  Clone o repositório:

    ```bash
    git clone https://github.com/siqueiradaniel/t2-ia-saude.git
    cd t2-ia-saude
    ```

2.  (Recomendado) Crie e ative um ambiente virtual:

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate     # Windows
    ```

3.  Instale as dependências:

    ```bash
    pip install -r requirements.txt
    ```

## Download e Configuração do Dataset

1.  Baixe o dataset do Kaggle:
    [https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)


2.  Crie a estrutura de pastas esperada dentro do diretório do projeto:

    ```bash
    mkdir -p data
    ```

3.  Coloque as pastas `jpeg/` e `csv/` do dataset dentro de `data/`.

## Execução

O script principal `main.py` executa o pipeline completo.

**Modo de Desenvolvimento:**  
Para rodar rapidamente com um subconjunto dos dados, defina a flag `ISDEVELOPING = True` em `main.py`.

**Modo de Produção (Avaliação Final):**  
Para a avaliação completa, defina a flag `ISDEVELOPING = False`.

Execute o script:

```bash
python main.py
```

## Estrutura Esperada

```
.
├── data/
│   ├── jpeg/
│   └── csv/
│       ├── dicom_info.csv
│       ├── mass_case_description_train_set.csv
│       ├── calc_case_description_train_set.csv
│       ├── mass_case_description_test_set.csv
│       └── calc_case_description_test_set.csv
├── datasets/
│   └── dataloader.py
├── models/
│   ├── my_cnn.py
│   └── pretrained.py
├── training/
│   ├── train.py
│   ├── validate.py
│   └── test.py
├── utils/
│   └── augmentation.py
├── main.py
├── requirements.txt
└── README.md
```

## Autores
```
Arthur Roberto Barboza Maciel
Daniel Maximo Gramlich
Daniel Siqueira de Oliveira
```

