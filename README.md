# CBIS-DDSM Loader

Este projeto fornece uma interface para carregar e explorar o dataset **CBIS-DDSM** (mamografias) com um `torch.utils.data.Dataset` personalizado.

## Download do Dataset

1. Baixe o dataset no Kaggle:
   [https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)


2. Crie a pasta `CBIS-DDSM` no diretório do projeto:

   ```bash
   mkdir CBIS-DDSM
   ```

3. Coloque a pasta `jpeg/` do dataset dentro da pasta `CBIS-DDSM/`:

   ```
   CBIS-DDSM/
   └── jpeg/
   ```

## Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/siqueiradaniel/t2-ia-saude.git
   cd t2-ia-saude
   ```

2. (Recomendado) Crie e ative um ambiente virtual:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

## Execução

Execute o script principal:

```bash
python loader.py
```

Esse script irá carregar o primeiro paciente do dataset e salvar a primeira imagem como `output_image.jpg`.

A saída no terminal incluirá informações como:

* `Label`
* `UID`
* Metadados associados


## Estrutura Esperada

```
.
├── CBIS-DDSM/
│   └── jpeg/
├── csv/
│   ├── dicom_info.csv
│   ├── mass_case_description_train_set.csv
│   ├── calc_case_description_train_set.csv
│   ├── mass_case_description_test_set.csv
│   └── calc_case_description_test_set.csv
├── loader.py
├── requirements.txt
└── README.md
```

## Autores
```
Arthur
Daniel 
Daniel
```

