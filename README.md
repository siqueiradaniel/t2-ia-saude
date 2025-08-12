Aqui está um `README.md` claro e direto para o seu projeto com instruções de uso, download do dataset e execução do código:

---

## 📄 README.md

````markdown
# CBIS-DDSM Loader

Este projeto permite carregar e explorar imagens do dataset **CBIS-DDSM** (mamografias) usando um `torch.utils.data.Dataset` personalizado.

---

## 📥 Download do Dataset

1. Acesse o dataset no Kaggle:

   👉 https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

2. Baixe o dataset completo.

3. Crie a pasta `CBIS-DDSM` no diretório do projeto:

   ```bash
   mkdir CBIS-DDSM
````

4. Coloque a pasta `jpeg/` (que vem do dataset) dentro da pasta `CBIS-DDSM/`:

   ```
   CBIS-DDSM/
   └── jpeg/
   ```

5. Copie os arquivos `.csv` da pasta do dataset para `./csv/`:

   ```
   csv/
   ├── dicom_info.csv
   ├── mass_case_description_train_set.csv
   ├── calc_case_description_train_set.csv
   ├── mass_case_description_test_set.csv
   └── calc_case_description_test_set.csv
   ```

---

## ⚙️ Instalação

Recomenda-se usar um ambiente virtual.

1. Clone o repositório:

   ```bash
   git clone <url-do-repo>
   cd <nome-da-pasta>
   ```

2. Crie o ambiente e ative:

   ```bash
   python -m venv venv
   source venv/bin/activate  # ou venv\Scripts\activate no Windows
   ```

3. Instale os requisitos:

   ```bash
   pip install -r requirements.txt
   ```

---



## ▶️ Como Rodar

Execute o script principal no terminal:

```bash
python loader.py
```

> Isso irá carregar o primeiro paciente e salvar a primeira imagem como `output_image.jpg`.

---

## 🖼️ Saída Esperada

* A imagem `output_image.jpg` será salva no diretório atual.
* Informações como `Label`, `UID` e `Meta-data` serão exibidas no terminal.

---

## 📁 Estrutura esperada do projeto

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
├── script.py
├── requirements.txt
└── README.md
```

---

## 👤 Autores

Daniel Siqueira de Oliveira

```

---

```
