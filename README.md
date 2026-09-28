# 🫀 MVP Machine Learning — Doenças Cardíacas

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![PUC--Rio](https://img.shields.io/badge/PUC--Rio-Pós--Graduação-red)](https://www.puc-rio.br/)

## 📌 Sobre o Projeto

Este projeto foi desenvolvido como parte da **Sprint de Machine Learning da Pós-Graduação em Ciência de Dados & Analytics da PUC-Rio**.

O objetivo foi desenvolver um **MVP de Machine Learning para classificação binária**, utilizando dados clínicos do **Heart Disease Dataset**, disponibilizado pelo **UCI Machine Learning Repository**.

O modelo busca classificar pacientes em duas categorias:

* `0` → Ausência de doença cardíaca
* `1` → Presença de doença cardíaca

O projeto percorre as principais etapas de um pipeline de Machine Learning:

**Dados → Análise → Pré-processamento → Treinamento → Avaliação → Otimização → Modelo Final**

> ⚠️ **Aviso:** este projeto possui finalidade exclusivamente acadêmica e de portfólio. O modelo não é clinicamente validado e não deve ser utilizado para diagnóstico, tratamento ou tomada de decisão médica.

---

# 🎯 Objetivo

Desenvolver e avaliar modelos de **Machine Learning supervisionado** capazes de identificar padrões associados à presença ou ausência de doença cardíaca a partir de características clínicas dos pacientes.

A métrica principal definida para o projeto foi o **F1-Score**, buscando equilibrar Precision e Recall.

Também foram analisadas:

* Accuracy;
* Precision;
* Recall;
* F1-Score;
* ROC-AUC;
* Matriz de Confusão.

---

# 📊 Dataset

O projeto utiliza o **Heart Disease Dataset**, disponibilizado pelo **UCI Machine Learning Repository**.

### Características do dataset

| Característica       |                   Valor |
| -------------------- | ----------------------: |
| Registros            |                 **303** |
| Variáveis preditoras |                  **13** |
| Variável-alvo        |                `target` |
| Tipo de problema     |   Classificação binária |
| Treinamento          | **242 registros (80%)** |
| Teste                |  **61 registros (20%)** |
| Seed                 |                    `42` |

O notebook confirma que o dataset possui **303 registros e 14 colunas**, sendo 13 utilizadas como features e uma como variável-alvo.

---

# 🧬 Variáveis Utilizadas

Foram utilizadas 13 variáveis preditoras:

```text
age
sex
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
```

### Descrição

| Variável   | Descrição                                 |
| ---------- | ----------------------------------------- |
| `age`      | Idade                                     |
| `sex`      | Sexo                                      |
| `cp`       | Tipo de dor no peito                      |
| `trestbps` | Pressão arterial em repouso               |
| `chol`     | Colesterol sérico                         |
| `fbs`      | Glicemia de jejum                         |
| `restecg`  | Resultado do eletrocardiograma em repouso |
| `thalach`  | Frequência cardíaca máxima atingida       |
| `exang`    | Angina induzida por exercício             |
| `oldpeak`  | Depressão do segmento ST                  |
| `slope`    | Inclinação do segmento ST                 |
| `ca`       | Número de vasos principais                |
| `thal`     | Resultado relacionado à talassemia        |

---

# 🔎 Análise Exploratória

A primeira etapa do projeto foi dedicada à compreensão da estrutura dos dados.

Foram realizadas análises envolvendo:

* estrutura do DataFrame;
* tipos de dados;
* estatísticas descritivas;
* distribuição das variáveis;
* distribuição da variável-alvo;
* análise de valores ausentes;
* visualizações exploratórias.

O dataset utilizado no notebook apresenta **303 observações e 14 colunas**.

---

# ⚙️ Pré-processamento

Foi construído um pipeline utilizando recursos do **Scikit-learn**.

### Tratamento aplicado

```text
Dados
  │
  ├── Variáveis numéricas
  │       │
  │       ├── Imputação pela mediana
  │       │
  │       └── StandardScaler
  │
  └── Modelo de Machine Learning
```

O pipeline utilizou:

* `SimpleImputer(strategy="median")`
* `StandardScaler`
* `ColumnTransformer`
* `Pipeline`

As 13 features foram identificadas pelo notebook como variáveis numéricas.

O uso de `Pipeline` permite manter o pré-processamento integrado ao treinamento dos modelos.

---

# 🧪 Divisão dos Dados

Foi utilizado o método **Holdout**, com:

```text
80% → Treinamento
20% → Teste
```

Resultado:

```text
Treinamento: 242 registros
Teste:        61 registros
```

Foi utilizada **estratificação da variável-alvo**, mantendo a distribuição das classes entre os conjuntos de treinamento e teste.

A divisão foi realizada com:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

# 🤖 Modelos Avaliados

Foram utilizados três níveis de comparação:

### 1. Baseline

**DummyClassifier**

Estratégia:

```text
most_frequent
```

O objetivo foi estabelecer uma referência mínima de desempenho.

### 2. Regressão Logística

Modelo utilizado como uma abordagem linear para classificação binária.

Características:

* simples;
* rápido;
* baixo custo computacional;
* boa interpretabilidade.

### 3. Random Forest

Modelo baseado em múltiplas árvores de decisão.

Foi escolhido por sua capacidade de capturar relações não lineares entre as variáveis.

O notebook utiliza explicitamente **DummyClassifier, LogisticRegression e RandomForestClassifier** como baseline e modelos candidatos.

---

# 📈 Resultados dos Modelos

Os resultados obtidos no conjunto de teste foram:

| Modelo              |   Accuracy | F1-Score Weighted |    ROC-AUC | Tempo de Treinamento |
| ------------------- | ---------: | ----------------: | ---------: | -------------------: |
| DummyClassifier     |     54,10% |            0,3798 |     0,5000 |              0,010 s |
| Logistic Regression |     86,89% |            0,8690 |     0,9513 |              0,009 s |
| Random Forest       | **88,52%** |        **0,8852** | **0,9513** |              0,208 s |

Os valores acima correspondem aos resultados efetivamente registrados no notebook.

### Resultado observado

O **Random Forest inicial** apresentou:

```text
Accuracy       = 88,52%
F1-Score       = 0,8852
ROC-AUC        = 0,9513
```

A **Regressão Logística** apresentou:

```text
Accuracy       = 86,89%
F1-Score       = 0,8690
ROC-AUC        = 0,9513
```

Ambos os modelos superaram significativamente o baseline.

---

# 🔧 Otimização de Hiperparâmetros

Após a avaliação inicial, foi realizada uma etapa de otimização do **Random Forest** utilizando:

```text
RandomizedSearchCV
```

### Configuração

* 5 folds;
* validação estratificada;
* 5 combinações testadas;
* 25 ajustes realizados no total;
* métrica de otimização: `f1_weighted`;
* `random_state=42`.

O melhor F1-Score médio obtido durante a validação cruzada foi:

```text
F1-Score = 0,8122
```

Os melhores hiperparâmetros encontrados foram:

```python
{
    'model__max_depth': 16,
    'model__min_samples_split': 4,
    'model__n_estimators': 121
}
```

Esses valores estão registrados no resultado do `RandomizedSearchCV` executado no notebook.

---

# 🏆 Modelo Final

Após a otimização, o modelo selecionado foi:

## 🌲 Random Forest Otimizado

```text
RandomForest_otimizado
```

### Hiperparâmetros

| Parâmetro           |   Valor |
| ------------------- | ------: |
| `n_estimators`      | **121** |
| `max_depth`         |  **16** |
| `min_samples_split` |   **4** |
| `random_state`      |  **42** |

---

# 📊 Resultado Final no Conjunto de Teste

O modelo otimizado foi avaliado sobre **61 registros que não participaram do treinamento**.

### Desempenho

| Métrica                  | Resultado |
| ------------------------ | --------: |
| **Accuracy**             |   **92%** |
| **Precision — Classe 0** |  **0,97** |
| **Recall — Classe 0**    |  **0,88** |
| **F1-Score — Classe 0**  |  **0,92** |
| **Precision — Classe 1** |  **0,87** |
| **Recall — Classe 1**    |  **0,96** |
| **F1-Score — Classe 1**  |  **0,92** |
| **F1-Score Macro**       |  **0,92** |
| **F1-Score Weighted**    |  **0,92** |

O relatório de classificação registrado no notebook mostra **92% de acurácia e F1-score de 0,92 para ambas as classes**, com 33 observações da classe 0 e 28 da classe 1 no conjunto de teste.

---

# 🎯 Matriz de Confusão

A avaliação final também incluiu a geração da **Matriz de Confusão** utilizando:

```python
ConfusionMatrixDisplay.from_estimator(
    final_model,
    X_test,
    y_test
)
```

A matriz foi gerada para analisar os acertos e erros do modelo final nas duas classes.

---

# 📌 Interpretação dos Resultados

O experimento mostrou uma evolução clara entre o baseline e os modelos de Machine Learning.

```text
Baseline
   │
   └── Accuracy: 54,10%
             ↓
Logistic Regression
   │
   └── Accuracy: 86,89%
             ↓
Random Forest
   │
   └── Accuracy: 88,52%
             ↓
Random Forest Otimizado
   │
   └── Accuracy: 92%
```

O baseline apresentou desempenho de **54,10%**, enquanto os modelos supervisionados apresentaram desempenho consideravelmente superior no mesmo conjunto de teste.

No modelo final, o **Recall da classe 1 foi 0,96**, enquanto a Precision da classe 1 foi 0,87. O F1-score ficou em **0,92 para ambas as classes**.

Esses resultados indicam que, **dentro das condições específicas desse experimento e desse conjunto de dados**, o modelo conseguiu identificar padrões relevantes para a classificação.

---

# 🧠 Pipeline do Projeto

```text
                ┌──────────────────────┐
                │   Heart Disease      │
                │       Dataset        │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Análise Exploratória │
                │        (EDA)          │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   Pré-processamento  │
                │ Imputação + Scaling  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Train/Test Split     │
                │      80% / 20%       │
                └──────────┬───────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
   ┌─────────────────┐         ┌─────────────────┐
   │ Logistic        │         │ Random Forest   │
   │ Regression      │         │                 │
   └────────┬────────┘         └────────┬────────┘
            │                           │
            └─────────────┬─────────────┘
                          ▼
                ┌──────────────────────┐
                │ Avaliação dos        │
                │ modelos              │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ RandomizedSearchCV   │
                │ 5-Fold Stratified CV │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Random Forest        │
                │ Otimizado             │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Modelo Final         │
                │ Accuracy: 92%        │
                │ F1-Score: 0,92       │
                └──────────────────────┘
```

---

# 🛠️ Tecnologias

### Linguagem

* 🐍 Python 3.12

### Bibliotecas

* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* SciPy

### Ambiente

* Jupyter Notebook
* Google Colab
* GitHub

O notebook foi executado com **Python 3.12.13** e `random_state=42` para reprodutibilidade.

---

# 📁 Estrutura do Repositório

```text
uci-heart-disease-dataset/
│
├── MVP_ML_Doencas_Cardiacas.ipynb
│
├── heart-disease.csv
│
└── README.md
```

### `MVP_ML_Doencas_Cardiacas.ipynb`

Notebook contendo o desenvolvimento completo do MVP:

* definição do problema;
* análise exploratória;
* preparação dos dados;
* construção do pipeline;
* treinamento;
* comparação de modelos;
* otimização de hiperparâmetros;
* avaliação final;
* matriz de confusão;
* conclusões.

### `heart-disease.csv`

Dataset utilizado para o desenvolvimento do projeto.

---

# ▶️ Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/vinicius-datatech/uci-heart-disease-dataset.git
```

### 2. Acesse a pasta

```bash
cd uci-heart-disease-dataset
```

### 3. Instale as dependências

```bash
pip install pandas numpy matplotlib scikit-learn scipy jupyter
```

### 4. Execute o notebook

```bash
jupyter notebook
```

Depois, abra:

```text
MVP_ML_Doencas_Cardiacas.ipynb
```

---

# 🎓 Contexto Acadêmico

Este MVP foi desenvolvido durante a **Pós-Graduação em Ciência de Dados & Analytics da PUC-Rio**, como parte da **Sprint de Machine Learning**.

O projeto permitiu aplicar, de forma prática, conceitos relacionados a:

* Machine Learning supervisionado;
* classificação binária;
* análise exploratória de dados;
* preparação de dados;
* pipelines;
* validação cruzada;
* seleção de modelos;
* métricas de classificação;
* otimização de hiperparâmetros;
* Random Forest;
* avaliação de modelos.

---

# 📚 Principais Aprendizados

O desenvolvimento deste projeto proporcionou experiência prática em todo o fluxo de construção de um MVP de Machine Learning:

### 🔹 Data Preparation

Tratamento e preparação de dados tabulares para utilização em modelos preditivos.

### 🔹 Model Selection

Comparação entre baseline, Regressão Logística e Random Forest.

### 🔹 Model Evaluation

Avaliação por Accuracy, Precision, Recall, F1-Score e ROC-AUC.

### 🔹 Hyperparameter Optimization

Aplicação do `RandomizedSearchCV` para buscar uma configuração otimizada do Random Forest.

### 🔹 Reproducibility

Utilização de `random_state=42` e pipelines para tornar o experimento mais reprodutível.

### 🔹 Model Interpretation

Análise do desempenho por classe utilizando relatório de classificação e matriz de confusão.

---

# ⚠️ Limitações

Apesar dos resultados obtidos, algumas limitações devem ser consideradas:

* O dataset possui apenas **303 registros**.
* O conjunto de teste possui somente **61 observações**.
* Os resultados são específicos para a divisão e configuração utilizadas neste experimento.
* O modelo não passou por validação externa em uma base independente.
* O projeto possui finalidade acadêmica e não constitui uma solução clínica.
* Resultados de Machine Learning não devem ser interpretados como diagnóstico médico.

O próprio notebook destaca o tamanho reduzido do dataset e recomenda, como evolução, testes com bases maiores e validação externa.

---

# 🚀 Próximos Passos

Como evolução futura do projeto, podem ser exploradas as seguintes possibilidades:

* [ ] Aumentar a quantidade de dados utilizados;
* [ ] Testar novos algoritmos de classificação;
* [ ] Ampliar a busca de hiperparâmetros;
* [ ] Aplicar técnicas de validação mais robustas;
* [ ] Avaliar diferentes estratégias de tratamento de dados;
* [ ] Explorar Feature Importance;
* [ ] Implementar explicabilidade com SHAP;
* [ ] Criar uma API para servir o modelo;
* [ ] Desenvolver uma interface para demonstração;
* [ ] Containerizar a solução com Docker;
* [ ] Implementar um pipeline de MLOps;
* [ ] Realizar validação externa com outro conjunto de dados.

---

# 🔗 Repositório

**GitHub:**
https://github.com/vinicius-datatech/uci-heart-disease-dataset

---

# 📚 Fonte dos Dados

**UCI Machine Learning Repository — Heart Disease Dataset**

O conjunto de dados utilizado neste projeto é disponibilizado pelo UCI Machine Learning Repository.

Fonte:

https://archive.ics.uci.edu/dataset/45/heart+disease

---

# 👨‍💻 Autor

**Vinícius Araújo Moraes da Silva**

**Analista de Projetos e Dados | Data Science | Machine Learning | Power BI | SQL | Python**

🎓 MBA em Gestão de Projetos e Metodologias Ágeis — PUCRS
🎓 Pós-Graduação em Ciência de Dados & Analytics — PUC-Rio

---

⭐ **Projeto desenvolvido para fins acadêmicos e de portfólio profissional em Data Science e Machine Learning.**
