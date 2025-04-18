# 🗣️ Interjections and Interactions in the German Parliament: An Exploratory Study 

This repository contains the code for the report analyzing parliamentary dynamics using Natural Language Processing (NLP).

## 📂 Repository Structure
Due to storage limitations, the `data/` directory and `new_speeches_output.csv` contain only XML files and dataframes with meta-information for the last five debates. These XML files were sourced from [GermaParl](https://github.com/PolMine/GermaParlTEI) (Blätte, 2017).

The repository is designed to be flexible: you can replace the XML files in `data/` to analyze other debates from [GermaParl](https://github.com/PolMine/GermaParlTEI). To generate annotated dataframes for new debates, update the content in `data/` and run `preprocess.py`. This will save a new dataframe with meta-data (`new_speeches_output.csv`) for further analyses.

Annotation details: 
| **Column**                 | **Type**     | **Description** |
|----------------------------|-------------|----------------|
| `Filename`                 | String      | Name of debate file |
| `Period`                   | String      | Legislature Period |
| `Date`                     | String      | Date in datetime format |
| `Item`                     | String      | Name of agenda item of paragraph (e.g., *Antrag der Abgeordneten...*) |
| `Speech #`                 | Integer     | Number of speech overall |
| `Paragraph #`              | Integer     | Number of paragraph within speech |
| `Speaker`                  | String      | Name of the speaker |
| `Role`                     | String      | Role of the speaker (e.g., MP, government) |
| `Gender`                   | String      | Gender of the speaker (as noted in Bundestag Stammdaten) |
| `Party`                    | String      | Party of the speaker |
| `Paragraph`                | String      | Content of the paragraph |
| `Interjection`             | Boolean     | `True` if the paragraph is an interjection |
| `Interjector`              | String      | Name of the interjecting person |
| `Interjector Gender`       | String      | Gender of the interjecting person |
| `Interjector Party`        | String      | Party of the interjecting person |
| `Verbal interjection`      | Boolean     | `True` if the interjection is verbal (e.g., *Zuruf, Widerspruch*) |
| `Nonverbal interjection`   | Boolean     | `True` if the interjection is nonverbal (e.g., *Beifall, Lachen*) |
| `Interjection type`        | String      | More specific interjection type (e.g., *Zuruf, Beifall, Lachen*) |


⚠ **Run time warning:** Processing a large number of debates can be time-consuming. If you are interested in the full `new_speeches_output.csv` for the 19th legislature period without running it yourself, please contact the author.

## 📝 Code Execution Order
Some notebooks and scripts build upon each other. The recommended execution order is:

1. **`preprocess.py`**
   - Reads in debates, transforms them into a dataframe (`new_speeches_output.csv`) with meta-information.
2. **`plot.py`**
   - Generates an overview of verbal and nonverbal interjections, including interactions between parties and interjectors.
3. **`gender.ipynb`**
   - Provides gender-specific analyses and visualizations.
4. **`interjections.ipynb`**
   - Analyzes interjections (e.g., average token length, bag-of-words).
5. **`speechact_classifier.ipynb`**
   - Prepares data for speech act classification using the BERT model from [Reinig et al., 2024](https://github.com/umanlp/speechact/tree/main).
   - Predictions for the last five debates of the 19th legislature period are stored in `predictions/`.
   - Note: This is only the code for preprocessing the files to obtain predictions from the BERT model, not the code for for running the model.
6. **`classifier_predictions.ipynb`**
   - Uses manually annotated data from `annotations/` (thus only applicable to the last five debates of the 19th legislature period, for which parts were manually annotated).
7. **`evaluation.ipynb`**
   - Compares the performance of the Speech Act Classifier against manual annotations (thus only applicable to the last five debates of the 19th legislature period, for which parts were manually annotated).
8. **Party Mentions Analysis (`party_mentions/`)**
   - **`party_mentions_preprocessing.ipynb`**: Prepares data for analysis, dependent on classifier_predictions.ipynb to create **`speeches.csv`**.
   - **`party_mentions.ipynb`**: Generates plots and prepares further analyses.
   - **`classified_partymentions.ipynb`**: Analyzes speech acts and interjections when parties mention themselves or others; (parts of) speeches and interjections from the last 5 debates of the 19th legislature period were manually annotated based on [Burkhardt, 1993](#references) and [Reinig et al., 2024](#references)).
   - 
     Note: Unfortunately, the file  **`classified_partymentions.ipynb`** does not render properly in the preview. To view it, please download the file and open it in an IDE or Jupyter Notebook.

Files 1-4 are flexible and can be run with any debate in the `data/` directory. However, files 5-8 require manual annotations and are therefore limited to the provided files. 

## 📸 Example Outputs
Here are a few example plots generated by the code:

- **Verbal and Nonverbal Interjections Overview:**
  ![Verbal and Nonverbal Interjections](plots/plot_2025-03-18-17-28-22_33.png)
  ![Nonverbal Interjections](plots/plot_2025-03-18-17-28-22_34.png)
  ![Verbal Interjections](plots/plot_2025-03-18-17-28-22_35.png)

- **Gender-Specific Analysis:**
  ![Gender Analysis](plots/gender.png)

- **Party Mentions Analysis:**
  ![Party Mentions](party_mentions/plots/sentence_types_used_other_party_mentioned.png)
  ![Party Mentions](party_mentions/plots/sentence_types_used_own_party_mentioned.png)
  ![Party Mentions](party_mentions/plots/interjection_types_received_other_party_mentioned.png)
  ![Party Mentions](party_mentions/plots/interjection_types_received_own_party_mentioned.png)


## 📧 Contact
For questions or issues, contact the author: **hannahsteinbach0312@gmail.com**.

## 📚 References
- Andreas Blätte. 2017. *GermaParl: Corpus of Plenary Protocols of the German Bundestag.* TEI files available at [GitHub](https://github.com/PolMine/GermaParlTEI).
- Armin Burkhardt. 1993. *Der Einfluss der Medien auf das parlamentarische Sprechen*, pp. 158-203. Max Niemeyer Verlag, Berlin, New York.
- Ines Reinig, Ines Rehbein, and Simone Paolo Ponzetto. 2024. *How to do politics with words: Investigating speech acts in parliamentary debates.* In *Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)*, pp. 8287-8300, Torino, Italia. ELRA and ICCL.

##  License
The corpus follows a CLARIN PUB+BY+NC+SA license. Code is released under the MIT License.

### Requirements
To run the code, please install the necessary dependencies listed in the environment.yml file. You can set up the environment by running:
`conda env create -f environment.yml`

