## 🗣️ Interjections and Interactions in the German Parliament: An Exploratory Study 

This repository contains the code for the report analyzing parliamentary dynamics using Natural Language Processing (NLP).

### 📂 Repository Structure
Due to storage limitations, the `data/` directory and `new_speeches_output.csv` contain only XML files and dataframes with meta-information for the last five debates. These XML files were sourced from [GermaParl](https://github.com/PolMine/GermaParlTEI) (Blätte, 2017).

The repository is designed to be flexible: you can replace the XML files in `data/` to analyze other debates from [GermaParl](https://github.com/PolMine/GermaParlTEI). To generate annotated dataframes for new debates, update the content in `data/` and run `preprocess.py`. This will save a new dataframe with meta-data (`new_speeches_output.csv`) for further analyses.

⚠ **Run time warning:** Processing a large number of debates can be time-consuming. If you are interested in the full `new_speeches_output.csv` for the 19th legislature period without running it yourself, please contact the author.

### 📝 Code Execution Order
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
   - **`party_mentions_preprocessing.ipynb`**: Prepares data for analysis.
   - **`party_mentions.ipynb`**: Generates plots and prepares further analyses.
   - **`classified_partymentions.ipynb`**: Analyzes speech acts and interjections when parties mention themselves or others; (parts of) speeches and interjections from the last 5 debates of the 19th legislature period were manually annotated based on [Burkhardt, 1993](#references) and [Reinig et al., 2024](#references)).

### 📧 Contact
For questions or issues, contact the author: **hannahsteinbach0312@gmail.com**.

### 📚 References
- Andreas Blätte. 2017. *GermaParl: Corpus of Plenary Protocols of the German Bundestag.* TEI files available at [GitHub](https://github.com/PolMine/GermaParlTEI).
- Armin Burkhardt. 1993. *Der Einfluss der Medien auf das parlamentarische Sprechen*, pp. 158-203. Max Niemeyer Verlag, Berlin, New York.
- Ines Reinig, Ines Rehbein, and Simone Paolo Ponzetto. 2024. *How to do politics with words: Investigating speech acts in parliamentary debates.* In *Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)*, pp. 8287-8300, Torino, Italia. ELRA and ICCL.

###  License
The data in this repository follows the **CLARIN PUB+BY+NC+SA** license, which means:
- **PUB**: The language resource can be distributed publicly.
- **BY (Attribution)**: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NC (NonCommercial)**: You may not use the material for commercial purposes.
- **SA (ShareAlike)**: If you remix, transform, or build upon the material, you must distribute your contributions under the same license.

The CLARIN licenses are modeled on the [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License](https://creativecommons.org/licenses/by-nc-sa/3.0/). For further details, please check CLARIN's documentation.

