# Midterm Project — ITAI 1371

This folder contains the midterm project for ITAI 1371: an end-to-end machine learning investigation and data storytelling exercise.

## Files

- `MT_MelanieVillela_ITAI1371.ipynb` — Midterm notebook
- Dataset used for the project: the [Titanic Survival dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) (passenger survival classification)

## Dataset Columns

| Column | Description |
| :--- | :--- |
| `Survived` | **Label.** Survival — `0` did not survive, `1` survived. |
| `Pclass` | Ticket class — `1` first, `2` second, `3` third |
| `Name` | Passenger name |
| `Sex` | `male`, `female` |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number |
| `Embarked` | Port of embarkation — `C` Cherbourg, `Q` Queenstown, `S` Southampton |

## Project Structure

The notebook walks through a complete ML workflow:

1. **Data Loading** — Load the Titanic survival dataset into a DataFrame.
2. **Exploratory Data Analysis** — Create and interpret at least two visualizations of feature/target relationships.
3. **Data Preparation** — Handle missing values and encode categorical features, with written justification.
4. **Modeling** — Train a `LogisticRegression` baseline plus a second classification model of choice.
5. **Evaluation** — Compare accuracy, and interpret the `classification_report` and `confusion_matrix`.
6. **Conclusion** — Summarize the data story, key findings, and next steps.

## How to use

1. Open `MT_MelanieVillela_ITAI1371.ipynb` in VS Code or Jupyter Notebook.
2. Run the notebook cells in order (`Kernel` > `Restart & Run All` before submitting).
3. Complete each code and markdown cell per the in-notebook instructions.

## Notes

- Dataset loads from a URL at runtime, so an internet connection is required to run the notebook end-to-end.
- Save your changes to the notebook as you work.
- Follow the academic integrity policy stated in the notebook: AI assistance is allowed for code/syntax help, but analysis, interpretation, and reflective answers must be your own.
