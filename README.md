# CSC 4110 – Group G9
## HR Employee Attrition Prediction

### 1. Project Overview
This project uses machine learning to analyse employee information and identify patterns associated with staff attrition.

### 2. Explanation for HR Officers
The application analyses information such as job satisfaction, years at the company and working conditions to estimate whether an employee's profile resembles patterns found among employees who previously left. It is designed to support workforce analysis, not replace human judgement. Predictions should not be used to dismiss employees, deny promotions or reduce pay.

### 3. Project Structure
- `data/` – Project dataset
- `models/` – Trained machine learning model and defaults
- `notebooks/` – Project notebooks
- `app/` – Streamlit application and requirements
- `outputs/` – Evaluation results, charts and screenshots
- `report/` – Model card and reports
- `tests/` – Smoke tests

### 4. Running the Notebooks
1. Open Google Colab.
2. Open the required notebook from the `notebooks/` folder.
3. Mount Google Drive.
4. Confirm the dataset is available in the `data/` folder.
5. Run the notebook cells in order.

### 5. Running the Application Locally
1. Clone or download this GitHub repository.
2. Open a terminal in the project folder.
3. Install the dependencies:
   `pip install -r app/requirements.txt`
4. Start the application:
   `python -m streamlit run app/app.py`

### 6. Smoke Test
From the project root, run:
`python tests/smoke_test.py`

The test loads the trained model, predicts three test rows and reports whether the test passed.

### 7. Public Application Link
To be added after deployment on Streamlit Community Cloud.

### 8. Group Task Ownership

| Group Member | Task Owned | Contribution |
|---|---|---|
| Member 1 – Add name | Task 9.1 | Model packaging and Streamlit app |
| Member 2 – Add name | Task 9.2 | Responsible AI and model card |
| Member 3 – Add name | Task 9.3 | GitHub publishing and deployment |
| Other members – Add names | Review and testing | Review and testing contributions |

Replace the placeholders with the actual group members and their contributions.

### 9. Responsible Use
This model has limitations and must not be used as the sole basis for employment decisions. Do not use its predictions to dismiss staff, deny promotions or reduce pay. Individual risk scores should only be accessible to authorised personnel.
