# Model Card – Group G9

## 1. Model Purpose
This model analyses historical employee data to estimate patterns associated with employee attrition. It is intended for research, learning and responsible HR workforce analysis, not automated employment decisions.

## 2. Intended Users
The intended users are authorised HR analysts, data scientists and researchers. Individual risk scores must only be accessible to authorised HR personnel.

## 3. Data Source and Licence
Dataset: IBM HR Analytics Employee Attrition Dataset (WA_Fn-UseC_-HR-Employee-Attrition.csv).

The dataset was obtained from the project data files. Its original source and licence should be verified before redistribution or commercial use.

## 4. Features Used
The model uses 30 features, including Age, Gender, MaritalStatus, JobRole, MonthlyIncome, JobSatisfaction and YearsAtCompany.

EmployeeNumber was excluded because it is an employee identifier. EmployeeCount, Over18 and StandardHours were excluded because they are constant columns.

Gender, MaritalStatus and Age require careful ethical and legal review before any production use.

## 5. Performance on Test Data
The model was evaluated on a stratified hold-out test set containing 294 employees.

- Accuracy: 86.05%
- Precision: 71.43%
- Recall: 21.28%
- F1 Score: 32.79%

The test set contained 47 employees who actually left. The model correctly identified 10 and missed 37.

The low recall indicates that the model misses many actual attrition cases. Accuracy alone should not be used to judge its usefulness.

## 6. Fairness Check Results
Recall was compared across Gender, MaritalStatus and age bands.

- Gender: Female 31.25%; Male 16.13%.
- MaritalStatus: Divorced 0%; Married 16.67%; Single 28%.
- Age: 18–25: 57.14%; 26–35: 15%; 36–45: 18.18%; 46–55: 16.67%; 56+: 0%.

The largest observed gap was across age bands: 57.14 percentage points.

The 56+ group had only 3 actual leavers, so its recall estimate is unstable. These results indicate differences in model performance but do not by themselves prove discrimination.

## 7. Limitations
- The model has low recall for employees who leave.
- The dataset may not represent other organisations, countries or current workforce conditions.
- Small subgroup sample sizes make some fairness estimates unreliable.
- Fairness checks based on recall alone do not establish overall fairness.
- Correlation checks do not rule out every form of data leakage.
- Historical patterns may not remain valid over time.

## 8. Who Should Not Use the Model
The model should not be used by employers to make automated decisions about employees. It should not be used as the sole basis for any employment action.

## 9. Testing Before Release

### Test set used
A stratified 20% hold-out test set containing 294 rows, with random_state=42.

### Proposed release acceptance criteria
Before any controlled pilot, the model should meet these proposed minimum targets:

- Recall for attrition: at least 70%.
- Precision: at least 30%.
- F1 Score: at least 40%.
- Maximum subgroup recall gap: 15 percentage points, subject to adequate subgroup sample sizes.

These are proposed governance thresholds, not achieved model results. The current model does not meet the proposed recall or F1 targets and should not be released for operational use.

### Who pilots the app first
Any future pilot should begin with authorised HR analysts and data-governance reviewers in a controlled shadow-testing environment. Predictions must not influence employment decisions during the pilot.

## 10. Responsible Use Rules
- The model must NEVER be used to dismiss staff.
- The model must NEVER be used to refuse promotion.
- The model must NEVER be used to cut pay.
- Individual risk scores must be visible only to authorised HR personnel.
- Predictions must not replace human judgement or fair employment procedures.
- Any future deployment requires independent fairness, privacy and legal review.
