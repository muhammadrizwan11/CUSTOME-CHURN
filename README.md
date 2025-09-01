# ML Intern Assessment Test

A comprehensive evaluation for Machine Learning intern candidates focusing on Python data handling and basic ML implementation skills.

## 📋 Overview

This assessment is designed for university students applying for ML internships, with tasks that reflect real-world data science workflows. The test is beginner-friendly while still challenging enough to differentiate skill levels.

**Duration:** 40-50 minutes  
**Points:** 60 points total  
**Skill Level:** Beginner to Intermediate  

## 🎯 Assessment Structure

```markdown
### Part 1: Python & Data Fundamentals (25 points, 20-25 minutes)
- **Task 1.1:** Data Loading & Basic Exploration (8 points)
- **Task 1.2:** Data Quality Assessment (7 points)  
- **Task 1.3:** Simple Data Visualization (5 points)
- **Task 1.4:** Basic Data Preparation (5 points)

### Part 2: Basic Machine Learning (35 points, 20-25 minutes)
- **Task 2.1:** Data Preparation for ML (8 points)
- **Task 2.2:** Build Basic ML Models (12 points)
- **Task 2.3:** Model Evaluation & Understanding (10 points)
- **Task 2.4:** Basic Feature Importance (5 points)
``` 


## 🏗️ Repository Structure
```markdown
ml-intern-assessment/
├── README.md
├── requirements.txt
├── data/
│   └── customer_data.csv
├── templates/
    ├── part1_data_fundamentals.ipynb
    └── part2_basic_ml.ipynb
``` 

## 🚀 Quick Start for Candidates

### Prerequisites
- Python 3.7+ installed
- Basic familiarity with Jupyter notebooks
- Understanding of Python fundamentals

### Setup Instructions
1. **Install dependencies:**
   - You have to create your own requirements.txt
   ```bash
   pip install -r requirements.txt
   ```

2. **Start with Part 1:**
   - Open `templates/part1_data_fundamentals.ipynb`
   - Complete all tasks in order
   - Save your work regularly

3. **Proceed to Part 2:**
   - Open `templates/part2_basic_ml.ipynb`
   - Use the cleaned data from Part 1
   - Complete all ML tasks

## 📊 Dataset Information

**File:** `data/customer_data.csv`  
**Description:** Telecom company customer data for churn prediction  
**Size:** 1,000 customers, 8 features  
**Target Variable:** Customer churn (Yes/No)  

**Features:**
- `customer_id`: Unique identifier
- `age`: Customer age  
- `gender`: Male/Female
- `monthly_charges`: Monthly bill amount
- `total_charges`: Total charges to date
- `contract_type`: Month-to-month, One year, Two year
- `internet_service`: DSL, Fiber optic, No
- `churn`: Target variable (Yes/No)

## 📝 Submission Guidelines

### What to Submit
1. **Completed Jupyter notebooks:**
   - `part1_data_fundamentals.ipynb` (with your solutions)
   - `part2_basic_ml.ipynb` (with your solutions)
   - `cleaned_customer_data.csv` (generated in Part 1)


### Quality Standards
- ✅ All code cells must run without errors
- ✅ Include markdown explanations for your approach
- ✅ Clear, readable code with meaningful variable names
- ✅ Complete all required tasks
- ✅ Provide interpretations where requested

## 🎓 Skills Assessed

### Technical Skills
- **Python Programming:** Basic syntax, data structures
- **Pandas:** DataFrame manipulation, data cleaning
- **Data Visualization:** Matplotlib/Seaborn basics
- **Scikit-learn:** Model training, evaluation
- **Data Preprocessing:** Handling missing values, encoding

### Conceptual Understanding
- **ML Workflow:** Train-test split, model evaluation
- **Model Interpretation:** Confusion matrices, feature importance
- **Business Insights:** Connecting technical results to decisions

## ⚠️ Important Notes

### For Test Takers
- **Time Management:** Don't spend too long on any single task
- **Focus on Functionality:** Working code is more important than perfect optimization
- **Ask for Help:** Clarify requirements if instructions are unclear
- **Save Frequently:** Prevent losing your work

### Common Mistakes to Avoid
- Not importing required libraries
- Forgetting to save intermediate results
- Skipping data verification steps
- Not providing interpretations where requested
- Poor time management between parts

## 📈 Success Tips
- Follow the instructions step by step
- Use the provided code patterns
- Focus on getting basic functionality working
- Don't worry about perfect optimization
- Complete all tasks efficiently
- Add thoughtful insights in markdown cells
- Demonstrate understanding beyond basic requirements
- Provide meaningful business recommendations

## 🔧 Troubleshooting

### Common Issues
1. **Import Errors:** Check that all packages are installed correctly
2. **File Path Issues:** Ensure you're in the correct directory
3. **Data Loading Problems:** Verify the CSV file exists in the data folder

### Getting Help
- Review error messages carefully
- Ensure all file paths are correct
- Verify Python environment is properly configured

## 📊 Evaluation Criteria

### Technical Implementation (70%)
- Code functionality and correctness
- Proper use of libraries and methods
- Data handling and preprocessing
- Model building and evaluation

### Understanding & Communication (30%)
- Interpretation of results
- Business insights and recommendations
- Code organization and documentation
- Problem-solving approach
