# 🔍 DataLens — Personal Data Science Toolkit

> A powerful, interactive data science dashboard built with Python and Streamlit. Upload any CSV dataset and instantly explore, analyze, clean, and visualize your data — no coding required.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Live Demo
👉 **[Launch DataLens](https://your-streamlit-url.streamlit.app)**
> Replace the link above with your Streamlit Cloud URL after deploying

---

## 📌 What is DataLens?

DataLens is a personal data science toolkit that allows anyone to upload any CSV file and instantly perform professional-grade statistical analysis and visualization — all through a clean, interactive web interface.

Built as a portfolio project to demonstrate end-to-end data science skills including data loading, cleaning, statistical analysis, machine learning, and visualization.

---

## ✨ Features

### 📂 Data Loader
- Upload any CSV file (up to 200MB)
- Instant dataset preview
- Row, column, and missing value counts
- Column data type inspection

### 📊 Descriptive Statistics
- Mean, Median, Min, Max, Std Dev
- 25th, 50th, 75th Percentiles
- Skewness and Kurtosis
- Missing value counts per column

### 📈 Visualizations
- Histogram with KDE curve
- Boxplot
- Scatter Plot
- Correlation Heatmap
- Bar Chart (categorical columns)
- Pie Chart (categorical columns)

### 🧪 Statistical Tests
- **T-Test** — test if two groups are statistically different
- **Pearson Correlation** — measure relationship strength between two columns
- **Linear Regression** — build a predictive model with chart, slope, intercept, R², and p-value

### 🧹 Data Cleaner
- Drop rows with missing values
- Fill missing values with column means
- Remove duplicate rows
- Export cleaned dataset as CSV

### 📅 Time Series Analysis
- Plot any numeric column over time
- Auto date parsing
- Mean, Min, Max summary stats

### 📉 Multiple Regression
- Select multiple feature variables (X)
- Predict any target variable (Y)
- View R-Squared, RMSE, and feature coefficients
- Actual vs Predicted scatter plot

### 💬 Sentiment Analysis
- Analyze any text column
- Positive, Neutral, Negative classification
- Polarity score distribution chart
- Row-level sentiment results

### 🚨 Outlier Detection
- IQR-based outlier detection
- Visual boxplot and distribution with bounds
- One-click outlier removal
- Outlier row preview

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| Python 3.13 | Core language |
| Streamlit | Web dashboard framework |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib | Chart rendering |
| Seaborn | Statistical visualizations |
| SciPy | Statistical tests |
| Scikit-learn | Multiple regression |
| TextBlob | Sentiment analysis |

---

## 📦 Installation (Run Locally)

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/DataLens.git
cd DataLens
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch DataLens**
```bash
streamlit run dashboard.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## 📁 Project Structure

```
DataLens/
├── dashboard.py          # Main Streamlit app
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
```

---

## 📊 Example Datasets to Try

- 🏛️ [Household Debt by State — Federal Reserve](https://www.federalreserve.gov/releases/z1/dataviz/household_debt/)
- 🎓 [College Scorecard — data.gov](https://collegescorecard.ed.gov/data/)
- 📈 [Kaggle Datasets](https://www.kaggle.com/datasets)
- 🏥 [Health Data — data.gov](https://catalog.data.gov/dataset?tags=health)

---

## 👨‍💻 About the Developer

**TG** — Aspiring Data Scientist

Building real-world data science projects to develop skills in Python, statistics, machine learning, and data visualization.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

⭐ If you found this useful, please consider giving it a star on GitHub!
