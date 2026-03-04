import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from textblob import TextBlob
import io

st.set_page_config(page_title="DataLens", page_icon="🔍", layout="wide")
st.title("🔍 DataLens")
st.caption("Your personal data science toolkit — load, explore, analyze, and visualize any dataset.")

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", [
    "📂 Data Loader",
    "📊 Descriptive Stats",
    "📈 Visualizations",
    "🧪 Statistical Tests",
    "🧹 Data Cleaner",
    "📅 Time Series",
    "📉 Multiple Regression",
    "💬 Sentiment Analysis",
    "🚨 Outlier Detection"
])

# ── Session State ─────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None

# ── Tab 1: Data Loader ────────────────────────────────────
if tab == "📂 Data Loader":
    st.header("📂 Data Loader")
    uploaded = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.df = df
        st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns!")
        st.dataframe(df.head(20))
        st.subheader("Dataset Info")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())
        st.subheader("Column Data Types")
        st.write(df.dtypes)
    else:
        st.info("Please upload a CSV file to get started.")

# ── Tab 2: Descriptive Stats ──────────────────────────────
elif tab == "📊 Descriptive Stats":
    st.header("📊 Descriptive Statistics")
    if st.session_state.df is None:
        st.warning("Please load a dataset first in the Data Loader tab.")
    else:
        df = st.session_state.df
        numeric_df = df.select_dtypes(include=np.number)
        st.subheader("Summary Statistics")
        st.dataframe(numeric_df.describe().round(2))
        st.subheader("Skewness & Kurtosis")
        sk = pd.DataFrame({
            "Skewness": numeric_df.skew().round(3),
            "Kurtosis": numeric_df.kurtosis().round(3)
        })
        st.dataframe(sk)
        st.subheader("Missing Values per Column")
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Count"]
        st.dataframe(missing)

# ── Tab 3: Visualizations ─────────────────────────────────
elif tab == "📈 Visualizations":
    st.header("📈 Visualizations")
    if st.session_state.df is None:
        st.warning("Please load a dataset first in the Data Loader tab.")
    else:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        chart_type = st.selectbox("Chart Type", [
            "Histogram", "Boxplot", "Scatter Plot",
            "Correlation Heatmap", "Bar Chart", "Pie Chart"
        ])

        if chart_type == "Histogram":
            col = st.selectbox("Select Column", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

        elif chart_type == "Boxplot":
            col = st.selectbox("Select Column", numeric_cols)
            fig, ax = plt.subplots()
            sns.boxplot(y=df[col].dropna(), ax=ax)
            ax.set_title(f"Boxplot of {col}")
            st.pyplot(fig)

        elif chart_type == "Scatter Plot":
            col_x = st.selectbox("X Axis", numeric_cols)
            col_y = st.selectbox("Y Axis", numeric_cols, index=1)
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax)
            ax.set_title(f"{col_x} vs {col_y}")
            st.pyplot(fig)

        elif chart_type == "Correlation Heatmap":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[numeric_cols].corr().round(2),
                       annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

        elif chart_type == "Bar Chart":
            if cat_cols:
                col = st.selectbox("Select Categorical Column", cat_cols)
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind="bar", ax=ax, color="steelblue")
                ax.set_title(f"Value Counts — {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                st.pyplot(fig)
            else:
                st.warning("No categorical columns found in dataset.")

        elif chart_type == "Pie Chart":
            if cat_cols:
                col = st.selectbox("Select Categorical Column", cat_cols)
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind="pie", ax=ax,
                    autopct="%1.1f%%", startangle=90)
                ax.set_title(f"Distribution — {col}")
                ax.set_ylabel("")
                st.pyplot(fig)
            else:
                st.warning("No categorical columns found in dataset.")

# ── Tab 4: Statistical Tests ──────────────────────────────
elif tab == "🧪 Statistical Tests":
    st.header("🧪 Statistical Tests")
    if st.session_state.df is None:
        st.warning("Please load a dataset first in the Data Loader tab.")
    else:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        test = st.selectbox("Select Test", [
            "T-Test", "Correlation", "Linear Regression"
        ])

        if test == "T-Test":
            col = st.selectbox("Select Numeric Column", numeric_cols)
            split_col = st.selectbox("Split Groups By",
                df.select_dtypes(include=["object", "category"]).columns.tolist())
            if st.button("Run T-Test"):
                groups = df[split_col].dropna().unique()
                if len(groups) >= 2:
                    g1 = df[df[split_col] == groups[0]][col].dropna()
                    g2 = df[df[split_col] == groups[1]][col].dropna()
                    t_stat, p_value = stats.ttest_ind(g1, g2)
                    st.metric("T-Statistic", round(t_stat, 4))
                    st.metric("P-Value", round(p_value, 4))
                    if p_value < 0.05:
                        st.success("Statistically significant difference (p < 0.05)")
                    else:
                        st.info("No significant difference (p >= 0.05)")

        elif test == "Correlation":
            col_x = st.selectbox("Column X", numeric_cols)
            col_y = st.selectbox("Column Y", numeric_cols, index=1)
            if st.button("Run Correlation"):
                clean = df[[col_x, col_y]].dropna()
                corr, p_value = stats.pearsonr(clean[col_x], clean[col_y])
                st.metric("Pearson Correlation", round(corr, 4))
                st.metric("P-Value", round(p_value, 4))
                if p_value < 0.05:
                    st.success("Statistically significant correlation (p < 0.05)")
                else:
                    st.info("No significant correlation (p >= 0.05)")

        elif test == "Linear Regression":
            col_x = st.selectbox("Independent Variable (X)", numeric_cols)
            col_y = st.selectbox("Dependent Variable (Y)", numeric_cols, index=1)
            if st.button("Run Regression"):
                clean = df[[col_x, col_y]].dropna()
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    clean[col_x], clean[col_y])
                st.metric("Slope", round(slope, 4))
                st.metric("Intercept", round(intercept, 4))
                st.metric("R-Squared", round(r_value**2, 4))
                st.metric("P-Value", round(p_value, 4))
                fig, ax = plt.subplots()
                sns.scatterplot(data=clean, x=col_x, y=col_y, ax=ax)
                ax.plot(clean[col_x], slope * clean[col_x] + intercept,
                       color="red")
                ax.set_title(f"Regression: {col_x} vs {col_y}")
                st.pyplot(fig)

# ── Tab 5: Data Cleaner ───────────────────────────────────
elif tab == "🧹 Data Cleaner":
    st.header("🧹 Data Cleaner")
    if st.session_state.df is None:
        st.warning("Please load a dataset first in the Data Loader tab.")
    else:
        df = st.session_state.df.copy()
        st.subheader("Current Missing Values")
        st.dataframe(df.isnull().sum().reset_index().rename(
            columns={"index": "Column", 0: "Missing"}))

        col1, col2, col3 = st.columns(3)
        if col1.button("Drop Missing Rows"):
            st.session_state.df = df.dropna()
            st.success(f"Dropped rows. New shape: {st.session_state.df.shape}")

        if col2.button("Fill with Mean"):
            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            st.session_state.df = df
            st.success("Filled missing values with column means!")

        if col3.button("Remove Duplicates"):
            before = len(df)
            st.session_state.df = df.drop_duplicates()
            after = len(st.session_state.df)
            st.success(f"Removed {before - after} duplicate rows!")

        st.subheader("Export Cleaned Data")
        csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

        st.subheader("Cleaned Dataset Preview")
        st.dataframe(st.session_state.df.head(20))

# ── Tab 6: Time Series ────────────────────────────────────
elif tab == "📅 Time Series":
    st.header("📅 Time Series Analysis")
    if st.session_state.df is None:
        st.warning("Please load a dataset first in the Data Loader tab.")
    else:
        df = st.session_state.df
        date_cols = df.select_dtypes(include=["object"]).columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        date_col = st.selectbox("Select Date Column", date_cols)
        value_col = st.selectbox("Select Value Column", numeric_cols)

        if st.button("Plot Time Series"):
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                ts = df[[date_col, value_col]].dropna().sort_values(date_col)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(ts[date_col], ts[value_col], color="steelblue")
                ax.set_title(f"{value_col} Over Time")
                ax.set_xlabel("Date")
                ax.set_ylabel(value_col)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                st.subheader("Time Series Stats")
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean", round(ts[value_col].mean(), 2))
                col2.metric("Min", round(ts[value_col].min(), 2))
                col3.metric("Max", round(ts[value_col].max(), 2))
            except Exception as e:
                st.error(f"Could not parse date column: {e}")

# ── Tab 7: Multiple Regression ────────────────────────────
elif tab == "📉 Multiple Regression":
    st.header("📉 Multiple Regression")
    if st.session_state.df is None:
        st.warning("Please load a dataset first in the Data Loader tab.")
    else:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        target = st.selectbox("Target Variable (Y)", numeric_cols)
        features = st.multiselect("Feature Variables (X)", 
            [c for c in numeric_cols if c != target])

        if st.button("Run Multiple Regression"):
            if len(features) < 2:
                st.warning("Please select at least 2 feature variables.")
            else:
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score, mean_squared_error

                clean = df[features + [target]].dropna()
                X = clean[features]
                y = clean[target]

                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)

                st.subheader("Model Results")
                col1, col2 = st.columns(2)
                col1.metric("R-Squared", round(r2_score(y, y_pred), 4))
                col2.metric("RMSE", round(np.sqrt(mean_squared_error(y, y_pred)), 4))

                st.subheader("Feature Coefficients")
                coef_df = pd.DataFrame({
                    "Feature": features,
                    "Coefficient": model.coef_.round(4)
                })
                st.dataframe(coef_df)

                fig, ax = plt.subplots()
                ax.scatter(y, y_pred, alpha=0.5, color="steelblue")
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 
                       color="red", linestyle="--")
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig)

# ── Tab 8: Sentiment Analysis ─────────────────────────────
elif tab == "💬 Sentiment Analysis":
    st.header("💬 Sentiment Analysis")
    if st.session_state.df is None:
        st.warning("Please load a dataset first in the Data Loader tab.")
    else:
        df = st.session_state.df
        text_cols = df.select_dtypes(include="object").columns.tolist()

        text_col = st.selectbox("Select Text Column", text_cols)

        if st.button("Run Sentiment Analysis"):
            df["Polarity"] = df[text_col].dropna().apply(
                lambda x: TextBlob(str(x)).sentiment.polarity)
            df["Sentiment"] = df["Polarity"].apply(
                lambda x: "Positive" if x > 0 
                else ("Negative" if x < 0 else "Neutral"))

            st.subheader("Sentiment Distribution")
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", len(df[df["Sentiment"] == "Positive"]))
            col2.metric("Neutral", len(df[df["Sentiment"] == "Neutral"]))
            col3.metric("Negative", len(df[df["Sentiment"] == "Negative"]))

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            df["Sentiment"].value_counts().plot(kind="bar", ax=axes[0],
                color=["green", "gray", "red"])
            axes[0].set_title("Sentiment Counts")

            sns.histplot(df["Polarity"].dropna(), kde=True, ax=axes[1],
                color="steelblue")
            axes[1].set_title("Polarity Distribution")
            st.pyplot(fig)

            st.subheader("Sample Results")
            st.dataframe(df[[text_col, "Polarity", "Sentiment"]].head(20))

# ── Tab 9: Outlier Detection ──────────────────────────────
elif tab == "🚨 Outlier Detection":
    st.header("🚨 Outlier Detection")
    if st.session_state.df is None:
        st.warning("Please load a dataset first in the Data Loader tab.")
    else:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        col = st.selectbox("Select Column", numeric_cols)

        if st.button("Detect Outliers"):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)]

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Outliers", len(outliers))
            col2.metric("Lower Bound", round(lower, 2))
            col3.metric("Upper Bound", round(upper, 2))

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.boxplot(y=df[col], ax=axes[0])
            axes[0].set_title(f"Boxplot — {col}")

            sns.histplot(df[col].dropna(), kde=True, ax=axes[1],
                color="steelblue")
            axes[1].axvline(lower, color="red", linestyle="--",
                label="Lower Bound")
            axes[1].axvline(upper, color="red", linestyle="--",
                label="Upper Bound")
            axes[1].legend()
            axes[1].set_title(f"Distribution — {col}")
            st.pyplot(fig)

            if len(outliers) > 0:
                st.subheader("Outlier Rows")
                st.dataframe(outliers)

                if st.button("Remove Outliers"):
                    st.session_state.df = df[
                        (df[col] >= lower) & (df[col] <= upper)]
                    st.success(f"Removed {len(outliers)} outliers!")