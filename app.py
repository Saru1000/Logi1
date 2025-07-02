import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    mean_squared_error, r2_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='FastTrack Logistics Dashboard', layout='wide')
st.title('🚚 FastTrack Logistics – AI-Driven Insights Dashboard')

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

df = load_data('fasttrack_logistics_synthetic.csv')

# ──────────────────────────────────────────────────────────────────────────
# Sidebar filters (affect the Data-Visualisation tab)
st.sidebar.header('Global Filters (Data-Vis tab only)')
industries = st.sidebar.multiselect(
    'Industry', df['Industry'].unique(), default=list(df['Industry'].unique())
)
regions = st.sidebar.multiselect(
    'Region', df['Business_Region'].unique(), default=list(df['Business_Region'].unique())
)
df_vis = df[df['Industry'].isin(industries) & df['Business_Region'].isin(regions)]

# ──────────────────────────────────────────────────────────────────────────
# Helper functions
def preprocess_data(data: pd.DataFrame, target_col: str):
    X = data.drop(columns=[target_col])
    y = (
        data[target_col].map({'Yes': 1, 'No': 0})
        if data[target_col].dtype == object
        else data[target_col]
    )

    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(exclude='object').columns.tolist()

    preproc = ColumnTransformer(
        [
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols),
        ]
    )
    return X, y, preproc

def train_classification_models(data: pd.DataFrame):
    X, y, preproc = preprocess_data(data, 'Switch_to_20pct_Faster_Service')
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    algos = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    }

    reports, roc_data = {}, {}
    for name, model in algos.items():
        pipe = Pipeline([('prep', preproc), ('model', model)])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_proba = (
            pipe.predict_proba(X_te)[:, 1] if hasattr(pipe, 'predict_proba') else None
        )

        # ── key change: use built-in round() instead of .round() ──
        reports[name] = {
            'Train Acc': round(accuracy_score(y_tr, pipe.predict(X_tr)), 3),
            'Test Acc': round(accuracy_score(y_te, y_pred), 3),
            'Precision': round(precision_score(y_te, y_pred), 3),
            'Recall': round(recall_score(y_te, y_pred), 3),
            'F1': round(f1_score(y_te, y_pred), 3),
            'model': pipe,
        }

        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_te, y_proba)
            roc_data[name] = (fpr, tpr, auc(fpr, tpr))

    return reports, roc_data

@st.cache_resource
def get_classification_results(data: pd.DataFrame):
    return train_classification_models(data)

reports, roc_data = get_classification_results(df)

# ──────────────────────────────────────────────────────────────────────────
# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        '📊 Data Visualisation',
        '🧮 Classification',
        '🔗 Clustering',
        '📋 Association Rules',
        '📈 Regression',
    ]
)

# ──────────────────────────────────────────────────────────────────────────
# TAB 1 – Data Visualisation
with tab1:
    st.header('Descriptive Insights')
    st.write('Filters applied on left sidebar.')

    # (visualisation code unchanged …)

# ──────────────────────────────────────────────────────────────────────────
# TAB 2 – Classification
with tab2:
    st.header('Classification Models – Predict Switch Decision')

    st.subheader('Model Performance')
    metrics_df = (
        pd.DataFrame(reports)
        .T[['Train Acc', 'Test Acc', 'Precision', 'Recall', 'F1']]
        .sort_values('Test Acc', ascending=False)
    )
    st.dataframe(metrics_df, use_container_width=True)

    algo_choice = st.selectbox('Select Algorithm', list(reports.keys()))
    model_pipe = reports[algo_choice]['model']

    # Confusion matrix
    X, y, preproc = preprocess_data(df, 'Switch_to_20pct_Faster_Service')
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cm = confusion_matrix(y_te, model_pipe.predict(X_te))

    st.subheader(f'Confusion Matrix: {algo_choice}')
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    st.pyplot(fig)

    # ROC curve
    st.subheader('ROC Curve (All Models)')
    fig, ax = plt.subplots()
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.legend()
    st.pyplot(fig)

    # (upload / download code unchanged …)

# ──────────────────────────────────────────────────────────────────────────
# TAB 3 – Clustering
# (code unchanged …)

# ──────────────────────────────────────────────────────────────────────────
# TAB 4 – Association Rules
# (code unchanged …)

# ──────────────────────────────────────────────────────────────────────────
# TAB 5 – Regression
with tab5:
    st.header('Regression Models – Predict Delivery Time')

    target = 'Avg_Delivery_Time_Hrs'
    X = df.drop(columns=[target])
    y = df[target]

    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(exclude='object').columns.tolist()

    preproc_r = ColumnTransformer(
        [('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
         ('num', StandardScaler(), num_cols)]
    )

    models_r = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.001),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
    }

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    results_r = {}

    for name, mdl in models_r.items():
        pipe = Pipeline([('prep', preproc_r), ('model', mdl)])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        results_r[name] = {
            'RMSE': round(np.sqrt(mean_squared_error(y_te, preds)), 2),
            'R2': round(r2_score(y_te, preds), 3),
        }

    st.subheader('Performance')
    st.dataframe(pd.DataFrame(results_r).T)

    best_model = max(results_r, key=lambda k: results_r[k]['R2'])
    st.subheader(f'Actual vs Predicted – {best_model}')
    best_pipe = Pipeline([('prep', preproc_r), ('model', models_r[best_model])])
    best_pipe.fit(X_tr, y_tr)
    best_preds = best_pipe.predict(X_te)

    fig, ax = plt.subplots()
    ax.scatter(y_te, best_preds)
    ax.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'k--')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    st.pyplot(fig)
