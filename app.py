
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             mean_squared_error, r2_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title='FastTrack Logistics Dashboard', layout='wide')
st.title('🚚 FastTrack Logistics – AI‑Driven Insights Dashboard')

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data('fasttrack_logistics_synthetic.csv')

# Sidebar filters (data‑vis only)
st.sidebar.header('Global Filters (apply to Visualisation tab)')
industries = st.sidebar.multiselect('Industry', df['Industry'].unique(), default=list(df['Industry'].unique()))
regions = st.sidebar.multiselect('Region', df['Business_Region'].unique(), default=list(df['Business_Region'].unique()))
df_vis = df[(df['Industry'].isin(industries)) & (df['Business_Region'].isin(regions))]

#############################
# Helper functions
#############################

def preprocess_data(data, target_col):
    X = data.drop(columns=[target_col])
    y = data[target_col].map({'Yes':1, 'No':0}) if data[target_col].dtype == object else data[target_col]

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ])

    return X, y, preprocessor

def train_classification_models(data):
    X, y, preprocessor = preprocess_data(data, 'Switch_to_20pct_Faster_Service')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    algorithms = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    reports = {}
    roc_data = {}

    for name, model in algorithms.items():
        pipe = Pipeline(steps=[('prep', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, 'predict_proba') else None

        reports[name] = {
            'Train Acc': accuracy_score(y_train, pipe.predict(X_train)).round(3),
            'Test Acc': accuracy_score(y_test, y_pred).round(3),
            'Precision': precision_score(y_test, y_pred).round(3),
            'Recall': recall_score(y_test, y_pred).round(3),
            'F1': f1_score(y_test, y_pred).round(3),
            'model': pipe
        }

        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            roc_data[name] = (fpr, tpr, roc_auc)

    return reports, roc_data

@st.cache_resource
def get_classification_results(data):
    return train_classification_models(data)

reports, roc_data = get_classification_results(df)

# --------- TABS -----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(['📊 Data Visualisation', 
                                        '🧮 Classification', 
                                        '🔗 Clustering', 
                                        '📋 Association Rules', 
                                        '📈 Regression'])

#############################################
# TAB 1 – Data Visualisation
#############################################
with tab1:
    st.header('Descriptive Insights')
    st.write('Filters applied on left sidebar.')

    col1, col2 = st.columns(2)
    # 1. Delivery time distribution
    with col1:
        st.subheader('Distribution of Delivery Time (hrs)')
        fig, ax = plt.subplots()
        sns.histplot(df_vis['Avg_Delivery_Time_Hrs'], kde=True, ax=ax)
        st.pyplot(fig)
    # 2. Cost by Industry
    with col2:
        st.subheader('Average Delivery Cost by Industry')
        avg_cost = df_vis.groupby('Industry')['Avg_Delivery_Cost'].mean().sort_values()
        fig, ax = plt.subplots()
        avg_cost.plot(kind='barh', ax=ax)
        ax.set_xlabel('Avg Cost')
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    # 3. Urgent deliveries vs switch willingness
    with col3:
        st.subheader('Urgent Deliveries vs Switch Decision')
        fig, ax = plt.subplots()
        sns.boxplot(x='Switch_to_20pct_Faster_Service', 
                    y='Urgent_Deliveries_per_Week', data=df_vis, ax=ax)
        st.pyplot(fig)
    # 4. Heatmap correlations (numeric)
    with col4:
        st.subheader('Correlation of Key Numeric Features')
        num_cols = ['Avg_Delivery_Time_Hrs', 'Avg_Delivery_Cost', 
                    'Urgent_Deliveries_per_Week', 'Avg_Fuel_Cost', 
                    'Driver_Wage_per_Hour', 'Avg_Route_Distance_KM']
        corr = df_vis[num_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, ax=ax)
        st.pyplot(fig)

    # 5. Pie – Pricing preference
    st.subheader('Pricing Preference Breakdown')
    pricing_counts = df_vis['Pricing_Preference'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(pricing_counts.values, labels=pricing_counts.index, autopct='%1.1f%%')
    ax.axis('equal')
    st.pyplot(fig)

    # 6. Industry vs switch rate
    st.subheader('Switch Rate by Industry')
    switch_rate = df_vis.groupby('Industry')['Switch_to_20pct_Faster_Service'].apply(lambda x: (x=='Yes').mean())
    fig, ax = plt.subplots()
    switch_rate.sort_values().plot(kind='barh', ax=ax)
    ax.set_xlabel('Switch Probability')
    st.pyplot(fig)

    # 7. Distribution of maintenance cost
    st.subheader('Vehicle Maintenance Cost Distribution')
    fig, ax = plt.subplots()
    sns.violinplot(data=df_vis, y='Vehicle_Maintenance_Cost_Monthly', ax=ax)
    st.pyplot(fig)

    # 8. Last‑mile cost percent by region
    st.subheader('Last‑Mile Cost % by Region')
    fig, ax = plt.subplots()
    sns.boxplot(x='Business_Region', y='Last_Mile_Delivery_Cost_Percent', data=df_vis, ax=ax)
    st.pyplot(fig)

    # 9. Relationship between distance and fuel cost
    st.subheader('Distance vs Fuel Cost')
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_vis, x='Avg_Route_Distance_KM', y='Avg_Fuel_Cost', hue='Industry', ax=ax)
    st.pyplot(fig)

    # 10. Histogram of satisfaction scores
    st.subheader('Current Provider Satisfaction Scores')
    fig, ax = plt.subplots()
    sns.histplot(df_vis['Current_Provider_Satisfaction'], bins=20, ax=ax)
    st.pyplot(fig)

#############################################
# TAB 2 – Classification
#############################################
with tab2:
    st.header('Classification Models – Predict Switch Decision')

    # Show metrics table
    st.subheader('Model Performance')
    metrics_df = pd.DataFrame(reports).T[['Train Acc','Test Acc','Precision','Recall','F1']].sort_values('Test Acc', ascending=False)
    st.dataframe(metrics_df, use_container_width=True)

    # Select algorithm for confusion matrix
    algo_choice = st.selectbox('Select Algorithm for Detailed View', list(reports.keys()))
    model_pipe = reports[algo_choice]['model']

    # Confusion matrix
    X, y, preproc = preprocess_data(df, 'Switch_to_20pct_Faster_Service')
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

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
    ax.plot([0,1],[0,1],'k--')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.legend()
    st.pyplot(fig)

    # Upload new data
    st.subheader('Upload New Data for Prediction')
    uploaded = st.file_uploader('Upload CSV without target column', type='csv')
    if uploaded is not None:
        new_df = pd.read_csv(uploaded)
        preds = model_pipe.predict(new_df)
        result = new_df.copy()
        result['Predicted_Switch'] = preds
        st.success('Predictions generated!')
        st.dataframe(result.head())

        csv = result.to_csv(index=False).encode('utf-8')
        st.download_button('Download Predictions', csv, 'predictions.csv', 'text/csv')

#############################################
# TAB 3 – Clustering
#############################################
with tab3:
    st.header('K‑Means Customer Segmentation')

    num_clusters = st.slider('Select number of clusters', 2, 10, 4)
    numeric_data = df.select_dtypes(exclude=['object']).dropna(axis=1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_data)

    # Elbow chart (computed once)
    @st.cache_data
    def elbow_data():
        inertias = []
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42, n_init='auto')
            km.fit(scaled)
            inertias.append(km.inertia_)
        return inertias
    inertias = elbow_data()

    st.subheader('Elbow Method')
    fig, ax = plt.subplots()
    ax.plot(range(2,11), inertias, marker='o')
    ax.axvline(num_clusters, color='r', linestyle='--')
    ax.set_xlabel('k'); ax.set_ylabel('Inertia')
    st.pyplot(fig)

    # Fit chosen k
    km_model = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    clusters = km_model.fit_predict(scaled)
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters

    # Persona table (mean of numerical fields)
    st.subheader('Cluster Personas (Numeric Means)')
    persona = df_clustered.groupby('Cluster')[['Avg_Delivery_Time_Hrs',
                                               'Avg_Delivery_Cost',
                                               'Urgent_Deliveries_per_Week',
                                               'Current_Provider_Satisfaction']].mean().round(2)
    st.dataframe(persona)

    # Download button
    csv_clustered = df_clustered.to_csv(index=False).encode('utf-8')
    st.download_button('Download Clustered Data', csv_clustered, 'fasttrack_clustered.csv', 'text/csv')

#############################################
# TAB 4 – Association Rule Mining
#############################################
with tab4:
    st.header('Market Basket – Association Rules')

    cols_multi = st.multiselect('Select categorical columns for Apriori',
                                options=['Switch_Factors','Spike_Months'],
                                default=['Switch_Factors','Spike_Months'])

    min_support = st.slider('Min Support', 0.01, 0.5, 0.05, step=0.01)
    min_conf = st.slider('Min Confidence', 0.1, 1.0, 0.4, step=0.05)

    # Prepare one-hot
    basket = pd.DataFrame()
    for col in cols_multi:
        exploded = df[col].str.get_dummies(sep=', ')
        basket = pd.concat([basket, exploded], axis=1)

    freq_items = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_items, metric='confidence', min_threshold=min_conf)
    rules = rules.sort_values('confidence', ascending=False).head(10)

    st.subheader('Top 10 Rules')
    st.dataframe(rules[['antecedents','consequents','support','confidence','lift']])

#############################################
# TAB 5 – Regression
#############################################
with tab5:
    st.header('Regression Models – Predict Delivery Time')

    target = 'Avg_Delivery_Time_Hrs'
    features = df.drop(columns=[target])

    # Prep
    X = features
    y = df[target]

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    preprocessor_r = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ])

    models_r = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.001),
        'Decision Tree': DecisionTreeRegressor(random_state=42)
    }

    results_r = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, mdl in models_r.items():
        pipe = Pipeline([('prep', preprocessor_r), ('model', mdl)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        results_r[name] = {'RMSE': round(rmse,2), 'R2': round(r2,3)}

    st.subheader('Performance')
    st.dataframe(pd.DataFrame(results_r).T)

    # Scatter plot for best R2
    best_model = max(results_r.items(), key=lambda x: x[1]['R2'])[0]
    st.subheader(f'Actual vs Predicted – {best_model}')
    best_pipe = Pipeline([('prep', preprocessor_r), ('model', models_r[best_model])])
    best_pipe.fit(X_train, y_train)
    preds_best = best_pipe.predict(X_test)
    fig, ax = plt.subplots()
    ax.scatter(y_test, preds_best)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    st.pyplot(fig)
