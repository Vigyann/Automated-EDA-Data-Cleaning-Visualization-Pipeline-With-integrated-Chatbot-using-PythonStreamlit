import streamlit as st
from data_cleaning import (
    handle_missing_values, remove_duplicates, convert_data_type,
    rename_columns, clean_special, check_unique_values, drop_selected_columns
)
from data_visualisation import validate_chart_type, plot_chart
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import numpy as np
from data_assistant import ask_model
from modelbuilding import detect_task_type, evaluate_models, plot_results


# ---------- Utilities ----------
def numeric_cols(frame: pd.DataFrame):
    return [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]


def categorical_cols(frame: pd.DataFrame):
    return [
        c for c in frame.columns
        if pd.api.types.is_object_dtype(frame[c]) or pd.api.types.is_categorical_dtype(frame[c])
    ]


# ---------- Streamlit App ----------
st.title(" Automated Data Cleaning, Visualization & Model Building")

uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # --- Step 1: Load and persist dataset in session_state ---
    if 'original_df' not in st.session_state:
        st.session_state['original_df'] = pd.read_csv(uploaded_file)
    if 'cleaned_df' not in st.session_state:
        st.session_state['cleaned_df'] = st.session_state['original_df'].copy()

    df = st.session_state['cleaned_df']
    st.write("### Current Active Data:")
    st.dataframe(df.head())

    # ===========================
    # 🧹 Data Cleaning Sidebar
    # ===========================
    st.sidebar.header(" Data Cleaning Options")

    # ---------- Convert Data Types ----------
    if st.sidebar.checkbox("Convert Data Types"):
        chosen_cols = st.sidebar.multiselect("Pick columns to convert", df.columns)
        for col in chosen_cols:
            tgt = st.sidebar.selectbox(
                f"{col} →",
                ('(no change)', 'int', 'float', 'str', 'datetime', 'bool', 'category'),
                key=f"dtype_{col}"
            )
            if tgt != '(no change)':
                df = convert_data_type(df, col, tgt)
                st.session_state['cleaned_df'] = df
        st.success("Data type conversion applied.")

    # ---------- Check Unique Values ----------
    if st.sidebar.checkbox("Check Unique Values"):
        cols_to_check = st.sidebar.multiselect("Select columns to check", df.columns)
        if st.sidebar.button("Show Unique Values"):
            unique_vals = check_unique_values(df, cols_to_check)
            for col, vals in unique_vals.items():
                st.write(f"**{col}** → {vals}")

    # ---------- Drop Columns ----------
    if st.sidebar.checkbox("Drop Columns"):
        cols_to_drop = st.sidebar.multiselect("Select columns to drop", df.columns)
        if st.sidebar.button("Drop Selected Columns"):
            df = drop_selected_columns(df, cols_to_drop)
            st.session_state['cleaned_df'] = df
            st.success(f"Dropped columns: {cols_to_drop}")
            st.dataframe(df)

    # ---------- Handle Missing Values ----------
    if st.sidebar.checkbox("Handle Missing Values"):
        strategy = st.sidebar.selectbox(
            "Strategy for Missing Values",
            ('drop', 'mean', 'median', 'most_frequent')
        )
        df = handle_missing_values(df, strategy=strategy)
        st.session_state['cleaned_df'] = df
        st.success("Handled missing values successfully!")
        st.dataframe(df.head())

    # ---------- Remove Duplicates ----------
    if st.sidebar.checkbox("Remove Duplicates"):
        df = remove_duplicates(df)
        st.session_state['cleaned_df'] = df
        st.success("Duplicates removed successfully!")
        st.dataframe(df.head())

    # ---------- Rename Columns ----------
    if st.sidebar.checkbox("Rename Columns"):
        st.sidebar.write("### Rename Columns")
        rename_dict = {}
        for col in df.columns:
            new_name = st.sidebar.text_input(f"Rename '{col}' to:", col, key=f"rename_{col}")
            if new_name and new_name != col:
                rename_dict[col] = new_name
        if st.sidebar.button("Apply Renaming"):
            df = rename_columns(df, rename_dict)
            st.session_state['cleaned_df'] = df
            st.success("Column names updated successfully!")
            st.dataframe(df.head())

    # ---------- Clean String Columns ----------
    if st.sidebar.checkbox("Clean String Columns"):
        st.sidebar.subheader("Configure Cleaning Rules")
        cols_to_clean = st.sidebar.multiselect("Select columns to clean", df.columns)
        cleaning_rules = {}
        for col in cols_to_clean:
            st.sidebar.markdown(f"**Column: {col}**")
            modes = st.sidebar.multiselect(
                f"Select cleaning modes for {col}",
                ["remove_special", "remove_custom", "keep_before", "keep_after"],
                default=["remove_special"],
                key=f"modes_{col}"
            )
            rule = {"modes": modes, "keep_part": None, "remove_chars": None}
            if "remove_custom" in modes:
                rule["remove_chars"] = st.sidebar.text_input(
                    f"Enter custom characters to remove (for {col})", value="", key=f"custom_{col}"
                )
            if "keep_before" in modes or "keep_after" in modes:
                rule["keep_part"] = st.sidebar.text_input(
                    f"Enter delimiter for keep_before/after (for {col})", value="/", key=f"delimiter_{col}"
                )
            cleaning_rules[col] = rule

        if st.sidebar.button("Apply Cleaning"):
            for col, rule in cleaning_rules.items():
                for mode in rule["modes"]:
                    df = clean_special(
                        df,
                        col,
                        mode=mode,
                        keep_part=rule["keep_part"],
                        remove_chars=rule["remove_chars"]
                    )
            st.session_state['cleaned_df'] = df
            st.success("Cleaning applied successfully!")
            st.dataframe(df)

    # ---------- Data Visualization ----------
    if st.sidebar.checkbox("Data Visualisation"):
        chart_type = st.sidebar.selectbox(
            "Select chart type",
            [
                "Histogram", "Bar", "Line", "Scatter", "Boxplot", "Area",
                "Pie", "Donut", "Countplot", "Treemap", "Bubble",
                "Rolling Average", "Correlation Heatmap"
            ]
        )
        x_col, y_col, size_col, window_size = None, None, None, None

        if chart_type == "Histogram":
            nums = numeric_cols(df)
            x_col = st.sidebar.selectbox("Select numeric column", nums)

        elif chart_type in ["Pie", "Donut", "Countplot"]:
            cats = categorical_cols(df)
            x_col = st.sidebar.selectbox("Select categorical column", cats)

        elif chart_type == "Treemap":
            cats = categorical_cols(df)
            nums = numeric_cols(df)
            x_col = st.sidebar.selectbox("Select category column", cats)
            y_col = st.sidebar.selectbox("Select numeric size column", [None] + nums)

        elif chart_type in ["Bar", "Line", "Scatter", "Boxplot", "Area"]:
            x_col = st.sidebar.selectbox("Select X-axis column", df.columns)
            y_col = st.sidebar.selectbox("Select Y-axis column", df.columns)

        elif chart_type == "Bubble":
            x_col = st.sidebar.selectbox("Select X-axis column", df.columns)
            y_col = st.sidebar.selectbox("Select Y-axis column", df.columns)
            size_col = st.sidebar.selectbox("Select Size column", numeric_cols(df))

        elif chart_type == "Rolling Average":
            x_col = st.sidebar.selectbox("Select X-axis column", df.columns)
            y_col = st.sidebar.selectbox("Select Y-axis column", df.columns)
            window_size = st.sidebar.slider("Select Rolling Window Size", 2, 50, 5)

        show_values = st.sidebar.checkbox("Show values on chart", value=False)

        if st.sidebar.button("Generate Chart"):
            valid, error_msg = validate_chart_type(df, chart_type, x_col, y_col)
            if not valid:
                st.error(error_msg)
            else:
                plot_chart(df, chart_type, x_col, y_col, show_values, window=window_size or 3)

    # ---------- AI Data Assistant ----------
    st.subheader(" Got a doubt? Ask your Data Assistant!")
    user_question = st.text_area("Ask something about your dataset:")
    if st.button("Ask Assistant"):
        if user_question.strip():
            with st.spinner("Thinking..."):
                answer = ask_model(user_question, df)
            st.success("Assistant Reply:")
            st.write(answer)
        else:
            st.warning("Please type a question first.")

    # ---------- Download Cleaned CSV ----------
    csv_file = df.to_csv(index=False)
    st.download_button(
        label="💾 Download Cleaned CSV",
        data=csv_file,
        file_name='cleaned_data.csv',
        mime='text/csv'
    )

    # ===========================
    #  Model Building Section
    # ===========================
    st.markdown("---")
    st.header(" Automated Model Building & Evaluation")

    data_to_use = st.session_state['cleaned_df']
    available_columns = data_to_use.columns.tolist()

    target_col = st.selectbox(" Select the target/output column", available_columns)

    if st.button(" Run Auto Model Builder"):
        with st.spinner("Building and evaluating models..."):
            try:
                X = data_to_use.drop(columns=[target_col])
                y = data_to_use[target_col]

                task = detect_task_type(y)
                st.info(f"**Detected Task Type:** {task.capitalize()}")

                results_df, pipelines = evaluate_models(X, y, task)

                st.subheader("📊 Model Performance Comparison")
                st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))

                plot_results(results_df, task, output_path="model_comparison.png")
                st.image("model_comparison.png", caption="Model Comparison Chart")

            except Exception as e:
                st.error(f"Error during model building: {e}")

else:
    st.warning("Please upload a dataset to begin.")
