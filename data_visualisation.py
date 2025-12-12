# data_visualisation.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import streamlit as st


def validate_chart_type(df, chart_type, x_col=None, y_col=None):
    """
    Validates if the selected chart is compatible with the chosen columns.
    """
    try:
        # Charts requiring numeric columns
        if chart_type in ["Histogram", "Boxplot"]:
            if not pd.api.types.is_numeric_dtype(df[x_col]):
                return False, f"❌ {chart_type} requires a numeric column for {x_col}."

        if chart_type in ["Line", "Scatter", "Bar", "Area", "Bubble", "Rolling Average"]:
            if y_col is None or not pd.api.types.is_numeric_dtype(df[y_col]):
                return False, f"❌ {chart_type} requires a numeric y-axis column."

        # Charts requiring categorical only (x_col)
        if chart_type in ["Pie", "Donut", "Countplot", "Treemap"]:
            if not (pd.api.types.is_object_dtype(df[x_col]) or pd.api.types.is_categorical_dtype(df[x_col])):
                return False, f"❌ {chart_type} requires a categorical column for {x_col}."

        return True, None

    except Exception as e:
        return False, f"⚠️ Validation Error: {e}"


def plot_chart(df, chart_type, x_col=None, y_col=None, show_values=False, window=3):
    """
    Plots a chart based on user selection.
    Supports optional value annotations on charts.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    try:
        if chart_type == "Histogram":
            counts, bins, patches = ax.hist(df[x_col].dropna(), bins=10, edgecolor="black")
            ax.set_title(f"Histogram of {x_col}")
            if show_values:
                for c, b in zip(counts, bins):
                    if c > 0:
                        ax.text(b, c, str(int(c)), ha="center", va="bottom")

        elif chart_type == "Bar":
            sns.barplot(x=df[x_col], y=df[y_col], ax=ax, estimator=sum)
            ax.set_title(f"Bar Chart of {x_col} vs {y_col}")
            if show_values:
                for p in ax.patches:
                    ax.annotate(str(round(p.get_height(), 2)),
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha="center", va="bottom")

        elif chart_type == "Line":
            sns.lineplot(x=df[x_col], y=df[y_col], ax=ax, marker="o")
            ax.set_title(f"Line Chart of {x_col} vs {y_col}")
            if show_values:
                for x, y in zip(df[x_col], df[y_col]):
                    ax.text(x, y, str(y), ha="center", va="bottom")

        elif chart_type == "Area":
            ax.fill_between(df[x_col], df[y_col], alpha=0.5)
            ax.plot(df[x_col], df[y_col], marker="o")
            ax.set_title(f"Area Chart of {x_col} vs {y_col}")

        elif chart_type == "Scatter":
            sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
            ax.set_title(f"Scatter Plot of {x_col} vs {y_col}")
            if show_values:
                for i in range(len(df)):
                    ax.text(df[x_col].iloc[i], df[y_col].iloc[i],
                            f"({df[x_col].iloc[i]}, {df[y_col].iloc[i]})",
                            fontsize=8, ha="left")

        elif chart_type == "Bubble":
            if "size" in df.columns:
                sns.scatterplot(x=df[x_col], y=df[y_col], size=df["size"], ax=ax, sizes=(20, 200))
                ax.set_title(f"Bubble Chart of {x_col} vs {y_col} (Size: size column)")
            else:
                st.warning("⚠️ Bubble chart requires a 'size' column in the dataset.")

        elif chart_type == "Pie":
            counts = df[x_col].value_counts()
            wedges, texts, autotexts = ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
            ax.set_title(f"Pie Chart of {x_col}")
            if show_values:
                labels_with_counts = [f"{label}: {val}" for label, val in counts.items()]
                ax.legend(wedges, labels_with_counts, title="Counts",
                          loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        elif chart_type == "Donut":
            counts = df[x_col].value_counts()
            wedges, texts, autotexts = ax.pie(
                counts, labels=counts.index, autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(width=0.3)  # donut effect
            )
            ax.set_title(f"Donut Chart of {x_col}")
            if show_values:
                labels_with_counts = [f"{label}: {val}" for label, val in counts.items()]
                ax.legend(wedges, labels_with_counts, title="Counts",
                          loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        elif chart_type == "Boxplot":
            sns.boxplot(x=df[x_col], ax=ax)
            ax.set_title(f"Boxplot of {x_col}")

        elif chart_type == "Countplot":
            sns.countplot(x=df[x_col], ax=ax)
            ax.set_title(f"Countplot of {x_col}")
            if show_values:
                for p in ax.patches:
                    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha="center", va="bottom")

        elif chart_type == "Treemap":
            counts = df[x_col].value_counts()
            squarify.plot(sizes=counts.values, label=counts.index, alpha=0.7, ax=ax)
            ax.axis("off")
            ax.set_title(f"Treemap of {x_col}")

        elif chart_type == "Rolling Average":
            if pd.api.types.is_numeric_dtype(df[y_col]):
                df["rolling_avg"] = df[y_col].rolling(window=window).mean()
                sns.lineplot(x=df[x_col], y=df["rolling_avg"], ax=ax, marker="o", label=f"{window}-period rolling avg")
                ax.set_title(f"Rolling Average ({window}) of {y_col} vs {x_col}")
            else:
                st.warning("⚠️ Rolling Average requires numeric y column.")

        elif chart_type == "Correlation Heatmap":
            corr = df.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")

        else:
            st.warning("⚠️ Unknown chart type selected.")
            return None

        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error plotting chart: {e}")
