from activity import GuidedActivity
import pandas as pd


sales_data = pd.DataFrame(
    {
        "category": [
            "Technology",
            "Furniture",
            "Technology",
            "Office Supplies",
            "Furniture",
            "Office Supplies",
        ],
        "sales": [1200, 850, 1500, 400, 950, 600],
    }
)


def validate_grouped_data(namespace):
    if "grouped_data" not in namespace:
        return False

    try:
        groups = list(namespace["grouped_data"])
    except (TypeError, AttributeError):
        return False

    group_names = []

    for group_name, group_data in groups:
        if isinstance(group_name, tuple):
            return False

        if not (group_data["category"] == group_name).all():
            return False

        group_names.append(group_name)

    expected_categories = set(sales_data["category"].unique())

    return set(group_names) == expected_categories


def validate_category_summary(namespace):
    if "category_summary" not in namespace:
        return False

    category_summary = namespace["category_summary"]

    if not isinstance(category_summary, pd.DataFrame):
        return False

    expected_summary = (
        sales_data.groupby("category", as_index=False)
        .aggregate(total_sales=("sales", "sum"))
        .sort_values("category")
        .reset_index(drop=True)
    )

    actual_summary = (
        category_summary
        .sort_values("category")
        .reset_index(drop=True)
    )

    return actual_summary.equals(expected_summary)


def validate_summary_exists(namespace):
    return "category_summary" in namespace


steps = [
    {
        "title": "Step 1: Group the data",
        "instruction": (
            "Create a variable called grouped_data by grouping "
            "sales_data using the category column."
        ),
        "placeholder": (
            'grouped_data = sales_data.groupby('
            '"your_category_column", as_index=False)'
        ),
        "validator": validate_grouped_data,
    },
    {
        "title": "Step 2: Aggregate the sales",
        "instruction": (
            "Create a variable called category_summary. Use "
            "aggregate to calculate the total sales for each category. "
            "Name the new column total_sales."
        ),
        "placeholder": (
            "category_summary = grouped_data.aggregate("
            'total_sales=("your_value_column", "your_function"))'
        ),
        "validator": validate_category_summary,
    },
    {
        "title": "Step 3: Display the summary",
        "instruction": (
            "Print category_summary to display the total sales "
            "for each category."
        ),
        "placeholder": "print(category_summary)",
        "validator": validate_summary_exists,
    },
]


def create_activity():
    return GuidedActivity(
        title="Aggregating data by category",
        introduction=(
            "Group the sales records by product category, then use "
            "an aggregate function to calculate the total sales for "
            "each group. Variables created in one step remain "
            "available in later steps."
        ),
        steps=steps,
        namespace={
            "pd": pd,
            "sales_data": sales_data,
        },
    )