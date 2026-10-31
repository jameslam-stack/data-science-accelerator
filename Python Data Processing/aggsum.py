import pandas as pd
from IPython.display import display

from activity import GuidedActivity


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
        "sales": [
            1200,
            850,
            1500,
            400,
            950,
            600,
        ],
    }
)


def show_dataset(namespace):
    print("Dataset available as: sales_data")
    display(namespace["sales_data"])


def validate_grouped_data(namespace):
    if "grouped_data" not in namespace:
        return False

    result = namespace["grouped_data"]

    if not isinstance(
        result,
        pd.core.groupby.generic.DataFrameGroupBy,
    ):
        return False

    expected_categories = set(
        namespace["sales_data"]["category"].unique()
    )

    actual_categories = set(result.groups.keys())

    return actual_categories == expected_categories


def validate_category_summary(namespace):
    if "category_summary" not in namespace:
        return False

    result = namespace["category_summary"]

    expected = (
        namespace["sales_data"]
        .groupby("category", as_index=False)
        .aggregate(total_sales=("sales", "sum"))
    )

    return (
        isinstance(result, pd.DataFrame)
        and result.sort_values("category")
        .reset_index(drop=True)
        .equals(
            expected.sort_values("category")
            .reset_index(drop=True)
        )
    )


def validate_summary_exists(namespace):
    return "category_summary" in namespace


steps = [
    {
        "title": "Step 1: Group the data by category",
        "instruction": (
            "Create a variable called grouped_data by grouping "
            "sales_data using the category column. Set "
            "as_index to False."
        ),
        "placeholder": (
            "your_grouped_data = your_dataframe.groupby(\n"
            '    "your_category_column",\n'
            "    as_index=False,\n"
            ")"
        ),
        "validator": validate_grouped_data,
    },
    {
        "title": "Step 2: Calculate the sum for each group",
        "instruction": (
            "Create a variable called category_summary. Use "
            "aggregate to calculate the sum of the sales column "
            "for each category. Name the result total_sales."
        ),
        "placeholder": (
            "your_summary = your_grouped_data.aggregate(\n"
            '    new_column=("your_value_column", "function")\n'
            ")"
        ),
        "validator": validate_category_summary,
    },
    {
        "title": "Step 3: Display the summary",
        "instruction": (
            "Display category_summary to view the total sales "
            "for each category."
        ),
        "placeholder": "display(your_summary)",
        "validator": validate_summary_exists,
    },
]


def create_activity():
    return GuidedActivity(
        title="Aggregating sales by category",
        introduction=(
            "Use groupby and aggregate to summarise the dataset. "
            "You will group the records by product category and "
            "calculate the total sales for each group."
        ),
        steps=steps,
        namespace={
            "pd": pd,
            "sales_data": sales_data,
        },
        preview=show_dataset,
    )