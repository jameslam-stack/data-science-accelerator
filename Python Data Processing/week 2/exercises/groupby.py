import pandas as pd
from IPython.display import display

from activity import GuidedActivity


sales_data = pd.DataFrame(
    {
        "department": [
            "Sales",
            "Finance",
            "Sales",
            "Finance",
            "HR",
            "HR",
        ],
        "employee": [
            "Aisha",
            "Ben",
            "Chloe",
            "Daniel",
            "Eva",
            "Farah",
        ],
        "sales": [
            120,
            80,
            175,
            95,
            60,
            75,
        ],
    }
)


def show_dataset(namespace):
    print("Dataset available as: sales_data")
    display(namespace["sales_data"])


def validate_grouped_sales(namespace):
    if "department_sales" not in namespace:
        return False

    result = namespace["department_sales"]

    expected = (
        namespace["sales_data"]
        .groupby("department")["sales"]
        .sum()
    )

    return (
        isinstance(result, pd.Series)
        and result.sort_index().equals(expected.sort_index())
    )


def validate_grouped_mean(namespace):
    if "average_sales" not in namespace:
        return False

    result = namespace["average_sales"]

    expected = (
        namespace["sales_data"]
        .groupby("department")["sales"]
        .mean()
    )

    return (
        isinstance(result, pd.Series)
        and result.sort_index().equals(expected.sort_index())
    )


def validate_grouped_dataframe(namespace):
    if "department_summary" not in namespace:
        return False

    result = namespace["department_summary"]

    expected = (
        namespace["sales_data"]
        .groupby("department", as_index=False)["sales"]
        .sum()
    )

    return (
        isinstance(result, pd.DataFrame)
        and result.sort_values("department")
        .reset_index(drop=True)
        .equals(
            expected.sort_values("department")
            .reset_index(drop=True)
        )
    )


steps = [
    {
        "title": "Step 1: Calculate total sales by department",
        "instruction": (
            "Create a variable called department_sales that groups "
            "sales_data by department and calculates the total sales "
            "for each department."
        ),
        "placeholder": (
            'your_result = (\n'
            '    your_dataframe\n'
            '    .groupby("your_group_column")["your_value_column"]\n'
            '    .sum()\n'
            ')'
        ),
        "validator": validate_grouped_sales,
    },
    {
        "title": "Step 2: Calculate average sales by department",
        "instruction": (
            "Create a variable called average_sales that groups "
            "sales_data by department and calculates the mean sales "
            "for each department."
        ),
        "placeholder": (
            'your_result = (\n'
            '    your_dataframe\n'
            '    .groupby("your_group_column")["your_value_column"]\n'
            '    .mean()\n'
            ')'
        ),
        "validator": validate_grouped_mean,
    },
    {
        "title": "Step 3: Return the grouped result as a DataFrame",
        "instruction": (
            "Create a variable called department_summary that groups "
            "sales_data by department, calculates total sales and keeps "
            "department as a normal column."
        ),
        "placeholder": (
            "your_result = (\n"
            "    your_dataframe\n"
            '    .groupby("your_group_column", as_index=False)\n'
            '    ["your_value_column"]\n'
            "    .sum()\n"
            ")"
        ),
        "validator": validate_grouped_dataframe,
    },
]

def create_activity():
    return GuidedActivity(
        title="Grouping data with pandas",
        introduction=(
            "Use groupby to divide rows into department groups and "
            "calculate summary values for each group."
        ),
        steps=steps,
        namespace={
            "pd": pd,
            "sales_data": sales_data,
        },
        preview=show_dataset,
    )