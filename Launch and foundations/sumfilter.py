from activity import GuidedActivity
import pandas as pd


sales_data = pd.DataFrame(
    {
        "region": ["North", "South", "North", "West", "North"],
        "sales": [1200, 950, 1500, 1100, 800],
    }
)


def validate_filtered_data(namespace):
    if "filtered_data" not in namespace:
        return False

    filtered_data = namespace["filtered_data"]

    if not isinstance(filtered_data, pd.DataFrame):
        return False

    expected_data = sales_data[sales_data["region"] == "North"]

    return filtered_data.reset_index(drop=True).equals(
        expected_data.reset_index(drop=True)
    )


def validate_total(namespace):
    return (
        "total" in namespace
        and namespace["total"]
        == sales_data.loc[
            sales_data["region"] == "North",
            "sales",
        ].sum()
    )


def validate_total_exists(namespace):
    return "total" in namespace


steps = [
    {
        "title": "Step 1: Filter the data",
        "instruction": (
            "Create a variable called filtered_data containing "
            "only the rows where region is North."
        ),
        "placeholder": (
            'filtered_data = sales_data['
            'sales_data["region"] == "your_condition"]'
        ),
        "validator": validate_filtered_data,
    },
    {
        "title": "Step 2: Calculate the filtered total",
        "instruction": (
            "Create a variable called total by summing the sales "
            "column in filtered_data."
        ),
        "placeholder": (
            'total = filtered_data["your_column"].sum()'
        ),
        "validator": validate_total,
    },
    {
        "title": "Step 3: Display the result",
        "instruction": "Print the value stored in total.",
        "placeholder": "print(total)",
        "validator": validate_total_exists,
    },
]


def create_activity():
    return GuidedActivity(
        title="Summing data with conditional filtering",
        introduction=(
            "Use conditional filtering to select sales records "
            "from the North region, then calculate their total. "
            "Variables created in one step remain available in "
            "later steps."
        ),
        steps=steps,
        namespace={
            "pd": pd,
            "sales_data": sales_data,
        },
    )