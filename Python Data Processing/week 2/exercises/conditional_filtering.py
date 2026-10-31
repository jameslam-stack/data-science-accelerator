import pandas as pd
from IPython.display import display

from activity import GuidedActivity


employee_data = pd.DataFrame(
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
        "salary": [
            52000,
            61000,
            48000,
            70000,
            45000,
            55000,
        ],
    }
)


def show_dataset(namespace):
    print("Dataset available as: employee_data")
    display(namespace["employee_data"])


def validate_sales_department(namespace):
    if "sales_employees" not in namespace:
        return False

    result = namespace["sales_employees"]

    expected = namespace["employee_data"][
        namespace["employee_data"]["department"] == "Sales"
    ]

    return (
        isinstance(result, pd.DataFrame)
        and result.reset_index(drop=True).equals(
            expected.reset_index(drop=True)
        )
    )


def validate_high_salary(namespace):
    if "high_salary" not in namespace:
        return False

    result = namespace["high_salary"]

    expected = namespace["employee_data"][
        namespace["employee_data"]["salary"] > 50000
    ]

    return (
        isinstance(result, pd.DataFrame)
        and result.reset_index(drop=True).equals(
            expected.reset_index(drop=True)
        )
    )


def validate_hr_high_salary(namespace):
    if "hr_high_salary" not in namespace:
        return False

    result = namespace["hr_high_salary"]

    expected = namespace["employee_data"][
        (namespace["employee_data"]["department"] == "HR")
        & (namespace["employee_data"]["salary"] > 50000)
    ]

    return (
        isinstance(result, pd.DataFrame)
        and result.reset_index(drop=True).equals(
            expected.reset_index(drop=True)
        )
    )


steps = [
    {
        "title": "Step 1: Filter rows by department",
        "instruction": (
            "Create a variable called sales_employees that contains "
            "only employees in the Sales department."
        ),
        "placeholder": (
            'your_result = your_dataframe['
            'your_dataframe["column"] == "value"]'
        ),
        "validator": validate_sales_department,
    },
    {
        "title": "Step 2: Filter rows using a numeric condition",
        "instruction": (
            "Create a variable called high_salary that contains only "
            "employees with a salary greater than 50000."
        ),
        "placeholder": (
            'your_result = your_dataframe['
            'your_dataframe["column"] > value]'
        ),
        "validator": validate_high_salary,
    },
    {
        "title": "Step 3: Filter using multiple conditions",
        "instruction": (
            "Create a variable called hr_high_salary that contains "
            "employees who work in HR and have a salary greater than "
            "50000."
        ),
        "placeholder": (
            "your_result = your_dataframe[\n"
            '    (your_dataframe["column1"] == "value")\n'
            "    &\n"
            '    (your_dataframe["column2"] > value)\n'
            "]"
        ),
        "validator": validate_hr_high_salary,
    },
]


def create_activity():
    return GuidedActivity(
        title="Conditional filtering with pandas",
        introduction=(
            "Use boolean conditions to select only the rows you need. "
            "You will filter data using equality, comparison operators, "
            "and multiple conditions."
        ),
        steps=steps,
        namespace={
            "pd": pd,
            "employee_data": employee_data,
        },
        preview=show_dataset,
    )