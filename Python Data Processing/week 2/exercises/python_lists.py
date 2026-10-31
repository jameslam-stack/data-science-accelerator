from activity import GuidedActivity


def validate_numbers(namespace):
    return (
        "numbers" in namespace
        and isinstance(namespace["numbers"], list)
        and len(namespace["numbers"]) >= 3
    )


def validate_total(namespace):
    return (
        "numbers" in namespace
        and "total" in namespace
        and namespace["total"] == sum(namespace["numbers"])
    )


def validate_total_exists(namespace):
    return "total" in namespace


steps = [
    {
        "title": "Step 1: Create a list",
        "instruction": (
            "Create a variable called numbers containing "
            "at least three values."
        ),
        "placeholder": (
            "your_list_name = "
            "[your_value_1, your_value_2, your_value_3]"
        ),
        "validator": validate_numbers,
    },
    {
        "title": "Step 2: Calculate the total",
        "instruction": (
            "Create a variable called total using the numbers list."
        ),
        "placeholder": (
            "your_result_name = sum(your_list_name)"
        ),
        "validator": validate_total,
    },
    {
        "title": "Step 3: Display the result",
        "instruction": "Print the value stored in total.",
        "placeholder": (
            "print(your_result_name)"
        ),
        "validator": validate_total_exists,
    },
]

def create_activity():
    return GuidedActivity(
        title="Working with Python lists",
        introduction=(
            "Complete the steps below. Variables created in one "
            "step remain available in later steps."
        ),
        steps=steps,
        namespace={},
    )