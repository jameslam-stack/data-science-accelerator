def preview_head(data, n=5):
    """
    Display the first n rows of a dataset.
    Ask the user what type of ML problem they are working on.
    Then ask them to select the target column from the DataFrame columns.
    """
    from IPython.display import display
    import ipywidgets as widgets

    display(data.head(n))

    question_1 = widgets.HTML("<b>What problem type is your project?</b>")

    regression_button = widgets.Button(description="Regression")
    time_series_button = widgets.Button(description="Time series")
    classification_button = widgets.Button(description="Classification")

    output = widgets.Output()

    def ask_target_column(problem_type):
        with output:
            output.clear_output()

            display(widgets.HTML(f"<b>Problem type selected:</b> {problem_type}"))

            question_2 = widgets.HTML("""
            <b>What is the target column?</b><br>
            <span style="color: gray;">Hint: this is the column you are trying to predict.</span>
            """)

            target_output = widgets.Output()
            column_buttons = []

            for column in data.columns:
                button = widgets.Button(description=str(column))

                def on_column_clicked(button, selected_column=column):
                    with target_output:
                        target_output.clear_output()

                        feature_columns = [
                            col for col in data.columns
                            if col != selected_column
                        ]

                        if problem_type in ["Regression", "Classification"]:
                            display(widgets.HTML(
                                f"""
                                <b>Problem statement:</b><br>
                                You are using <b>{', '.join(map(str, feature_columns))}</b>
                                to predict <b>{selected_column}</b>.
                                """
                            ))
                        else:
                            display(widgets.HTML(
                                f"<b>Target column selected:</b> {selected_column}"
                            ))

                button.on_click(on_column_clicked)
                column_buttons.append(button)

            display(question_2)
            display(widgets.HBox(column_buttons))
            display(target_output)

    def on_regression_clicked(button):
        ask_target_column("Regression")

    def on_time_series_clicked(button):
        ask_target_column("Time series")

    def on_classification_clicked(button):
        ask_target_column("Classification")

    regression_button.on_click(on_regression_clicked)
    time_series_button.on_click(on_time_series_clicked)
    classification_button.on_click(on_classification_clicked)

    display(question_1)
    display(widgets.HBox([
        regression_button,
        time_series_button,
        classification_button
    ]))
    display(output)

def data_cleaning_classification():
    """
    Guided data cleaning checklist.

    Check 1:
    - Have you had a brief look at your data using .head()?

    Check 2:
    - Do the column headers make sense?

    Check 3:
    - Have you checked for null values?

    Check 4:
    - Have you checked data types using .info()?
    """
    from IPython.display import display
    import ipywidgets as widgets

    main_output = widgets.Output()

    def ask_data_type_check():
        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked the data types using <code>.info()</code>?</b><br>
            <span style="color: gray;">
            This helps you check whether each column is stored in the right format.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_continue_button():
                continue_button = widgets.Button(
                    description="Next Check",
                    button_style="info"
                )

                def on_continue_clicked(button):
                    ask_duplicate_check()

                continue_button.on_click(on_continue_clicked)
                display(continue_button)

            def show_dtype_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Example Considerations:</b><br><br>

                    <b>1. Are numerical columns stored as numbers?</b><br>
                    For example, a column like <code>price</code>, <code>age</code>, or <code>sales_amount</code> should usually be <code>int</code> or <code>float</code>, not <code>object</code>.<br><br>

                    <b>2. Are date columns stored as dates?</b><br>
                    For example, a column like <code>purchase_date</code> should usually be converted to <code>datetime</code>.<br><br>

                    <b>3. Are category columns stored appropriately?</b><br>
                    For example, a column like <code>region</code>, <code>product_type</code>, or <code>customer_segment</code> may appear as <code>object</code>, which can be fine, but it is worth checking.<br><br>

                    <b>4. Are true/false columns stored as booleans?</b><br>
                    For example, a column like <code>is_member</code> or <code>has_discount</code> should usually be <code>bool</code>, not text like <code>"Yes"</code> and <code>"No"</code>.<br><br>

                    <b>5. If this is a classification problem, is your target column binary?</b><br>
                    For example, if your target column is <code>purchased</code>, <code>churned</code>, or <code>approved</code>, does it only contain two possible outcomes such as <code>Yes/No</code>, <code>True/False</code>, or <code>0/1</code>?<br>
                    If it has more than two outcomes, it may be a multi-class classification problem instead.
                    """))

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Use this to inspect the data types:
                    """))

                    code = """
    your_data.info()
    """
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>Example Code:</b><br>
                    Use these only where relevant to your dataset.
                    """))

                    example_code = """
    # Convert a column to a number
    your_data["price"] = your_data["price"].astype(float)

    # Convert a column to a date
    your_data["purchase_date"] = pd.to_datetime(your_data["purchase_date"])

    # Convert a yes/no column to True/False
    your_data["is_member"] = your_data["is_member"].map({
        "Yes": True,
        "No": False
    })

    # Convert a column to category
    your_data["region"] = your_data["region"].astype("category")

    # Check the unique values in your target column
    your_data["target_column"].unique()

    # Count how often each target value appears
    your_data["target_column"].value_counts()

    # Example: if your target column is called "churned"
    your_data["churned"].value_counts()
    """
                    display(widgets.HTML(f"<pre><code>{example_code}</code></pre>"))

                    display(widgets.HTML("""
                    <span style="color: gray;">
                    Hint: replace <code>your_data</code> with whatever you called your DataFrame, and replace the column names with columns from your own dataset.
                    </span>
                    """))

                    show_continue_button()

            def on_yes_clicked(button):
                show_dtype_guidance()

            def on_no_clicked(button):
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Run this first to inspect your column data types:
                    """))

                    code = """
    your_data.info()
    """
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    <span style="color: gray;">
                    Hint: replace <code>your_data</code> with whatever you called your DataFrame.
                    </span>
                    """))

                    continue_button = widgets.Button(
                        description="Examples",
                        button_style="info"
                    )

                    def on_continue_clicked(button):
                        show_dtype_guidance()

                    continue_button.on_click(on_continue_clicked)
                    display(continue_button)

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)


    def ask_null_check():
        with main_output:
            main_output.clear_output()

            question = widgets.HTML("<b>Have you checked for null values?</b>")

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_continue_button():
                continue_button = widgets.Button(
                    description="Next Check",
                    button_style="info"
                )

                def on_continue_clicked(button):
                    ask_data_type_check()

                continue_button.on_click(on_continue_clicked)
                display(continue_button)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Good.</b> The next step is to assess whether the number of null values is small or large relative to the size of your dataset.
                    """))

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    This shows the percentage of missing values in each column:
                    """))

                    code = """
your_data.isnull().mean() * 100
"""
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    <span style="color: gray;">
                    Hint: replace <code>your_data</code> with whatever you called your DataFrame.
                    </span>
                    """))

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    A column with a very small percentage of nulls may be easy to clean.
                    A column with a large percentage of nulls may need closer investigation before you decide whether to fill, remove, or keep it.
                    If there are missing values in your Target column you will have to delete these rows 
                    """))

                    display(widgets.HTML("""
                    <b>Example scenario:</b><br>
                    <ul>
                        <li>If a column has only a small amount of missing data, such as under <b>5%</b>, you might be able to fill or remove those rows depending on the context.</li>

                        <li>If a column has an excessive amount of missing data, for example over <b>30%</b> of the rows are null, treat this as a warning sign.</li>

                        <li>Before dropping or filling the column, ask whether the missingness itself tells you something useful, whether the column is important to the project, and whether changing it could distort the analysis.</li>

                        <li>Be careful about dropping rows or columns too quickly. Dropping data can reduce the size of your dataset, remove useful patterns, or accidentally remove a specific group of records more than others.</li>

                        <li>For example, if <code>income</code> is missing for <b>40%</b> of customers, dropping every row with a missing income value could remove a large part of your dataset.</li>

                        <li>Filling those values with the average income could also create a misleading dataset. You may need to investigate why the values are missing, compare rows with and without missing values, or consider whether the column should be excluded.</li>

                        <li>The exact threshold depends on the dataset, the project goal, and how important the column is.</li>
                    </ul>
                    """))

                    

                    show_continue_button()

            def on_no_clicked(button):
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Run this to check how many null values are in each column:
                    """))

                    code = """
your_data.isnull().sum()
"""
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    <span style="color: gray;">
                    Hint: replace <code>your_data</code> with whatever you called your DataFrame.
                    </span>
                    """))

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    If your DataFrame is called <code>sales_data</code>, you would write:
                    """))

                    example_code = """
sales_data.isnull().sum()
"""
                    display(widgets.HTML(f"<pre><code>{example_code}</code></pre>"))

                    show_continue_button()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_column_header_check():
        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Do your column headers make sense?</b><br>
            <span style="color: gray;">
            For example, are they clear, readable, and easy to use in code?
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def on_yes_clicked(button):
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Great.</b> Let’s move on to the next check.
                    """))

                    continue_button = widgets.Button(
                        description="Next Check",
                        button_style="info"
                    )

                    def on_continue_clicked(button):
                        ask_null_check()

                    continue_button.on_click(on_continue_clicked)
                    display(continue_button)
                

            def on_no_clicked(button):
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    You can rename columns using <code>.rename()</code>.
                    """))

                    code = """
your_data = your_data.rename(columns={
    "old_column_name": "new_column_name"
})
"""
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    <span style="color: gray;">
                    Hint: replace <code>your_data</code> with whatever you called your DataFrame.
                    </span>
                    """))

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    """))

                    example_code = """
your_data = your_data.rename(columns={
    "Customer Name": "customer_name",
    "Sale Amount (£)": "sale_amount",
    "Date of Purchase": "purchase_date"
})
"""
                    display(widgets.HTML(f"<pre><code>{example_code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>Why this helps:</b><br>
                    Clear column names make your code easier to write, easier to read, and less likely to break.
                    """))

                    continue_button = widgets.Button(
                        description="Next Check",
                        button_style="info"
                    )

                    def on_continue_clicked(button):
                        ask_null_check()

                    continue_button.on_click(on_continue_clicked)
                    display(continue_button)

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_head_check():
        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you had a brief look at your data?</b><br>
            <span style="color: gray;">
            A quick look at the first few rows can help you understand what the dataset contains before cleaning it.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_head_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Use <code>.head()</code> to preview the first few rows:
                    """))

                    code = """
your_data.head()
"""
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>What to look for:</b><br><br>

                    <b>1. Do the columns look like the data you expected?</b><br>
                    For example, if this is a customer dataset, you might expect columns such as <code>customer_id</code>, <code>age</code>, <code>region</code>, or <code>purchased</code>.<br><br>

                    <b>2. Do the values look sensible?</b><br>
                    For example, ages should look like realistic ages, prices should look like numbers, and dates should look like dates.<br><br>

                    <b>3. Are there obvious missing or strange values?</b><br>
                    Look out for blanks, <code>None</code>, <code>NaN</code>, <code>?</code>, <code>unknown</code>, or values that do not belong.<br><br>

                    <b>4. Does each row seem to represent one observation?</b><br>
                    For example, one customer, one transaction, one product, one patient, or one time period.<br><br>

                    <b>5. Can you identify your likely target column?</b><br>
                    This is the column you are trying to predict, such as <code>price</code>, <code>churned</code>, <code>sales</code>, or <code>approved</code>.
                    """))

                    example_button = widgets.Button(
                        description="Show example interpretation",
                        button_style="info"
                    )

                    example_output = widgets.Output()

                    def on_example_clicked(button):
                        with example_output:
                            example_output.clear_output()

                            display(widgets.HTML("""
                            <b>Example interpretation of <code>.head()</code>:</b><br><br>

                            Imagine <code>your_data.head()</code> shows columns called
                            <code>age</code>, <code>income</code>, <code>region</code>, and <code>purchased</code>.<br><br>

                            A good first interpretation might be:<br><br>

                            <i>
                            Each row appears to represent one customer.
                            The columns <code>age</code>, <code>income</code>, and <code>region</code> look like possible input features.
                            The column <code>purchased</code> looks like the target column because it tells us whether the customer bought something.
                            The next checks should confirm whether the column names are clear, whether there are missing values, and whether the data types are correct.
                            </i>
                            """))

                    example_button.on_click(on_example_clicked)

                    display(example_button)
                    display(example_output)

                    continue_button = widgets.Button(
                        description="Next Check",
                        button_style="info"
                    )

                    def on_continue_clicked(button):
                        ask_column_header_check()

                    continue_button.on_click(on_continue_clicked)
                    display(continue_button)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Great.</b> Let’s move on to the next check.
                    """))

                    continue_button = widgets.Button(
                        description="Next Check",
                        button_style="info"
                    )

                    def on_continue_clicked(button):
                        ask_column_header_check()

                    continue_button.on_click(on_continue_clicked)
                    display(continue_button)

            def on_no_clicked(button):
                show_head_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    display(widgets.HTML("<b>Data cleaning checklist</b>"))
    display(main_output)

    ask_head_check()


    def ask_duplicate_check():
        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked for duplicate rows?</b><br>
            <span style="color: gray;">
            Duplicate rows can make a model look better than it really is, especially if the same record appears in both training and test data.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_continue_button():
                continue_button = widgets.Button(
                    description="Next Check",
                    button_style="info"
                )

                def on_continue_clicked(button):
                    ask_target_column_check()

                continue_button.on_click(on_continue_clicked)
                display(continue_button)

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Check how many duplicate rows are in the dataset:
                    """))

                    code = """
    your_data.duplicated().sum()
    """
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>Optional inspection:</b><br>
                    View the duplicated rows before deciding whether to remove them:
                    """))

                    example_code = """
    your_data[your_data.duplicated()]
    """
                    display(widgets.HTML(f"<pre><code>{example_code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>If the duplicates are accidental, they may need to be removed.</li>
                        <li>If the duplicates are valid repeated events, they may need to stay.</li>
                        <li>Do not drop duplicates automatically without checking what one row represents.</li>
                    </ul>
                    """))

                    show_continue_button()

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_target_column_check():
        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you identified the target column?</b><br>
            <span style="color: gray;">
            In a classification project, the target is the category or class you want the model to predict.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_continue_button():
                continue_button = widgets.Button(
                    description="Next Check",
                    button_style="info"
                )

                def on_continue_clicked(button):
                    ask_class_balance_check()

                continue_button.on_click(on_continue_clicked)
                display(continue_button)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Store the name of your target column in a variable:
                    """))

                    code = """
    target_column = "your_target_column"
    """
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    If you are predicting whether a customer purchased, the target might be:
                    """))

                    example_code = """
    target_column = "purchased"
    """
                    display(widgets.HTML(f"<pre><code>{example_code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    The target column should contain the outcome you want to predict, not an input feature.
                    """))

                    show_continue_button()

            def on_no_clicked(button):
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>How to identify it:</b><br>
                    Ask: <i>What am I trying to predict?</i>
                    """))

                    display(widgets.HTML("""
                    <b>Examples:</b><br>
                    <ul>
                        <li>Predicting churn → target might be <code>churned</code></li>
                        <li>Predicting loan approval → target might be <code>approved</code></li>
                        <li>Predicting fraud → target might be <code>is_fraud</code></li>
                        <li>Predicting purchase behaviour → target might be <code>purchased</code></li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Look at your column names again:
                    """))

                    code = """
    your_data.columns
    """
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    show_continue_button()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)


    def ask_class_balance_check():
        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked the class balance?</b><br>
            <span style="color: gray;">
            This shows whether one class appears much more often than another.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_continue_button():
                continue_button = widgets.Button(
                    description="Next Check",
                    button_style="info"
                )

                def on_continue_clicked(button):
                    ask_unique_values_check()

                continue_button.on_click(on_continue_clicked)
                display(continue_button)

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Count how many examples there are in each class:
                    """))

                    code = """
    your_data[target_column].value_counts()
    """
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>Percentage version:</b><br>
                    This is often easier to interpret:
                    """))

                    percent_code = """
    your_data[target_column].value_counts(normalize=True) * 100
    """
                    display(widgets.HTML(f"<pre><code>{percent_code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>If one class is much larger than the others, your dataset is imbalanced.</li>
                        <li>In an imbalanced dataset, accuracy can be misleading.</li>
                        <li>You may need to look at metrics such as precision, recall, F1 score, or ROC-AUC.</li>
                        <li>You should usually use <code>stratify=y</code> when splitting the data.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    If 95% of customers did not churn and only 5% did churn, a model could get high accuracy by mostly predicting the majority class.
                    """))

                    show_continue_button()

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_unique_values_check():
        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked the number of unique values in each column?</b><br>
            <span style="color: gray;">
            This helps you spot constant columns, ID columns, binary columns, and high-cardinality categorical columns.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_continue_button():
                continue_button = widgets.Button(
                    description="Next Check",
                    button_style="info"
                )

                def on_continue_clicked(button):
                    ask_summary_statistics_check()

                continue_button.on_click(on_continue_clicked)
                display(continue_button)

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Count the number of unique values in each column:
                    """))

                    code = """
    your_data.nunique().sort_values()
    """
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>Useful follow-up checks:</b><br>
                    Find columns with only one unique value:
                    """))

                    constant_code = """
    your_data.nunique()[your_data.nunique() <= 1]
    """
                    display(widgets.HTML(f"<pre><code>{constant_code}</code></pre>"))

                    display(widgets.HTML("""
                    Find columns where almost every row has a different value:
                    """))

                    id_code = """
    possible_id_columns = [
        column for column in your_data.columns
        if your_data[column].nunique() / len(your_data) > 0.95
    ]

    possible_id_columns
    """
                    display(widgets.HTML(f"<pre><code>{id_code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>Columns with one unique value usually do not help the model.</li>
                        <li>Columns with a different value in almost every row may be ID columns.</li>
                        <li>ID columns usually should not go directly into a model.</li>
                        <li>High-cardinality categorical columns may need careful encoding.</li>
                    </ul>
                    """))

                    show_continue_button()

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_summary_statistics_check():
        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked the summary statistics?</b><br>
            <span style="color: gray;">
            Summary statistics help you spot impossible values, unusual ranges, and extreme outliers.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_continue_button():
                continue_button = widgets.Button(
                    description="Next Check",
                    button_style="info"
                )

                def on_continue_clicked(button):
                    ask_leakage_check()

                continue_button.on_click(on_continue_clicked)
                display(continue_button)

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Check the numeric columns:
                    """))

                    code = """
    your_data.describe()
    """
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>For categorical columns:</b>
                    """))

                    object_code = """
    your_data.describe(include="object")
    """
                    display(widgets.HTML(f"<pre><code>{object_code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Impossible values, such as negative ages</li>
                        <li>Very large maximum values</li>
                        <li>Unexpected minimum values</li>
                        <li>Columns with very different scales</li>
                        <li>Messy or repeated categories</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    If <code>age</code> has a minimum of <code>-2</code> or a maximum of <code>300</code>, that column probably needs cleaning.
                    """))

                    show_continue_button()

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)




def data_eda_classification():
    import ipywidgets as widgets
    from IPython.display import display

    main_output = widgets.Output()

    def ask_target_distribution_check():
        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you explored the distribution of your target column?</b><br>
            <span style="color: gray;">
            In classification, this helps you understand how many examples belong to each class.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_continue_button():
                continue_button = widgets.Button(
                    description="Next Check",
                    button_style="info"
                )

                def on_continue_clicked(button):
                    ask_numeric_summary_check()

                continue_button.on_click(on_continue_clicked)
                display(continue_button)

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    First, set the name of your target column:
                    """))

                    target_code = """
target_column = "your_target_column"
"""
                    display(widgets.HTML(f"<pre><code>{target_code}</code></pre>"))

                    display(widgets.HTML("""
                    Count how many examples are in each class:
                    """))

                    code = """
your_data[target_column].value_counts()
"""
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    Percentage version:
                    """))

                    percent_code = """
your_data[target_column].value_counts(normalize=True) * 100
"""
                    display(widgets.HTML(f"<pre><code>{percent_code}</code></pre>"))

                    display(widgets.HTML("""
                    Optional visual:
                    """))

                    plot_code = """
your_data[target_column].value_counts().plot(kind="bar")
"""
                    display(widgets.HTML(f"<pre><code>{plot_code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>If the classes are fairly balanced, accuracy may be a reasonable starting metric.</li>
                        <li>If one class is much more common, accuracy may be misleading.</li>
                        <li>For imbalanced datasets, consider metrics such as precision, recall, F1 score, ROC-AUC, or PR-AUC.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    If 95% of customers did not churn and only 5% did churn, a model could get 95% accuracy by always predicting “did not churn”.
                    """))

                    show_continue_button()

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_numeric_summary_check():
        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you explored the numeric features?</b><br>
            <span style="color: gray;">
            This helps you understand the range, average, spread, and possible unusual values in numeric columns.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_continue_button():
                continue_button = widgets.Button(
                    description="Next Check",
                    button_style="info"
                )

                def on_continue_clicked(button):
                    ask_categorical_summary_check()

                continue_button.on_click(on_continue_clicked)
                display(continue_button)

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Select numeric columns:
                    """))

                    code = """
numeric_columns = your_data.select_dtypes(include="number").columns

your_data[numeric_columns].describe()
"""
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Very large or very small values</li>
                        <li>Unexpected minimums or maximums</li>
                        <li>Columns with very different scales</li>
                        <li>Possible outliers</li>
                        <li>Numeric columns that may actually be IDs</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    If <code>age</code> has a minimum of <code>-3</code> or a maximum of <code>300</code>, this may indicate a data quality issue.
                    """))

                    show_continue_button()

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_categorical_summary_check():
        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you explored the categorical features?</b><br>
            <span style="color: gray;">
            This helps you understand labels, categories, and columns that may need encoding before modelling.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_continue_button():
                continue_button = widgets.Button(
                    description="Next Check",
                    button_style="info"
                )

                def on_continue_clicked(button):
                    ask_numeric_vs_target_check()

                continue_button.on_click(on_continue_clicked)
                display(continue_button)

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Select categorical columns:
                    """))

                    code = """
categorical_columns = your_data.select_dtypes(include="object").columns

your_data[categorical_columns].describe()
"""
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    Check how many unique values each categorical column has:
                    """))

                    unique_code = """
your_data[categorical_columns].nunique().sort_values(ascending=False)
"""
                    display(widgets.HTML(f"<pre><code>{unique_code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Messy labels, such as <code>Yes</code>, <code>yes</code>, and <code>YES</code></li>
                        <li>Columns with lots of unique values</li>
                        <li>Rare categories</li>
                        <li>Text columns that may not be simple categories</li>
                        <li>Categorical columns that may actually be dates or numbers</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    A column like <code>region</code> may be useful. A column like <code>customer_id</code> may have too many unique values to use directly.
                    """))

                    show_continue_button()

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_numeric_vs_target_check():
        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you compared numeric features against the target?</b><br>
            <span style="color: gray;">
            This helps you see whether numeric values differ between classes.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_continue_button():
                continue_button = widgets.Button(
                    description="Next Check",
                    button_style="info"
                )

                def on_continue_clicked(button):
                    ask_categorical_vs_target_check()

                continue_button.on_click(on_continue_clicked)
                display(continue_button)

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Compare average numeric values for each class:
                    """))

                    code = """
numeric_columns = your_data.select_dtypes(include="number").columns

your_data.groupby(target_column)[numeric_columns].mean()
"""
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    Median version:
                    """))

                    median_code = """
your_data.groupby(target_column)[numeric_columns].median()
"""
                    display(widgets.HTML(f"<pre><code>{median_code}</code></pre>"))

                    display(widgets.HTML("""
                    Optional boxplot for one numeric column:
                    """))

                    plot_code = """
your_data.boxplot(column="numeric_column", by=target_column)
"""
                    display(widgets.HTML(f"<pre><code>{plot_code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>Look for numeric features that differ noticeably between classes.</li>
                        <li>Large differences may suggest useful predictive signal.</li>
                        <li>Very strong separation may also be worth checking for leakage.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    If customers who churn have much lower activity than customers who do not churn, activity may be an important feature.
                    """))

                    show_continue_button()

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_categorical_vs_target_check():
        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you compared categorical features against the target?</b><br>
            <span style="color: gray;">
            This helps you see whether some categories are more associated with one class than another.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_continue_button():
                continue_button = widgets.Button(
                    description="Next Check",
                    button_style="info"
                )

                def on_continue_clicked(button):
                    ask_outlier_check()

                continue_button.on_click(on_continue_clicked)
                display(continue_button)

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Use a crosstab to compare one categorical column with the target:
                    """))

                    code = """
pd.crosstab(
    your_data["category_column"],
    your_data[target_column],
    normalize="index"
)
"""
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    Example:
                    """))

                    example_code = """
pd.crosstab(
    your_data["region"],
    your_data[target_column],
    normalize="index"
)
"""
                    display(widgets.HTML(f"<pre><code>{example_code}</code></pre>"))

                    display(widgets.HTML("""
                    Optional count version:
                    """))

                    count_code = """
pd.crosstab(
    your_data["region"],
    your_data[target_column]
)
"""
                    display(widgets.HTML(f"<pre><code>{count_code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>Look for categories where the class percentages are noticeably different.</li>
                        <li>Be careful with very small categories, as they may look important by chance.</li>
                        <li>If a category perfectly predicts the target, check whether it is genuine signal or leakage.</li>
                    </ul>
                    """))

                    show_continue_button()

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_outlier_check():
        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you explored possible outliers?</b><br>
            <span style="color: gray;">
            Outliers are unusual values. They may be errors, rare cases, or useful signals.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_continue_button():
                continue_button = widgets.Button(
                    description="Next Check",
                    button_style="info"
                )

                def on_continue_clicked(button):
                    ask_correlation_check()

                continue_button.on_click(on_continue_clicked)
                display(continue_button)

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Use summary statistics to spot unusual ranges:
                    """))

                    code = """
numeric_columns = your_data.select_dtypes(include="number").columns

your_data[numeric_columns].describe()
"""
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    Optional boxplot:
                    """))

                    boxplot_code = """
your_data[numeric_columns].boxplot(figsize=(12, 6))
"""
                    display(widgets.HTML(f"<pre><code>{boxplot_code}</code></pre>"))

                    display(widgets.HTML("""
                    Check one column in more detail:
                    """))

                    one_col_code = """
your_data["numeric_column"].describe()
"""
                    display(widgets.HTML(f"<pre><code>{one_col_code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>Do not remove outliers automatically.</li>
                        <li>First ask whether the value is impossible, unlikely, or genuinely rare.</li>
                        <li>Outliers can sometimes be important predictive signals in classification.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    A very high transaction amount may be an error in one dataset, but a useful fraud signal in another.
                    """))

                    show_continue_button()

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_correlation_check():
        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked correlations between numeric features?</b><br>
            <span style="color: gray;">
            Correlation helps you see whether numeric features move together.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_continue_button():
                continue_button = widgets.Button(
                    description="Next Check",
                    button_style="info"
                )

                def on_continue_clicked(button):
                    ask_multicollinearity_check()

                continue_button.on_click(on_continue_clicked)
                display(continue_button)

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Create a correlation matrix for numeric columns:
                    """))

                    code = """
numeric_columns = your_data.select_dtypes(include="number").columns

your_data[numeric_columns].corr()
"""
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    Optional visual:
                    """))

                    visual_code = """
import matplotlib.pyplot as plt

plt.imshow(your_data[numeric_columns].corr())
plt.colorbar()
plt.show()
"""
                    display(widgets.HTML(f"<pre><code>{visual_code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>Correlation values close to <code>1</code> or <code>-1</code> show strong relationships.</li>
                        <li>Highly correlated features may be giving similar information to the model.</li>
                        <li>Correlation only captures linear relationships.</li>
                    </ul>
                    """))

                    show_continue_button()

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_multicollinearity_check():
        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked for multicollinearity?</b><br>
            <span style="color: gray;">
            Multicollinearity happens when input features contain very similar information.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_continue_button():
                continue_button = widgets.Button(
                    description="Next Check",
                    button_style="info"
                )

                def on_continue_clicked(button):
                    ask_eda_conclusion_check()

                continue_button.on_click(on_continue_clicked)
                display(continue_button)

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Find highly correlated numeric feature pairs:
                    """))

                    code = """
import numpy as np

numeric_columns = your_data.select_dtypes(include="number").columns

correlation_matrix = your_data[numeric_columns].corr().abs()

high_corr_pairs = (
    correlation_matrix
    .where(~np.eye(correlation_matrix.shape[0], dtype=bool))
    .stack()
    .sort_values(ascending=False)
)

high_corr_pairs[high_corr_pairs > 0.8]
"""
                    display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>Optional advanced check:</b><br>
                    Use VIF to assess multicollinearity more formally:
                    """))

                    vif_code = """
from statsmodels.stats.outliers_influence import variance_inflation_factor

numeric_data = your_data[numeric_columns].dropna()

vif_data = pd.DataFrame()
vif_data["feature"] = numeric_data.columns
vif_data["VIF"] = [
    variance_inflation_factor(numeric_data.values, i)
    for i in range(numeric_data.shape[1])
]

vif_data.sort_values("VIF", ascending=False)
"""
                    display(widgets.HTML(f"<pre><code>{vif_code}</code></pre>"))

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>This is especially important for Logistic Regression and other coefficient-based models.</li>
                        <li>Tree-based models are usually less sensitive to multicollinearity.</li>
                        <li>High multicollinearity can make interpretation harder.</li>
                        <li>Do not remove columns automatically. First decide whether the feature is useful, interpretable, and available at prediction time.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Simple guide:</b><br>
                    <ul>
                        <li><code>VIF ≈ 1</code>: low multicollinearity</li>
                        <li><code>VIF > 5</code>: possible concern</li>
                        <li><code>VIF > 10</code>: strong concern</li>
                    </ul>
                    """))

                    show_continue_button()

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_eda_conclusion_check():
        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you written down your EDA conclusions?</b><br>
            <span style="color: gray;">
            Before modelling, summarise what your exploration tells you about the classification problem.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_finish_button():
                finish_button = widgets.Button(
                    description="Finish EDA Checklist",
                    button_style="success"
                )

                def on_finish_clicked(button):
                    with output:
                        output.clear_output()

                        display(widgets.HTML("""
                        <b>EDA checklist complete.</b><br>
                        You have now explored the main patterns, risks, and modelling implications for a classification dataset.
                        """))

                finish_button.on_click(on_finish_clicked)
                display(finish_button)

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested EDA conclusion questions:</b><br>
                    <ul>
                        <li>Is the target balanced or imbalanced?</li>
                        <li>Which features seem most related to the target?</li>
                        <li>Are there any suspicious features that could cause leakage?</li>
                        <li>Are there outliers or unusual values that need further investigation?</li>
                        <li>Are any features strongly correlated with each other?</li>
                        <li>What metrics might be most appropriate for this classification problem?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example conclusion:</b><br>
                    <i>
                    The target is imbalanced, so accuracy may be misleading. Some numeric features appear to differ between classes, especially customer activity and account age. A few categorical features show different class rates, but small category sizes should be treated carefully. There are also some highly correlated numeric features, so interpretation may need caution if using Logistic Regression.
                    </i>
                    """))

                    show_finish_button()

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    display(widgets.HTML("<b>Classification EDA checklist</b>"))
    display(main_output)

    ask_target_distribution_check()



def classification_preprocessing():
    """
    Guided classification preprocessing checklist.

    Check 1:
    - Have you separated your features and target?

    Check 2:
    - Have you split into training and test sets?

    Check 3:
    - Have you checked class balance?

    Check 4:
    - Have you identified categorical features?

    Check 5:
    - Have you checked for ordinal categorical features?

    Check 6:
    - Have you checked for nominal categorical features?

    Check 7:
    - Have you checked whether numerical features need scaling?

    Check 8:
    - Have you checked for data leakage?

    Check 9:
    - Have you confirmed the final train/test shapes?
    """
    from IPython.display import display
    import ipywidgets as widgets

    main_output = widgets.Output()
    checklist_output = widgets.Output()

    checklist_items = {
        1: "Separate features and target",
        2: "Split into train and test sets",
        3: "Check class balance",
        4: "Identify categorical features",
        5: "Check ordinal categorical features",
        6: "Check nominal categorical features",
        7: "Check whether scaling is needed",
        8: "Check for data leakage",
        9: "Confirm final train/test shapes"
    }

    checklist_status = {
        1: "current",
        2: "pending",
        3: "pending",
        4: "pending",
        5: "pending",
        6: "pending",
        7: "pending",
        8: "pending",
        9: "pending"
    }

    def update_checklist():
        with checklist_output:
            checklist_output.clear_output()

            html = """
            <div style="
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 12px;
                background-color: #fafafa;
                width: 300px;
            ">
            <b>Classification preprocessing checklist</b><br><br>
            """

            for number, label in checklist_items.items():
                status = checklist_status[number]

                if status == "complete":
                    icon = "✅"
                    style = "color: green;"
                elif status == "current":
                    icon = "➡️"
                    style = "color: #1f77b4; font-weight: bold;"
                else:
                    icon = "⬜"
                    style = "color: gray;"

                html += f"""
                <div style="{style}; margin-bottom: 8px;">
                    {icon} Check {number}: {label}
                </div>
                """

            html += "</div>"
            display(widgets.HTML(html))

    def move_to_check(check_number):
        for number in checklist_status:
            if number < check_number:
                checklist_status[number] = "complete"
            elif number == check_number:
                checklist_status[number] = "current"
            else:
                checklist_status[number] = "pending"

        update_checklist()

    def code_block(code):
        display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

    def continue_button(label, next_function):
        button = widgets.Button(description=label, button_style="info")

        def on_continue_clicked(button):
            next_function()

        button.on_click(on_continue_clicked)
        display(button)

    def ask_final_shape_check():
        move_to_check(9)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you confirmed the final train/test shapes?</b><br>
            <span style="color: gray;">
            Before modelling, check that your feature and target datasets still line up.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_shape_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Use this to check the final shapes:
                    """))

                    code_block("""
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
""")

                    display(widgets.HTML("""
                    <b>What to check:</b><br>
                    <ul>
                        <li><code>X_train</code> and <code>y_train</code> should have the same number of rows.</li>
                        <li><code>X_test</code> and <code>y_test</code> should have the same number of rows.</li>
                        <li>Your training and test feature sets should have the same columns after encoding.</li>
                    </ul>
                    """))

                    checklist_status[9] = "complete"
                    update_checklist()

                    display(widgets.HTML("""
                    <b>Preprocessing checklist complete.</b><br>
                    Your classification data should now be closer to model-ready.
                    """))

            def on_yes_clicked(button):
                show_shape_guidance()

            def on_no_clicked(button):
                show_shape_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_leakage_check():
        move_to_check(8)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked for data leakage?</b><br>
            <span style="color: gray;">
            Data leakage happens when your input features contain information that would not realistically be available at prediction time.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_leakage_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Probing questions:</b><br>
                    <ul>
                        <li>Does any column reveal the answer directly?</li>
                        <li>Was any column created after the outcome happened?</li>
                        <li>Would this information really be available when making a prediction?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Examples:</b><br>
                    <ul>
                        <li>If the target is <code>churned</code>, a column like <code>cancellation_date</code> may leak the answer.</li>
                        <li>If the target is <code>loan_default</code>, a column like <code>default_fee_charged</code> may leak the answer.</li>
                        <li>If the target is <code>passed_exam</code>, a column like <code>final_grade</code> may leak the answer.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    If you identify a leaky column, remove it from both training and test features:
                    """))

                    code_block("""
X_train = X_train.drop(["leaky_column"], axis=1)
X_test = X_test.drop(["leaky_column"], axis=1)
""")

                    display(widgets.HTML("""
                    <span style="color: gray;">
                    Hint: replace <code>leaky_column</code> with the name of the column you want to remove.
                    </span>
                    """))

                    continue_button("Continue to final shape check", ask_final_shape_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to the final shape check."))
                    continue_button("Continue to final shape check", ask_final_shape_check)

            def on_no_clicked(button):
                show_leakage_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_scaling_check():
        move_to_check(7)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked whether your numerical features need scaling?</b><br>
            <span style="color: gray;">
            Some models are sensitive to the scale of numerical columns.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_scaling_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>Models like <b>Logistic Regression</b>, <b>KNN</b>, <b>SVM</b>, and <b>Neural Networks</b> usually benefit from scaling.</li>
                        <li>Tree-based models like <b>Decision Trees</b>, <b>Random Forests</b>, and <b>Gradient Boosting</b> usually do not require scaling.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Fit the scaler on the training data, then apply it to the test data.
                    """))

                    code_block("""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)
""")

                    display(widgets.HTML("""
                    <span style="color: gray;">
                    Important: use <code>fit_transform()</code> on the training data, but only <code>transform()</code> on the test data.
                    </span>
                    """))

                    continue_button("Continue to leakage check", ask_leakage_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to the leakage check."))
                    continue_button("Continue to leakage check", ask_leakage_check)

            def on_no_clicked(button):
                show_scaling_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_nominal_encoding_check():
        move_to_check(6)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked for nominal categorical features?</b><br>
            <span style="color: gray;">
            Nominal categories are labels with no meaningful order.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_nominal_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Examples of nominal features:</b><br>
                    <ul>
                        <li><code>region</code>: North, South, East, West</li>
                        <li><code>department</code>: Sales, Finance, HR</li>
                        <li><code>country</code>: UK, US, France</li>
                        <li><code>product_type</code>: Basic, Premium, Enterprise</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Use one-hot encoding for categorical features with no natural order.
                    """))

                    code_block("""
X_train_encoded = pd.get_dummies(X_train, columns=["region"], drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=["region"], drop_first=True)

X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded,
    join="left",
    axis=1,
    fill_value=0
)
""")

                    display(widgets.HTML("""
                    <span style="color: gray;">
                    Hint: replace <code>region</code> with the nominal categorical column or columns from your dataset.
                    </span><br><br>
                    <span style="color: gray;">
                    The <code>.align()</code> step helps make sure the training and test sets have the same columns after one-hot encoding.
                    </span>
                    """))

                    continue_button("Continue to scaling check", ask_scaling_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to the scaling check."))
                    continue_button("Continue to scaling check", ask_scaling_check)

            def on_no_clicked(button):
                show_nominal_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_ordinal_encoding_check():
        move_to_check(5)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked for ordinal categorical features?</b><br>
            <span style="color: gray;">
            Ordinal categories have a meaningful order.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_ordinal_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Examples of ordinal features:</b><br>
                    <ul>
                        <li><code>Low</code> → <code>Medium</code> → <code>High</code></li>
                        <li><code>Bronze</code> → <code>Silver</code> → <code>Gold</code></li>
                        <li><code>Junior</code> → <code>Mid</code> → <code>Senior</code></li>
                        <li><code>Poor</code> → <code>Fair</code> → <code>Good</code> → <code>Excellent</code></li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Use ordinal encoding when the categories have a real order.
                    """))

                    code_block("""
from sklearn.preprocessing import OrdinalEncoder

ordinal_columns = ["customer_rating"]

ordinal_order = [["Low", "Medium", "High"]]

encoder = OrdinalEncoder(categories=ordinal_order)

X_train[ordinal_columns] = encoder.fit_transform(X_train[ordinal_columns])
X_test[ordinal_columns] = encoder.transform(X_test[ordinal_columns])
""")

                    display(widgets.HTML("""
                    <span style="color: gray;">
                    Hint: replace <code>customer_rating</code> and the category order with values from your own dataset.
                    </span><br><br>
                    <span style="color: gray;">
                    Be careful: do not use ordinal encoding for categories with no real order, such as region or department.
                    </span>
                    """))

                    continue_button("Continue to nominal encoding check", ask_nominal_encoding_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to nominal categorical features."))
                    continue_button("Continue to nominal encoding check", ask_nominal_encoding_check)

            def on_no_clicked(button):
                show_ordinal_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_categorical_feature_check():
        move_to_check(4)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you identified categorical features?</b><br>
            <span style="color: gray;">
            Categorical features are columns that contain labels or groups rather than continuous numbers.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_categorical_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Use this to find text/category columns in your training features:
                    """))

                    code_block("""
X_train.select_dtypes(include=["object", "category"]).columns
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Columns like <code>region</code>, <code>department</code>, <code>product_type</code>, or <code>customer_segment</code>.</li>
                        <li>Ask whether each categorical column is <b>ordinal</b> or <b>nominal</b>.</li>
                    </ul>
                    """))

                    continue_button("Continue to ordinal encoding check", ask_ordinal_encoding_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to ordinal categorical features."))
                    continue_button("Continue to ordinal encoding check", ask_ordinal_encoding_check)

            def on_no_clicked(button):
                show_categorical_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_class_balance_check():
        move_to_check(3)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked class balance?</b><br>
            <span style="color: gray;">
            This helps you see whether one target class is much more common than another.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_balance_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    First check the count of each class:
                    """))

                    code_block("""
y_train.value_counts()
""")

                    display(widgets.HTML("""
                    <b>Percentage version:</b><br>
                    This can make imbalance easier to interpret:
                    """))

                    code_block("""
y_train.value_counts(normalize=True) * 100
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>If one class is much more common than another, the model may learn to favour the majority class.</li>
                        <li>For example, if <code>95%</code> of customers did not churn and only <code>5%</code> did churn, a model could look accurate while still failing to identify churn cases.</li>
                        <li>If the classes are very imbalanced, accuracy alone may not be a good evaluation metric.</li>
                    </ul>
                    """))

                    continue_button("Continue to categorical feature check", ask_categorical_feature_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to categorical features."))
                    continue_button("Continue to categorical feature check", ask_categorical_feature_check)

            def on_no_clicked(button):
                show_balance_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_train_test_split_check():
        move_to_check(2)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you split into training and test sets?</b><br>
            <span style="color: gray;">
            This lets you train the model on one part of the data and test it on unseen data.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_split_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    For classification, use <code>stratify=y</code> to help preserve class proportions.
                    """))

                    code_block("""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
""")

                    display(widgets.HTML("""
                    <b>Why this matters:</b><br>
                    <ul>
                        <li>The model learns from the training set.</li>
                        <li>The test set is kept separate to check whether the model generalises.</li>
                        <li><code>stratify=y</code> helps keep the target class proportions similar in both sets.</li>
                    </ul>
                    """))

                    continue_button("Continue to class balance check", ask_class_balance_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to class balance."))
                    continue_button("Continue to class balance check", ask_class_balance_check)

            def on_no_clicked(button):
                show_split_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_features_target_check():
        move_to_check(1)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you separated your features and target?</b><br>
            <span style="color: gray;">
            Your features are the input columns. Your target is the column you are trying to predict.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_features_target_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Use <code>X</code> for the input features and <code>y</code> for the target column.
                    """))

                    code_block("""
X = your_data.drop("target_column", axis=1)
y = your_data["target_column"]
""")

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    If your target column is <code>churned</code>, you could write:
                    """))

                    code_block("""
X = your_data.drop("churned", axis=1)
y = your_data["churned"]
""")

                    display(widgets.HTML("""
                    <span style="color: gray;">
                    Hint: replace <code>target_column</code> with the column you are trying to predict.
                    </span>
                    """))

                    continue_button("Continue to train/test split", ask_train_test_split_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to the train/test split."))
                    continue_button("Continue to train/test split", ask_train_test_split_check)

            def on_no_clicked(button):
                show_features_target_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    title = widgets.HTML("<b>Classification preprocessing checklist</b>")

    layout = widgets.HBox(
        [
            widgets.VBox([title, main_output], layout=widgets.Layout(width="70%")),
            widgets.VBox([checklist_output], layout=widgets.Layout(width="30%"))
        ],
        layout=widgets.Layout(width="100%", align_items="flex-start")
    )

    display(layout)

    update_checklist()
    ask_features_target_check()


def low_code_classification():
    """
    Guided low-code modelling checklist for classification using PyCaret.

    Check 1:
    - Have you set up your PyCaret classification experiment?

    Check 2:
    - Have you compared multiple candidate models?

    Check 3:
    - Have you tuned the best model?

    Check 4:
    - Have you evaluated the tuned model on holdout/test data?

    Check 5:
    - Have you compared the tuned model against a dummy baseline?

    Check 6:
    - Have you visualised and interpreted confusion matrices?

    Check 7:
    - Have you finalised the model only after evaluation?
    """
    from IPython.display import display
    import ipywidgets as widgets

    main_output = widgets.Output()
    checklist_output = widgets.Output()

    checklist_items = {
        1: "Set up PyCaret experiment",
        2: "Compare candidate models",
        3: "Tune the selected model",
        4: "Evaluate on holdout/test data",
        5: "Compare against dummy baseline",
        6: "Visualise confusion matrices",
        7: "Finalise model after evaluation"
    }

    checklist_status = {
        1: "current",
        2: "pending",
        3: "pending",
        4: "pending",
        5: "pending",
        6: "pending",
        7: "pending"
    }

    def update_checklist():
        with checklist_output:
            checklist_output.clear_output()

            html = """
            <div style="
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 12px;
                background-color: #fafafa;
                width: 300px;
            ">
            <b>Low-code modelling checklist</b><br><br>
            """

            for number, label in checklist_items.items():
                status = checklist_status[number]

                if status == "complete":
                    icon = "✅"
                    style = "color: green;"
                elif status == "current":
                    icon = "➡️"
                    style = "color: #1f77b4; font-weight: bold;"
                else:
                    icon = "⬜"
                    style = "color: gray;"

                html += f"""
                <div style="{style}; margin-bottom: 8px;">
                    {icon} Check {number}: {label}
                </div>
                """

            html += "</div>"
            display(widgets.HTML(html))

    def move_to_check(check_number):
        for number in checklist_status:
            if number < check_number:
                checklist_status[number] = "complete"
            elif number == check_number:
                checklist_status[number] = "current"
            else:
                checklist_status[number] = "pending"

        update_checklist()

    def code_block(code):
        display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

    def continue_button(label, next_function):
        button = widgets.Button(description="Next step", button_style="info")

        def on_continue_clicked(button):
            next_function()

        button.on_click(on_continue_clicked)
        display(button)



    def ask_finalize_check():
        move_to_check(7)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you finalised the model only after evaluation?</b><br>
            <span style="color: gray;">
            Finalising trains the model on all available data, so it should happen after you are happy with the holdout/test performance.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_finalize_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Finalise the tuned model only after comparing it against the dummy baseline and checking the confusion matrix.
                    """))

                    code_block("""
final = exp.finalize_model(tuned.pipeline)
""")

                    display(widgets.HTML("""
                    <b>Why this matters:</b><br>
                    <ul>
                        <li>Before finalising, you are testing how well the model performs on unseen holdout/test data.</li>
                        <li>After finalising, the model is retrained using all available data.</li>
                        <li>Do not finalise early, because you may lose a clean evaluation point.</li>
                    </ul>
                    """))

                    checklist_status[7] = "complete"
                    update_checklist()

                    display(widgets.HTML("""
                    <b>Low-code modelling checklist complete.</b><br>
                    You now have a sensible PyCaret classification modelling flow.
                    """))

            def on_yes_clicked(button):
                show_finalize_guidance()

            def on_no_clicked(button):
                show_finalize_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_confusion_matrix_check():
        move_to_check(6)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you visualised and interpreted confusion matrices?</b><br>
            <span style="color: gray;">
            A confusion matrix helps you see the types of mistakes the model is making.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_confusion_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Compare the dummy baseline and tuned model using confusion matrices on the holdout/test data.
                    """))

                    code_block("""
from pycaret.plots.classification import confusion_matrix

fig = confusion_matrix(dummy.pipeline, exp.X_test, exp.y_test)
fig.show()

fig = confusion_matrix(tuned.pipeline, exp.X_test, exp.y_test)
fig.show()
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>How many predictions are correct?</li>
                        <li>Which class is the model getting wrong most often?</li>
                        <li>Are false positives or false negatives more costly for this problem?</li>
                        <li>Does the tuned model make better mistakes than the dummy baseline?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example interpretation:</b><br>
                    If the model is predicting customer churn, a false negative means the model predicted a customer would not churn, but they actually did. 
                    That may be costly because the business misses a chance to intervene.
                    """))

                    continue_button("Continue to finalisation check", ask_finalize_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to the finalisation check."))
                    continue_button("Continue to finalisation check", ask_finalize_check)

            def on_no_clicked(button):
                show_confusion_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_dummy_baseline_check():
        move_to_check(5)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you compared your tuned model against a dummy baseline?</b><br>
            <span style="color: gray;">
            A dummy baseline gives you a simple benchmark. Your real model should meaningfully outperform it.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_dummy_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Create a dummy model and evaluate it on the same holdout/test data.
                    """))

                    code_block("""
dummy = exp.create_model("dummy")

dummy_preds = exp.predict_model(dummy.pipeline)
print(dummy_preds.metrics)
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>The dummy model is intentionally simple.</li>
                        <li>It gives you a baseline to beat.</li>
                        <li>If your tuned model is only slightly better than the dummy model, the model may not be adding much value.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Important:</b><br>
                    In classification, accuracy can be misleading if the target classes are imbalanced. Compare metrics like precision, recall, F1, or AUC depending on the project goal.
                    """))

                    continue_button("Continue to confusion matrix check", ask_confusion_matrix_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to confusion matrices."))
                    continue_button("Continue to confusion matrix check", ask_confusion_matrix_check)

            def on_no_clicked(button):
                show_dummy_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_holdout_evaluation_check():
        move_to_check(4)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you evaluated the tuned model on holdout/test data?</b><br>
            <span style="color: gray;">
            This checks how well the tuned model performs on data it did not train on.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_holdout_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Use <code>predict_model()</code> to evaluate the tuned model on PyCaret's holdout/test data.
                    """))

                    code_block("""
preds = exp.predict_model(tuned.pipeline)
print(preds.metrics)
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Are the holdout/test results close to the cross-validation results?</li>
                        <li>Is the model performing well on the metric that matters most?</li>
                        <li>For example, if false positives are costly, precision may matter more.</li>
                        <li>If false negatives are costly, recall may matter more.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Important:</b><br>
                    Do not finalise the model yet. First compare it against a baseline and inspect the confusion matrix.
                    """))

                    continue_button("Continue to dummy baseline check", ask_dummy_baseline_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s compare it against a dummy baseline."))
                    continue_button("Continue to dummy baseline check", ask_dummy_baseline_check)

            def on_no_clicked(button):
                show_holdout_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_tuning_check():
        move_to_check(3)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you tuned the selected model?</b><br>
            <span style="color: gray;">
            Tuning tries different hyperparameter settings to improve model performance.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_tuning_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Tune the best model from your comparison.
                    """))

                    code_block("""
tuned = exp.tune_model(
    top.best,
    n_iter=20,
    optimize="Precision"
)

print("best params:", tuned.best_params)
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li><code>n_iter=20</code> controls how many tuning attempts PyCaret will try.</li>
                        <li><code>optimize="Precision"</code> means PyCaret will tune for precision.</li>
                        <li>Precision is useful when false positives are especially costly.</li>
                        <li>If false negatives are more costly, you might optimise for <code>"Recall"</code> instead.</li>
                        <li>If you want a balance between precision and recall, you might optimise for <code>"F1"</code>.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example decision:</b><br>
                    If you are predicting whether a transaction is fraud, precision may matter when you want to avoid wrongly flagging legitimate transactions. 
                    Recall may matter when missing actual fraud is more costly.
                    """))

                    continue_button("Continue to holdout evaluation", ask_holdout_evaluation_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s evaluate the tuned model on holdout/test data."))
                    continue_button("Continue to holdout evaluation", ask_holdout_evaluation_check)

            def on_no_clicked(button):
                show_tuning_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_compare_models_check():
        move_to_check(2)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you compared multiple candidate models?</b><br>
            <span style="color: gray;">
            Comparing models helps you avoid assuming one algorithm is best before testing alternatives.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_compare_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Compare multiple classification models and keep the top three.
                    """))

                    code_block("""
top = exp.compare_models(n_select=3)
print(top.leaderboard)
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Which model performs best on your chosen metric?</li>
                        <li>Are the top models very close, or is one clearly stronger?</li>
                        <li>Is the best model simple enough to explain?</li>
                        <li>Does the best model make sense for your project constraints?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Important:</b><br>
                    A leaderboard is not the final answer. It is a shortlist for further evaluation.
                    """))

                    continue_button("Continue to tuning check", ask_tuning_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to tuning the selected model."))
                    continue_button("Continue to tuning check", ask_tuning_check)

            def on_no_clicked(button):
                show_compare_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_setup_check():
        move_to_check(1)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you set up your PyCaret classification experiment?</b><br>
            <span style="color: gray;">
            This defines the target column, train/test split, cross-validation folds, and reproducibility settings.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_setup_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested imports:</b><br>
                    Run these first.
                    """))

                    code_block("""
from pycaret.classification import ClassificationExperiment
from pycaret.plots.classification import confusion_matrix
""")

                    display(widgets.HTML("""
                    <b>Suggested setup code:</b><br>
                    Replace <code>"target"</code> with the name of the column you are trying to predict.
                    """))

                    code_block("""
exp = ClassificationExperiment(
    target="target",
    session_id=42,
    train_size=0.8,
    fold=10,
).fit(your_data)
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li><code>target</code> is the column you are trying to predict.</li>
                        <li><code>session_id=42</code> makes results more reproducible.</li>
                        <li><code>train_size=0.8</code> keeps 80% for training and 20% for holdout/testing.</li>
                        <li><code>fold=10</code> means models are compared using 10-fold cross-validation.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    If your target column is called <code>churned</code>, you would write:
                    """))

                    code_block("""
exp = ClassificationExperiment(
    target="churned",
    session_id=42,
    train_size=0.8,
    fold=10,
).fit(your_data)
""")

                    continue_button("Continue to model comparison", ask_compare_models_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s compare candidate models."))
                    continue_button("Continue to model comparison", ask_compare_models_check)

            def on_no_clicked(button):
                show_setup_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    title = widgets.HTML("<b>Low-code classification modelling checklist</b>")

    layout = widgets.HBox(
        [
            widgets.VBox([title, main_output], layout=widgets.Layout(width="70%")),
            widgets.VBox([checklist_output], layout=widgets.Layout(width="30%"))
        ],
        layout=widgets.Layout(width="100%", align_items="flex-start")
    )

    display(layout)

    update_checklist()
    ask_setup_check()

def data_cleaning_regression():
    """
    Guided data cleaning checklist for regression problems.

    Check 1:
    - Have you had a brief look at your data using .head()?

    Check 2:
    - Do the column headers make sense?

    Check 3:
    - Have you checked for null values?

    Check 4:
    - Have you checked data types using .info()?

    Check 5:
    - Have you checked for duplicate rows?

    Check 6:
    - Have you identified the target column?

    Check 7:
    - Is your target column numerical?

    Check 8:
    - Have you checked the target distribution?

    Check 9:
    - Have you checked for target outliers?

    Check 10:
    - Have you checked the number of unique values in each column?

    Check 11:
    - Have you checked summary statistics?

    Check 12:
    - Have you checked for data leakage?
    """
    from IPython.display import display
    import ipywidgets as widgets

    main_output = widgets.Output()
    checklist_output = widgets.Output()

    checklist_items = {
        1: "Briefly inspect the data with .head()",
        2: "Check column headers make sense",
        3: "Check for null values",
        4: "Check data types with .info()",
        5: "Check for duplicate rows",
        6: "Identify the target column",
        7: "Check target column is numerical",
        8: "Check target distribution",
        9: "Check target outliers",
        10: "Check unique values",
        11: "Check summary statistics",
        12: "Check for data leakage"
    }

    checklist_status = {
        1: "current",
        2: "pending",
        3: "pending",
        4: "pending",
        5: "pending",
        6: "pending",
        7: "pending",
        8: "pending",
        9: "pending",
        10: "pending",
        11: "pending",
        12: "pending"
    }

    def update_checklist():
        with checklist_output:
            checklist_output.clear_output()

            html = """
            <div style="
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 12px;
                background-color: #fafafa;
                width: 320px;
            ">
            <b>Regression cleaning checklist</b><br><br>
            """

            for number, label in checklist_items.items():
                status = checklist_status[number]

                if status == "complete":
                    icon = "✅"
                    style = "color: green;"
                elif status == "current":
                    icon = "➡️"
                    style = "color: #1f77b4; font-weight: bold;"
                else:
                    icon = "⬜"
                    style = "color: gray;"

                html += f"""
                <div style="{style}; margin-bottom: 8px;">
                    {icon} Check {number}: {label}
                </div>
                """

            html += "</div>"
            display(widgets.HTML(html))

    def move_to_check(check_number):
        for number in checklist_status:
            if number < check_number:
                checklist_status[number] = "complete"
            elif number == check_number:
                checklist_status[number] = "current"
            else:
                checklist_status[number] = "pending"

        update_checklist()

    def code_block(code):
        display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

    def next_step_button(next_function):
        button = widgets.Button(description="Next step", button_style="info")

        def on_continue_clicked(button):
            next_function()

        button.on_click(on_continue_clicked)
        display(button)

    def ask_leakage_check():
        move_to_check(12)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked for data leakage?</b><br>
            <span style="color: gray;">
            Data leakage happens when your input features contain information that would not realistically be available at prediction time.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Regression leakage examples:</b><br>
                    <ul>
                        <li>If predicting <code>house_price</code>, a column like <code>final_sale_price_after_negotiation</code> would leak the answer.</li>
                        <li>If predicting <code>delivery_time</code>, a column like <code>actual_arrival_time</code> may leak the answer.</li>
                        <li>If predicting <code>customer_lifetime_value</code>, a column calculated after the prediction point may leak future information.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Questions to ask:</b><br>
                    <ul>
                        <li>Would this column be available at the moment the prediction is made?</li>
                        <li>Was this column created after the target outcome happened?</li>
                        <li>Does this column directly or indirectly reveal the target value?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    If you identify a leaky column, remove it:
                    """))

                    code_block("""
your_data = your_data.drop(["leaky_column"], axis=1)
""")

                    checklist_status[12] = "complete"
                    update_checklist()

                    display(widgets.HTML("""
                    <b>Regression data cleaning checklist complete.</b><br>
                    Your data should now be closer to modelling-ready.
                    """))

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_summary_statistics_check():
        move_to_check(11)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked the summary statistics?</b><br>
            <span style="color: gray;">
            Summary statistics help you spot impossible values, unusual ranges, and extreme values.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Check numeric columns:
                    """))

                    code_block("""
your_data.describe()
""")

                    display(widgets.HTML("""
                    <b>For categorical columns:</b>
                    """))

                    code_block("""
your_data.describe(include="object")
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Impossible values, such as negative ages.</li>
                        <li>Unexpected minimum or maximum values.</li>
                        <li>Very large values that may be outliers.</li>
                        <li>Columns with very different scales.</li>
                        <li>Messy repeated categories, such as <code>UK</code>, <code>U.K.</code>, and <code>United Kingdom</code>.</li>
                    </ul>
                    """))

                    next_step_button(ask_leakage_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to the leakage check."))
                    next_step_button(ask_leakage_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_unique_values_check():
        move_to_check(10)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked the number of unique values in each column?</b><br>
            <span style="color: gray;">
            This helps you spot constant columns, ID columns, binary columns, and high-cardinality categorical columns.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Count the number of unique values in each column:
                    """))

                    code_block("""
your_data.nunique().sort_values()
""")

                    display(widgets.HTML("""
                    <b>Useful follow-up checks:</b><br>
                    Find columns with only one unique value:
                    """))

                    code_block("""
your_data.nunique()[your_data.nunique() <= 1]
""")

                    display(widgets.HTML("""
                    Find columns where almost every row has a different value:
                    """))

                    code_block("""
possible_id_columns = [
    column for column in your_data.columns
    if your_data[column].nunique() / len(your_data) > 0.95
]

possible_id_columns
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>Columns with one unique value usually do not help the model.</li>
                        <li>Columns with a different value in almost every row may be ID columns.</li>
                        <li>ID columns usually should not go directly into a regression model.</li>
                        <li>High-cardinality categorical columns may need careful encoding.</li>
                    </ul>
                    """))

                    next_step_button(ask_summary_statistics_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to summary statistics."))
                    next_step_button(ask_summary_statistics_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_target_outlier_check():
        move_to_check(9)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked for outliers in the target column?</b><br>
            <span style="color: gray;">
            In regression, extreme target values can strongly affect model training and evaluation.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested visual check:</b>
                    """))

                    code_block("""
your_data[target_column].plot(kind="box")
""")

                    display(widgets.HTML("""
                    <b>Optional IQR-based check:</b>
                    """))

                    code_block("""
q1 = your_data[target_column].quantile(0.25)
q3 = your_data[target_column].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

your_data[
    (your_data[target_column] < lower_bound) |
    (your_data[target_column] > upper_bound)
]
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>Outliers are not automatically wrong.</li>
                        <li>They may be valid but rare cases.</li>
                        <li>They may also be errors, such as an extra zero in a price column.</li>
                        <li>Investigate before removing or changing them.</li>
                    </ul>
                    """))

                    next_step_button(ask_unique_values_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to unique values."))
                    next_step_button(ask_unique_values_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_target_distribution_check():
        move_to_check(8)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked the target distribution?</b><br>
            <span style="color: gray;">
            In regression, the target should usually be a numerical value. Its distribution can affect how the model learns.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Use summary statistics and a histogram to inspect the target:
                    """))

                    code_block("""
your_data[target_column].describe()
your_data[target_column].hist()
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Is the target heavily skewed?</li>
                        <li>Are there extreme values?</li>
                        <li>Are there impossible values, such as negative prices?</li>
                        <li>Does the target contain many zeros?</li>
                        <li>Does the target look continuous, or does it look more like categories?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    If you are predicting <code>house_price</code>, a few extremely expensive houses may stretch the target distribution and affect model performance.
                    """))

                    next_step_button(ask_target_outlier_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to target outliers."))
                    next_step_button(ask_target_outlier_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_target_numeric_check():
        move_to_check(7)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Is your target column numerical?</b><br>
            <span style="color: gray;">
            For regression, the target should usually be a number you are trying to predict.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Check the target data type and summary:
                    """))

                    code_block("""
your_data[target_column].dtype
your_data[target_column].describe()
""")

                    display(widgets.HTML("""
                    <b>Regression target examples:</b><br>
                    <ul>
                        <li><code>price</code></li>
                        <li><code>revenue</code></li>
                        <li><code>sales</code></li>
                        <li><code>temperature</code></li>
                        <li><code>delivery_time</code></li>
                        <li><code>customer_lifetime_value</code></li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Important:</b><br>
                    If your target is a category, such as <code>approved</code>, <code>churned</code>, or <code>purchased</code>, this is probably classification, not regression.
                    """))

                    next_step_button(ask_target_distribution_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s inspect the target distribution."))
                    next_step_button(ask_target_distribution_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_target_column_check():
        move_to_check(6)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you identified the target column?</b><br>
            <span style="color: gray;">
            In a regression project, the target is the numerical value you want the model to predict.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Store the name of your target column in a variable:
                    """))

                    code_block("""
target_column = "your_target_column"
""")

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    If you are predicting house prices:
                    """))

                    code_block("""
target_column = "price"
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    The target column should contain the numerical outcome you want to predict, not an input feature.
                    """))

                    next_step_button(ask_target_numeric_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s check whether the target is numerical."))
                    next_step_button(ask_target_numeric_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_duplicate_check():
        move_to_check(5)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked for duplicate rows?</b><br>
            <span style="color: gray;">
            Duplicate rows can make a model look better than it really is, especially if the same record appears in both training and test data.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Check how many duplicate rows are in the dataset:
                    """))

                    code_block("""
your_data.duplicated().sum()
""")

                    display(widgets.HTML("""
                    <b>Optional inspection:</b><br>
                    View the duplicated rows before deciding whether to remove them:
                    """))

                    code_block("""
your_data[your_data.duplicated()]
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>If the duplicates are accidental, they may need to be removed.</li>
                        <li>If the duplicates are valid repeated events, they may need to stay.</li>
                        <li>Do not drop duplicates automatically without checking what one row represents.</li>
                    </ul>
                    """))

                    next_step_button(ask_target_column_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to the target column check."))
                    next_step_button(ask_target_column_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_data_type_check():
        move_to_check(4)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked the data types using <code>.info()</code>?</b><br>
            <span style="color: gray;">
            This helps you check whether each column is stored in the right format.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Example considerations:</b><br><br>

                    <b>1. Are numerical columns stored as numbers?</b><br>
                    For regression, this is especially important for your target column and numerical input features.<br><br>

                    <b>2. Are date columns stored as dates?</b><br>
                    For example, <code>purchase_date</code> or <code>created_at</code> should usually be converted to <code>datetime</code>.<br><br>

                    <b>3. Are category columns stored appropriately?</b><br>
                    For example, <code>region</code>, <code>product_type</code>, or <code>customer_segment</code> may appear as <code>object</code>, which can be fine, but it is worth checking.<br><br>

                    <b>4. Are true/false columns stored as booleans?</b><br>
                    For example, <code>is_member</code> or <code>has_discount</code> should usually be <code>bool</code>, not text.
                    """))

                    display(widgets.HTML("""
                    <b>Suggested code:</b>
                    """))

                    code_block("""
your_data.info()
""")

                    display(widgets.HTML("""
                    <b>Example code:</b>
                    """))

                    code_block("""
# Convert a column to a number
your_data["price"] = your_data["price"].astype(float)

# Convert a column to a date
your_data["purchase_date"] = pd.to_datetime(your_data["purchase_date"])

# Convert a yes/no column to True/False
your_data["is_member"] = your_data["is_member"].map({
    "Yes": True,
    "No": False
})

# Convert a column to category
your_data["region"] = your_data["region"].astype("category")
""")

                    display(widgets.HTML("""
                    <span style="color: gray;">
                    Hint: replace <code>your_data</code> with whatever you called your DataFrame, and replace the column names with columns from your own dataset.
                    </span>
                    """))

                    next_step_button(ask_duplicate_check)

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_null_check():
        move_to_check(3)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("<b>Have you checked for null values?</b>")

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    This shows the percentage of missing values in each column:
                    """))

                    code_block("""
your_data.isnull().mean() * 100
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>A column with a very small percentage of nulls may be easy to clean.</li>
                        <li>A column with a large percentage of nulls may need closer investigation.</li>
                        <li>If there are missing values in your regression target column, those rows usually cannot be used for supervised learning.</li>
                        <li>Be careful about dropping rows or columns too quickly because it can reduce your dataset or remove useful patterns.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example scenario:</b><br>
                    <ul>
                        <li>If a column has under <b>5%</b> missing data, you might be able to fill or remove those rows depending on the context.</li>
                        <li>If a column has over <b>30%</b> missing data, treat this as a warning sign and investigate before deciding.</li>
                        <li>If <code>income</code> is missing for <b>40%</b> of customers, dropping every row with missing income could remove a large part of your dataset.</li>
                        <li>Filling those values with the average could also create a misleading dataset.</li>
                    </ul>
                    """))

                    next_step_button(ask_data_type_check)

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Run this to check how many null values are in each column:
                    """))

                    code_block("""
your_data.isnull().sum()
""")

                    display(widgets.HTML("""
                    <span style="color: gray;">
                    Hint: replace <code>your_data</code> with whatever you called your DataFrame.
                    </span>
                    """))

                    next_step_button(ask_data_type_check)

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_column_header_check():
        move_to_check(2)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Do your column headers make sense?</b><br>
            <span style="color: gray;">
            For example, are they clear, readable, and easy to use in code?
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    You can rename columns using <code>.rename()</code>.
                    """))

                    code_block("""
your_data = your_data.rename(columns={
    "old_column_name": "new_column_name"
})
""")

                    display(widgets.HTML("""
                    <b>Example:</b>
                    """))

                    code_block("""
your_data = your_data.rename(columns={
    "Customer Name": "customer_name",
    "Sale Amount (£)": "sale_amount",
    "Date of Purchase": "purchase_date"
})
""")

                    next_step_button(ask_null_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to the null values check."))
                    next_step_button(ask_null_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_head_check():
        move_to_check(1)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you had a brief look at your data?</b><br>
            <span style="color: gray;">
            A quick look at the first few rows can help you understand what the dataset contains before cleaning it.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Use <code>.head()</code> to preview the first few rows:
                    """))

                    code_block("""
your_data.head()
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Do the columns look like the data you expected?</li>
                        <li>Do the values look sensible?</li>
                        <li>Are there obvious missing or strange values?</li>
                        <li>Does each row seem to represent one observation?</li>
                        <li>Can you identify the numerical value you want to predict?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example interpretation:</b><br>
                    <i>
                    Each row appears to represent one house. 
                    Columns like <code>bedrooms</code>, <code>bathrooms</code>, <code>location</code>, and <code>size_sqft</code> look like possible input features. 
                    The column <code>price</code> looks like the regression target because it is the numerical value we want to predict.
                    </i>
                    """))

                    next_step_button(ask_column_header_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to the column header check."))
                    next_step_button(ask_column_header_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    title = widgets.HTML("<b>Regression data cleaning checklist</b>")

    layout = widgets.HBox(
        [
            widgets.VBox([title, main_output], layout=widgets.Layout(width="70%")),
            widgets.VBox([checklist_output], layout=widgets.Layout(width="30%"))
        ],
        layout=widgets.Layout(width="100%", align_items="flex-start")
    )

    display(layout)

    update_checklist()
    ask_head_check()


def data_eda_regression():
    """
    Guided EDA checklist for regression problems.

    Check 1:
    - Have you explored the target distribution?

    Check 2:
    - Have you explored numeric feature distributions?

    Check 3:
    - Have you explored categorical feature distributions?

    Check 4:
    - Have you compared numeric features against the target?

    Check 5:
    - Have you compared categorical features against the target?

    Check 6:
    - Have you explored possible outliers?

    Check 7:
    - Have you checked correlations with the target?

    Check 8:
    - Have you checked multicollinearity?

    Check 9:
    - Have you explored missingness patterns?

    Check 10:
    - Have you checked skewed numerical features?

    Check 11:
    - Have you written down your EDA conclusions?
    """
    import ipywidgets as widgets
    from IPython.display import display

    main_output = widgets.Output()
    checklist_output = widgets.Output()

    checklist_items = {
        1: "Explore target distribution",
        2: "Explore numeric feature distributions",
        3: "Explore categorical feature distributions",
        4: "Compare numeric features against target",
        5: "Compare categorical features against target",
        6: "Explore possible outliers",
        7: "Check correlations with target",
        8: "Check multicollinearity",
        9: "Explore missingness patterns",
        10: "Check skewed numerical features",
        11: "Write EDA conclusions"
    }

    checklist_status = {
        1: "current",
        2: "pending",
        3: "pending",
        4: "pending",
        5: "pending",
        6: "pending",
        7: "pending",
        8: "pending",
        9: "pending",
        10: "pending",
        11: "pending"
    }

    def update_checklist():
        with checklist_output:
            checklist_output.clear_output()

            html = """
            <div style="
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 12px;
                background-color: #fafafa;
                width: 320px;
            ">
            <b>Regression EDA checklist</b><br><br>
            """

            for number, label in checklist_items.items():
                status = checklist_status[number]

                if status == "complete":
                    icon = "✅"
                    style = "color: green;"
                elif status == "current":
                    icon = "➡️"
                    style = "color: #1f77b4; font-weight: bold;"
                else:
                    icon = "⬜"
                    style = "color: gray;"

                html += f"""
                <div style="{style}; margin-bottom: 8px;">
                    {icon} Check {number}: {label}
                </div>
                """

            html += "</div>"
            display(widgets.HTML(html))

    def move_to_check(check_number):
        for number in checklist_status:
            if number < check_number:
                checklist_status[number] = "complete"
            elif number == check_number:
                checklist_status[number] = "current"
            else:
                checklist_status[number] = "pending"

        update_checklist()

    def code_block(code):
        display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

    def next_step_button(next_function):
        button = widgets.Button(description="Next step", button_style="info")

        def on_continue_clicked(button):
            next_function()

        button.on_click(on_continue_clicked)
        display(button)

    def ask_eda_conclusion_check():
        move_to_check(11)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you written down your EDA conclusions?</b><br>
            <span style="color: gray;">
            Before modelling, summarise what your exploration tells you about the regression problem.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_finish_button():
                finish_button = widgets.Button(
                    description="Finish EDA checklist",
                    button_style="success"
                )

                def on_finish_clicked(button):
                    with output:
                        output.clear_output()

                        checklist_status[11] = "complete"
                        update_checklist()

                        display(widgets.HTML("""
                        <b>Regression EDA checklist complete.</b><br>
                        You have explored the main patterns, risks, and modelling implications for a regression dataset.
                        """))

                finish_button.on_click(on_finish_clicked)
                display(finish_button)

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested EDA conclusion questions:</b><br>
                    <ul>
                        <li>What does the target distribution look like?</li>
                        <li>Is the target skewed, noisy, or affected by extreme values?</li>
                        <li>Which numeric features appear most related to the target?</li>
                        <li>Do any categorical features show clear differences in average target value?</li>
                        <li>Are there outliers that need further investigation?</li>
                        <li>Are there missingness patterns that may carry information?</li>
                        <li>Are any numerical features heavily skewed?</li>
                        <li>Are any features strongly correlated with each other?</li>
                        <li>Are there any suspicious features that could cause leakage?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example conclusion:</b><br>
                    <i>
                    The target appears right-skewed, with a small number of very high values. 
                    Several numeric features show a moderate relationship with the target, especially size and usage volume. 
                    Some categories appear to have noticeably different average target values, but a few are based on small sample sizes. 
                    There are also some highly correlated numeric features, so interpretation may need caution if using Linear Regression.
                    </i>
                    """))

                    show_finish_button()

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_skew_check():
        move_to_check(10)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked for skewed numerical features?</b><br>
            <span style="color: gray;">
            Heavily skewed features can affect some regression models, especially linear models.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Check skewness for numerical columns:
                    """))

                    code_block("""
numeric_columns = your_data.select_dtypes(include="number").columns

your_data[numeric_columns].skew().sort_values(ascending=False)
""")

                    display(widgets.HTML("""
                    <b>Optional visual:</b><br>
                    Inspect one feature in more detail:
                    """))

                    code_block("""
your_data["numeric_column"].hist()
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>Columns like income, revenue, transaction amount, house price, or spend are often skewed.</li>
                        <li>Skew does not automatically mean the data is wrong.</li>
                        <li>For some models, skewed features may benefit from transformation later in preprocessing.</li>
                        <li>For now, the EDA goal is to notice the pattern, not necessarily fix it immediately.</li>
                    </ul>
                    """))

                    next_step_button(ask_eda_conclusion_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to your EDA conclusions."))
                    next_step_button(ask_eda_conclusion_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_missingness_pattern_check():
        move_to_check(9)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you explored missingness patterns?</b><br>
            <span style="color: gray;">
            This is different from simply checking for nulls. Here, you are asking whether missing values may be meaningful.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Check the percentage of missing values by column:
                    """))

                    code_block("""
your_data.isnull().mean().sort_values(ascending=False) * 100
""")

                    display(widgets.HTML("""
                    <b>Optional relationship with target:</b><br>
                    This checks whether rows with missing values have a different average target value.
                    """))

                    code_block("""
your_data.assign(
    column_missing=your_data["column_name"].isnull()
).groupby("column_missing")[target_column].mean()
""")

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    If <code>income</code> is missing, you could check whether rows with missing income have a different average target value:
                    """))

                    code_block("""
your_data.assign(
    income_missing=your_data["income"].isnull()
).groupby("income_missing")[target_column].mean()
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>Missing values may be random, but they may also contain useful signal.</li>
                        <li>If rows with missing values have a very different average target value, missingness may matter.</li>
                        <li>Do not automatically drop missing rows without considering whether that could bias the dataset.</li>
                        <li>The decision about filling, dropping, or adding missingness indicators usually belongs in preprocessing.</li>
                    </ul>
                    """))

                    next_step_button(ask_skew_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to skewed numerical features."))
                    next_step_button(ask_skew_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_multicollinearity_check():
        move_to_check(8)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked for multicollinearity?</b><br>
            <span style="color: gray;">
            Multicollinearity happens when input features contain very similar information.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Find highly correlated numeric feature pairs:
                    """))

                    code_block("""
import numpy as np

numeric_columns = your_data.select_dtypes(include="number").columns

correlation_matrix = your_data[numeric_columns].corr().abs()

high_corr_pairs = (
    correlation_matrix
    .where(~np.eye(correlation_matrix.shape[0], dtype=bool))
    .stack()
    .sort_values(ascending=False)
)

high_corr_pairs[high_corr_pairs > 0.8]
""")

                    display(widgets.HTML("""
                    <b>Optional advanced check:</b><br>
                    Use VIF to assess multicollinearity more formally:
                    """))

                    code_block("""
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

numeric_data = your_data[numeric_columns].dropna()

vif_data = pd.DataFrame()
vif_data["feature"] = numeric_data.columns
vif_data["VIF"] = [
    variance_inflation_factor(numeric_data.values, i)
    for i in range(numeric_data.shape[1])
]

vif_data.sort_values("VIF", ascending=False)
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>This is especially important for Linear Regression and other coefficient-based models.</li>
                        <li>Tree-based models are usually less sensitive to multicollinearity.</li>
                        <li>High multicollinearity can make feature interpretation harder.</li>
                        <li>Do not remove columns automatically. First decide whether the feature is useful, interpretable, and available at prediction time.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Simple guide:</b><br>
                    <ul>
                        <li><code>VIF ≈ 1</code>: low multicollinearity</li>
                        <li><code>VIF > 5</code>: possible concern</li>
                        <li><code>VIF > 10</code>: strong concern</li>
                    </ul>
                    """))

                    next_step_button(ask_missingness_pattern_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to missingness patterns."))
                    next_step_button(ask_missingness_pattern_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_correlation_with_target_check():
        move_to_check(7)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked correlations with the target?</b><br>
            <span style="color: gray;">
            For regression, this helps you see which numerical features have stronger linear relationships with the target.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Check correlations between numeric columns and the target:
                    """))

                    code_block("""
numeric_columns = your_data.select_dtypes(include="number").columns

your_data[numeric_columns].corr()[target_column].sort_values()
""")

                    display(widgets.HTML("""
                    <b>Optional visual:</b>
                    """))

                    code_block("""
import matplotlib.pyplot as plt

your_data[numeric_columns].corr()[target_column].sort_values().plot(kind="barh")
plt.show()
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>Values close to <code>1</code> suggest a strong positive linear relationship.</li>
                        <li>Values close to <code>-1</code> suggest a strong negative linear relationship.</li>
                        <li>Values close to <code>0</code> suggest little linear relationship.</li>
                        <li>Correlation only captures linear patterns. A feature can still be useful even if correlation is low.</li>
                        <li>Very strong correlation with the target can be useful, but it is also worth checking for leakage.</li>
                    </ul>
                    """))

                    next_step_button(ask_multicollinearity_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to multicollinearity."))
                    next_step_button(ask_multicollinearity_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_outlier_check():
        move_to_check(6)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you explored possible outliers?</b><br>
            <span style="color: gray;">
            Outliers are unusual values. In regression, they can strongly affect the target, relationships, and error metrics.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Use summary statistics to spot unusual ranges:
                    """))

                    code_block("""
numeric_columns = your_data.select_dtypes(include="number").columns

your_data[numeric_columns].describe()
""")

                    display(widgets.HTML("""
                    <b>Optional boxplot:</b>
                    """))

                    code_block("""
your_data[numeric_columns].boxplot(figsize=(12, 6))
""")

                    display(widgets.HTML("""
                    <b>Check one column in more detail:</b>
                    """))

                    code_block("""
your_data["numeric_column"].describe()
your_data["numeric_column"].plot(kind="box")
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>Do not remove outliers automatically.</li>
                        <li>First ask whether the value is impossible, unlikely, or genuinely rare.</li>
                        <li>Outliers in the target can have a large effect on regression models and metrics such as RMSE.</li>
                        <li>Outliers in features can also affect linear models and distance-based models.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    A house price of <code>£10,000,000</code> may be valid for a luxury property, but it could heavily influence a regression model if most homes are around <code>£300,000</code>.
                    """))

                    next_step_button(ask_correlation_with_target_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to correlations with the target."))
                    next_step_button(ask_correlation_with_target_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_categorical_vs_target_check():
        move_to_check(5)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you compared categorical features against the target?</b><br>
            <span style="color: gray;">
            This helps you see whether different categories have different average target values.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Compare the average target value by category:
                    """))

                    code_block("""
your_data.groupby("category_column")[target_column].mean().sort_values()
""")

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    If your category column is <code>region</code>:
                    """))

                    code_block("""
your_data.groupby("region")[target_column].mean().sort_values()
""")

                    display(widgets.HTML("""
                    <b>Useful count check:</b><br>
                    Check how many rows are in each category:
                    """))

                    code_block("""
your_data["category_column"].value_counts()
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>Do some categories have much higher or lower target values?</li>
                        <li>Are category averages based on enough rows to trust?</li>
                        <li>Could a category be leaking information from the future?</li>
                        <li>Rare categories can look important by chance, so treat small groups carefully.</li>
                    </ul>
                    """))

                    next_step_button(ask_outlier_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to possible outliers."))
                    next_step_button(ask_outlier_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_numeric_vs_target_check():
        move_to_check(4)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you compared numeric features against the target?</b><br>
            <span style="color: gray;">
            This helps you see whether numeric features appear related to the value you are trying to predict.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Start by checking correlations with the target:
                    """))

                    code_block("""
numeric_columns = your_data.select_dtypes(include="number").columns

your_data[numeric_columns].corr()[target_column].sort_values()
""")

                    display(widgets.HTML("""
                    <b>Optional scatter plot:</b><br>
                    Compare one numeric feature with the target:
                    """))

                    code_block("""
your_data.plot.scatter(x="numeric_column", y=target_column)
""")

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    If predicting <code>price</code>, compare <code>size_sqft</code> against the target:
                    """))

                    code_block("""
your_data.plot.scatter(x="size_sqft", y=target_column)
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>Does the feature increase as the target increases?</li>
                        <li>Is the relationship roughly linear?</li>
                        <li>Are there clusters or unusual patterns?</li>
                        <li>Are outliers affecting the relationship?</li>
                    </ul>
                    """))

                    next_step_button(ask_categorical_vs_target_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to categorical features against the target."))
                    next_step_button(ask_categorical_vs_target_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_categorical_summary_check():
        move_to_check(3)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you explored categorical feature distributions?</b><br>
            <span style="color: gray;">
            This helps you understand labels, category frequency, and columns that may need encoding before modelling.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Select categorical columns:
                    """))

                    code_block("""
categorical_columns = your_data.select_dtypes(include=["object", "category"]).columns

your_data[categorical_columns].describe()
""")

                    display(widgets.HTML("""
                    <b>Check how many unique values each categorical column has:</b>
                    """))

                    code_block("""
your_data[categorical_columns].nunique().sort_values(ascending=False)
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Messy labels, such as <code>Yes</code>, <code>yes</code>, and <code>YES</code>.</li>
                        <li>Columns with lots of unique values.</li>
                        <li>Rare categories.</li>
                        <li>Text columns that may not be simple categories.</li>
                        <li>Categorical columns that may actually be dates or numbers.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    A column like <code>region</code> may be useful. A column like <code>customer_id</code> may have too many unique values to use directly.
                    """))

                    next_step_button(ask_numeric_vs_target_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to numeric features against the target."))
                    next_step_button(ask_numeric_vs_target_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_numeric_summary_check():
        move_to_check(2)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you explored numeric feature distributions?</b><br>
            <span style="color: gray;">
            This helps you understand the range, average, spread, and possible unusual values in numeric columns.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Select numeric columns and review summary statistics:
                    """))

                    code_block("""
numeric_columns = your_data.select_dtypes(include="number").columns

your_data[numeric_columns].describe()
""")

                    display(widgets.HTML("""
                    <b>Optional histograms:</b>
                    """))

                    code_block("""
your_data[numeric_columns].hist(figsize=(12, 8))
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Very large or very small values.</li>
                        <li>Unexpected minimums or maximums.</li>
                        <li>Columns with very different scales.</li>
                        <li>Possible outliers.</li>
                        <li>Numeric columns that may actually be IDs.</li>
                        <li>Skewed variables such as income, spend, revenue, or price.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    If <code>age</code> has a minimum of <code>-3</code> or a maximum of <code>300</code>, this may indicate a data quality issue.
                    """))

                    next_step_button(ask_categorical_summary_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to categorical feature distributions."))
                    next_step_button(ask_categorical_summary_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_target_distribution_check():
        move_to_check(1)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you explored the distribution of your target column?</b><br>
            <span style="color: gray;">
            In regression, this helps you understand the numerical value you are trying to predict.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")

            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    First, set the name of your target column:
                    """))

                    code_block("""
target_column = "your_target_column"
""")

                    display(widgets.HTML("""
                    Then inspect the target:
                    """))

                    code_block("""
your_data[target_column].describe()
""")

                    display(widgets.HTML("""
                    Optional visual:
                    """))

                    code_block("""
your_data[target_column].hist()
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Is the target heavily skewed?</li>
                        <li>Are there extreme values?</li>
                        <li>Are there impossible values, such as negative prices?</li>
                        <li>Does the target contain many zeros?</li>
                        <li>Does the target look continuous, or does it look more like categories?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    If you are predicting <code>house_price</code>, a few extremely expensive houses may stretch the target distribution and affect model performance.
                    """))

                    next_step_button(ask_numeric_summary_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to numeric feature distributions."))
                    next_step_button(ask_numeric_summary_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    title = widgets.HTML("<b>Regression EDA checklist</b>")

    layout = widgets.HBox(
        [
            widgets.VBox([title, main_output], layout=widgets.Layout(width="70%")),
            widgets.VBox([checklist_output], layout=widgets.Layout(width="30%"))
        ],
        layout=widgets.Layout(width="100%", align_items="flex-start")
    )

    display(layout)

    update_checklist()
    ask_target_distribution_check()

def low_code_regression():
    """
    Guided low-code regression modelling checklist using PyCaret.

    Check 1:
    - Have you set up your PyCaret regression experiment?

    Check 2:
    - Have you compared candidate regression models?

    Check 3:
    - Have you selected an appropriate regression metric?

    Check 4:
    - Have you tuned the selected model?

    Check 5:
    - Have you evaluated the tuned model on holdout/test data?

    Check 6:
    - Have you compared against a dummy baseline?

    Check 7:
    - Have you checked residual diagnostics?

    Check 8:
    - Have you checked prediction error?

    Check 9:
    - Have you finalised the model only after evaluation?
    """
    from IPython.display import display
    import ipywidgets as widgets

    main_output = widgets.Output()
    checklist_output = widgets.Output()

    checklist_items = {
        1: "Set up PyCaret regression experiment",
        2: "Compare candidate models",
        3: "Select appropriate regression metric",
        4: "Tune selected model",
        5: "Evaluate on holdout/test data",
        6: "Compare against dummy baseline",
        7: "Check residual diagnostics",
        8: "Check prediction error",
        9: "Finalise model after evaluation"
    }

    checklist_status = {
        1: "current",
        2: "pending",
        3: "pending",
        4: "pending",
        5: "pending",
        6: "pending",
        7: "pending",
        8: "pending",
        9: "pending"
    }

    def update_checklist():
        with checklist_output:
            checklist_output.clear_output()

            html = """
            <div style="
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 12px;
                background-color: #fafafa;
                width: 320px;
            ">
            <b>Low-code regression checklist</b><br><br>
            """

            for number, label in checklist_items.items():
                status = checklist_status[number]

                if status == "complete":
                    icon = "✅"
                    style = "color: green;"
                elif status == "current":
                    icon = "➡️"
                    style = "color: #1f77b4; font-weight: bold;"
                else:
                    icon = "⬜"
                    style = "color: gray;"

                html += f"""
                <div style="{style}; margin-bottom: 8px;">
                    {icon} Check {number}: {label}
                </div>
                """

            html += "</div>"
            display(widgets.HTML(html))

    def move_to_check(check_number):
        for number in checklist_status:
            if number < check_number:
                checklist_status[number] = "complete"
            elif number == check_number:
                checklist_status[number] = "current"
            else:
                checklist_status[number] = "pending"

        update_checklist()

    def code_block(code):
        display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

    def next_step_button(next_function):
        button = widgets.Button(description="Next step", button_style="info")

        def on_continue_clicked(button):
            next_function()

        button.on_click(on_continue_clicked)
        display(button)

    def ask_finalise_check():
        move_to_check(9)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you finalised the model only after evaluation?</b><br>
            <span style="color: gray;">
            Finalising retrains the model on all available data, so it should happen only after you are satisfied with evaluation.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Finalise the tuned model only after completing evaluation.
                    """))

                    code_block("""
final = exp.finalize_model(tuned.pipeline)
""")

                    display(widgets.HTML("""
                    <b>Before finalising, check:</b><br>
                    <ul>
                        <li>Have you compared the tuned model against the dummy baseline?</li>
                        <li>Have you checked the residuals?</li>
                        <li>Have you checked the prediction error plot?</li>
                        <li>Are you satisfied with the holdout/test performance?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Why this matters:</b><br>
                    After finalising, the model is retrained using all available data. 
                    This is useful before deployment, but it should not happen before evaluation is complete.
                    """))

                    checklist_status[9] = "complete"
                    update_checklist()

                    display(widgets.HTML("""
                    <b>Low-code regression modelling checklist complete.</b><br>
                    You now have a sensible PyCaret regression modelling workflow.
                    """))

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_prediction_error_check():
        move_to_check(8)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked the prediction error plot?</b><br>
            <span style="color: gray;">
            A prediction error plot compares actual values against predicted values.
            Points closer to the diagonal line usually mean better predictions.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b>
                    """))

                    code_block("""
from pycaret.plots.regression import prediction_error

prediction_error(tuned.pipeline, exp.X_test, exp.y_test).show()
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Are the points close to the diagonal line?</li>
                        <li>Does the model overpredict or underpredict in certain ranges?</li>
                        <li>Are predictions worse for very low or very high actual values?</li>
                        <li>Are there clear outliers where prediction and actual value are far apart?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example interpretations:</b><br>
                    <ul>
                        <li><b>Good sign:</b> Most points sit close to the diagonal line, so the model’s predictions are generally close to the actual values.</li>
                        <li><b>Possible issue:</b> The model performs well for mid-range values but spreads out for higher actual values. This suggests it struggles with high-value cases.</li>
                        <li><b>Possible issue:</b> Many points sit below the diagonal line for high actual values. This suggests the model is underpredicting larger values.</li>
                        <li><b>Possible issue:</b> A few points are far from the diagonal line. These are cases where prediction error is large and may need investigation.</li>
                    </ul>
                    """))

                    next_step_button(ask_finalise_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to finalising the model."))
                    next_step_button(ask_finalise_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_residuals_check():
        move_to_check(7)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked residual diagnostics?</b><br>
            <span style="color: gray;">
            Residuals show the difference between the actual values and the predicted values.
            A good residual plot should look fairly random, without obvious curves, funnels, or patterns.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b>
                    """))

                    code_block("""
from pycaret.plots.regression import residuals

residuals(tuned.pipeline, exp.X_test, exp.y_test).show()
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Are residuals randomly scattered around zero?</li>
                        <li>Is there a visible curve or pattern?</li>
                        <li>Do the errors get larger as predicted values increase?</li>
                        <li>Are there a few very large errors?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example interpretations:</b><br>
                    <ul>
                        <li><b>Good sign:</b> The residuals are fairly randomly scattered around zero. This suggests the model is not making an obvious systematic error.</li>
                        <li><b>Possible issue:</b> The residuals form a curved pattern. This suggests the model may not be capturing a non-linear relationship.</li>
                        <li><b>Possible issue:</b> The residuals get wider as predicted values increase. This suggests the model is less reliable for high-value predictions.</li>
                        <li><b>Possible issue:</b> A few residuals are very large. These rows may be worth investigating because the model is making unusually large errors on them.</li>
                    </ul>
                    """))

                    next_step_button(ask_prediction_error_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to the prediction error plot."))
                    next_step_button(ask_prediction_error_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_dummy_baseline_check():
        move_to_check(6)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you compared against a dummy baseline?</b><br>
            <span style="color: gray;">
            A dummy baseline gives you a simple benchmark. Your real model should clearly outperform it.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b>
                    """))

                    code_block("""
dummy = exp.create_model("dummy")

dummy_preds = exp.predict_model(dummy.pipeline)
print(dummy_preds.metrics)
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>A dummy regression model may predict a simple value, such as the average target value.</li>
                        <li>Your tuned model should clearly beat this baseline.</li>
                        <li>If your tuned model only improves RMSE slightly, it may not be adding much predictive value.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example interpretation:</b><br>
                    If the dummy model has an RMSE of <code>50,000</code> and your tuned model has an RMSE of <code>48,000</code>, the improvement may be too small to justify using the model.
                    If your tuned model reduces RMSE from <code>50,000</code> to <code>30,000</code>, that is a much more meaningful improvement.
                    """))

                    next_step_button(ask_residuals_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to residual diagnostics."))
                    next_step_button(ask_residuals_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_holdout_evaluation_check():
        move_to_check(5)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you evaluated the tuned model on holdout/test data?</b><br>
            <span style="color: gray;">
            This checks how well the tuned model performs on data it did not train on.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b>
                    """))

                    code_block("""
preds = exp.predict_model(tuned.pipeline)
print(preds.metrics)
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Are the holdout/test results close to the cross-validation results?</li>
                        <li>Is the error small enough to be useful in the real-world context?</li>
                        <li>Which metric are you using to judge success?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example interpretation:</b><br>
                    An MAE of <code>£5,000</code> may be acceptable when predicting house prices, but unacceptable when predicting monthly rent.
                    The metric only makes sense when interpreted in the context of the problem.
                    """))

                    display(widgets.HTML("""
                    <b>Important:</b><br>
                    Do not finalise the model yet. First compare it against a dummy baseline and inspect the regression diagnostic plots.
                    """))

                    next_step_button(ask_dummy_baseline_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s compare against a dummy baseline."))
                    next_step_button(ask_dummy_baseline_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_tuning_check():
        move_to_check(4)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you tuned the selected model?</b><br>
            <span style="color: gray;">
            Tuning tries different hyperparameter settings to improve model performance.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b>
                    """))

                    code_block("""
tuned = exp.tune_model(
    best,
    n_iter=20,
    optimize="RMSE"
)

print("best params:", tuned.best_params)
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li><code>n_iter=20</code> controls how many tuning attempts PyCaret will try.</li>
                        <li><code>optimize="RMSE"</code> means PyCaret will tune for RMSE.</li>
                        <li>RMSE is useful when large errors are especially costly.</li>
                        <li>If you want an easier-to-explain average error, you might optimise for <code>"MAE"</code>.</li>
                        <li>If you want to maximise explained variance, you might consider <code>"R2"</code>, but do not rely on it alone.</li>
                    </ul>
                    """))

                    next_step_button(ask_holdout_evaluation_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s evaluate the tuned model on holdout/test data."))
                    next_step_button(ask_holdout_evaluation_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_metric_check():
        move_to_check(3)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you selected an appropriate regression metric?</b><br>
            <span style="color: gray;">
            Regression metrics measure how far predictions are from actual values.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Common regression metrics:</b><br>
                    <ul>
                        <li><b>MAE:</b> average absolute error. Often easiest to explain.</li>
                        <li><b>RMSE:</b> penalises larger errors more heavily.</li>
                        <li><b>R²:</b> shows proportion of variance explained, but should not be used alone.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Suggested comparison code:</b><br>
                    Sorting by RMSE is a sensible default when large errors are costly.
                    """))

                    code_block("""
top = exp.compare_models(n_select=3, sort="RMSE")
print(top.leaderboard)
""")

                    display(widgets.HTML("""
                    <b>Example interpretation:</b><br>
                    If predicting house prices, RMSE may be useful because very large pricing errors matter.
                    If explaining performance to non-technical stakeholders, MAE may be easier because it describes the average size of the error.
                    """))

                    next_step_button(ask_tuning_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to model tuning."))
                    next_step_button(ask_tuning_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_compare_models_check():
        move_to_check(2)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you compared candidate regression models?</b><br>
            <span style="color: gray;">
            Comparing models helps you avoid assuming one algorithm is best before testing alternatives.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Compare multiple regression models and keep the top three.
                    """))

                    code_block("""
top = exp.compare_models(n_select=3, sort="RMSE")
print(top.leaderboard)
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Which model performs best on your chosen metric?</li>
                        <li>Are the top models close together, or is one clearly stronger?</li>
                        <li>Is the best model simple enough to explain?</li>
                        <li>Does the best model make sense for your project constraints?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Important:</b><br>
                    The leaderboard is a shortlist, not the final answer.
                    """))

                    next_step_button(ask_metric_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s check your regression metric choice."))
                    next_step_button(ask_metric_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_setup_check():
        move_to_check(1)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you set up your PyCaret regression experiment?</b><br>
            <span style="color: gray;">
            This defines the target column, reproducibility settings, and any automatic preprocessing options.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested import:</b>
                    """))

                    code_block("""
from pycaret.regression import RegressionExperiment
""")

                    display(widgets.HTML("""
                    <b>Suggested setup code:</b><br>
                    Replace <code>"target_column"</code> with the numerical column you are trying to predict.
                    """))

                    code_block("""
exp = RegressionExperiment(
    target="target_column",
    session_id=42,
).fit(your_data)
""")

                    display(widgets.HTML("""
                    <b>Example:</b><br>
                    If your target column is <code>house_price</code>:
                    """))

                    code_block("""
exp = RegressionExperiment(
    target="house_price",
    session_id=42,
).fit(your_data)
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li><code>target</code> is the numerical value you are trying to predict.</li>
                        <li><code>session_id=42</code> makes results more reproducible.</li>
                        <li><code>normalize=True</code> can help models that are sensitive to feature scale.</li>
                        <li><code>transformation=True</code> allows PyCaret to apply transformations that may help with skewed data.</li>
                    </ul>
                    """))

                    next_step_button(ask_compare_models_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s compare candidate regression models."))
                    next_step_button(ask_compare_models_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    title = widgets.HTML("<b>Low-code regression modelling checklist</b>")

    layout = widgets.HBox(
        [
            widgets.VBox([title, main_output], layout=widgets.Layout(width="70%")),
            widgets.VBox([checklist_output], layout=widgets.Layout(width="30%"))
        ],
        layout=widgets.Layout(width="100%", align_items="flex-start")
    )

    display(layout)

    update_checklist()
    ask_setup_check()    


def data_cleaning_timeseries():
    """
    Guided data cleaning checklist for time series problems.

    Check 1:
    - Have you had a brief look at your data using .head()?

    Check 2:
    - Do your column headers make sense?

    Check 3:
    - Have you identified the date/time column?

    Check 4:
    - Have you converted the date/time column to datetime?

    Check 5:
    - Have you sorted the data by date/time?

    Check 6:
    - Have you checked for duplicate timestamps?

    Check 7:
    - Have you set the date/time column as the index?

    Check 8:
    - Have you checked the time frequency?

    Check 9:
    - Have you checked for missing timestamps or gaps?

    Check 10:
    - Have you checked for null values?

    Check 11:
    - Have you checked data types using .info()?

    Check 12:
    - Have you checked for outliers or impossible values?

    Check 13:
    - Have you checked for future information leakage?
    """
    from IPython.display import display
    import ipywidgets as widgets

    main_output = widgets.Output()
    checklist_output = widgets.Output()

    checklist_items = {
        1: "Briefly inspect the data with .head()",
        2: "Check column headers make sense",
        3: "Identify the date/time column",
        4: "Convert date/time column to datetime",
        5: "Sort data by date/time",
        6: "Check duplicate timestamps",
        7: "Set date/time column as index",
        8: "Check time frequency",
        9: "Check missing timestamps or gaps",
        10: "Check null values",
        11: "Check data types with .info()",
        12: "Check outliers or impossible values",
        13: "Check for future information leakage"
    }

    checklist_status = {
        1: "current",
        2: "pending",
        3: "pending",
        4: "pending",
        5: "pending",
        6: "pending",
        7: "pending",
        8: "pending",
        9: "pending",
        10: "pending",
        11: "pending",
        12: "pending",
        13: "pending"
    }

    def update_checklist():
        with checklist_output:
            checklist_output.clear_output()

            html = """
            <div style="
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 12px;
                background-color: #fafafa;
                width: 330px;
            ">
            <b>Time series cleaning checklist</b><br><br>
            """

            for number, label in checklist_items.items():
                status = checklist_status[number]

                if status == "complete":
                    icon = "✅"
                    style = "color: green;"
                elif status == "current":
                    icon = "➡️"
                    style = "color: #1f77b4; font-weight: bold;"
                else:
                    icon = "⬜"
                    style = "color: gray;"

                html += f"""
                <div style="{style}; margin-bottom: 8px;">
                    {icon} Check {number}: {label}
                </div>
                """

            html += "</div>"
            display(widgets.HTML(html))

    def move_to_check(check_number):
        for number in checklist_status:
            if number < check_number:
                checklist_status[number] = "complete"
            elif number == check_number:
                checklist_status[number] = "current"
            else:
                checklist_status[number] = "pending"

        update_checklist()

    def code_block(code):
        display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

    def next_step_button(next_function):
        button = widgets.Button(description="Next step", button_style="info")

        def on_continue_clicked(button):
            next_function()

        button.on_click(on_continue_clicked)
        display(button)

    def ask_future_leakage_check():
        move_to_check(13)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked for future information leakage?</b><br>
            <span style="color: gray;">
            In time series, leakage often happens when future values accidentally influence features used for earlier predictions.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Questions to ask:</b><br>
                    <ul>
                        <li>Was this feature known at the time the prediction would be made?</li>
                        <li>Does the feature include future values?</li>
                        <li>Was the target accidentally used to create an input feature?</li>
                        <li>Were rolling averages calculated using future data?</li>
                        <li>Were missing values filled using future values?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Safer rolling feature example:</b><br>
                    Use <code>.shift(1)</code> before calculating the rolling mean so the current target value is not used to predict itself.
                    """))

                    code_block("""
your_data["rolling_mean_7"] = (
    your_data["target_column"]
    .shift(1)
    .rolling(window=7)
    .mean()
)
""")

                    display(widgets.HTML("""
                    <b>Why <code>.shift(1)</code> matters:</b><br>
                    It moves the target values back by one time step, so the rolling feature only uses values that would have been available in the past.
                    """))

                    display(widgets.HTML("""
                    <b>Example leakage problem:</b><br>
                    If you are predicting tomorrow's sales, a feature that uses next week's average sales would leak future information and make the model look better than it really is.
                    """))

                    checklist_status[13] = "complete"
                    update_checklist()

                    display(widgets.HTML("""
                    <b>Time series data cleaning checklist complete.</b><br>
                    Your dataset should now be closer to being ready for time series EDA or modelling.
                    """))

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_outlier_check():
        move_to_check(12)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked for outliers or impossible values?</b><br>
            <span style="color: gray;">
            Time series outliers may be true events, data entry errors, system issues, or important signals.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Use summary statistics to spot unusual ranges:
                    """))

                    code_block("""
your_data.describe()
""")

                    display(widgets.HTML("""
                    <b>Plot the target over time:</b>
                    """))

                    code_block("""
your_data["target_column"].plot()
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Sudden spikes or drops.</li>
                        <li>Impossible values, such as negative sales or impossible temperatures.</li>
                        <li>Flat periods that may suggest missing or frozen data collection.</li>
                        <li>One-off events that may be real and important.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example interpretation:</b><br>
                    A temperature value of <code>114°C</code> in a daily weather dataset is probably an error. 
                    A sales spike on Black Friday may be real and important, so you should not remove it automatically.
                    """))

                    next_step_button(ask_future_leakage_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to the future leakage check."))
                    next_step_button(ask_future_leakage_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_data_type_check():
        move_to_check(11)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked data types using <code>.info()</code>?</b><br>
            <span style="color: gray;">
            This helps you check whether date, target, numeric, and categorical columns are stored correctly.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b>
                    """))

                    code_block("""
your_data.info()
""")

                    display(widgets.HTML("""
                    <b>What to check:</b><br>
                    <ul>
                        <li>Is your date/time column stored as <code>datetime64</code>?</li>
                        <li>Is your target column numeric?</li>
                        <li>Are categorical columns stored as text/category?</li>
                        <li>Are any numeric columns incorrectly stored as <code>object</code>?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example fixes:</b>
                    """))

                    code_block("""
# Convert date column
your_data[date_column] = pd.to_datetime(your_data[date_column])

# Convert a target column to numeric
your_data["target_column"] = pd.to_numeric(
    your_data["target_column"],
    errors="coerce"
)

# Convert a category column
your_data["region"] = your_data["region"].astype("category")
""")

                    next_step_button(ask_outlier_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to outliers and impossible values."))
                    next_step_button(ask_outlier_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_null_check():
        move_to_check(10)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked for null values?</b><br>
            <span style="color: gray;">
            In time series, nulls need careful handling because some filling methods can accidentally use future information.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b>
                    """))

                    code_block("""
your_data.isnull().sum()
""")

                    display(widgets.HTML("""
                    <b>Percentage version:</b>
                    """))

                    code_block("""
your_data.isnull().mean() * 100
""")

                    display(widgets.HTML("""
                    <b>Important time series note:</b><br>
                    <ul>
                        <li>Forward fill can be reasonable if the most recent past value is a sensible estimate.</li>
                        <li>Backfill can use future information, so be careful.</li>
                        <li>Interpolation may be useful for smooth sensor-like data, but it is not always appropriate.</li>
                        <li>Do not fill missing values automatically without understanding why they are missing.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example options:</b>
                    """))

                    code_block("""
# Forward fill using past values
your_data = your_data.ffill()

# Interpolate missing values
your_data = your_data.interpolate()
""")

                    next_step_button(ask_data_type_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to data types."))
                    next_step_button(ask_data_type_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_missing_timestamps_check():
        move_to_check(9)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked for missing timestamps or gaps?</b><br>
            <span style="color: gray;">
            A missing timestamp means an expected time period is absent from the sequence.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Daily data example:</b>
                    """))

                    code_block("""
full_date_range = pd.date_range(
    start=your_data.index.min(),
    end=your_data.index.max(),
    freq="D"
)

missing_dates = full_date_range.difference(your_data.index)

missing_dates
""")

                    display(widgets.HTML("""
                    <b>Monthly data example:</b>
                    """))

                    code_block("""
full_date_range = pd.date_range(
    start=your_data.index.min(),
    end=your_data.index.max(),
    freq="MS"
)

missing_dates = full_date_range.difference(your_data.index)

missing_dates
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>A missing timestamp is not the same as a null value.</li>
                        <li>A row can be missing entirely from the time sequence.</li>
                        <li>Missing time periods may need to be filled, removed, or explicitly handled depending on the problem.</li>
                    </ul>
                    """))

                    next_step_button(ask_null_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to null values."))
                    next_step_button(ask_null_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_frequency_check():
        move_to_check(8)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked the time frequency?</b><br>
            <span style="color: gray;">
            This helps you understand whether the series is daily, weekly, monthly, quarterly, or irregular.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b>
                    """))

                    code_block("""
your_data.index.inferred_freq
""")

                    display(widgets.HTML("""
                    <b>Alternative gap check:</b>
                    """))

                    code_block("""
your_data.index.to_series().diff().value_counts()
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Is the data daily, weekly, monthly, quarterly, or irregular?</li>
                        <li>Are the time gaps consistent?</li>
                        <li>Does the frequency match the forecasting problem?</li>
                        <li>Does the model you plan to use require a regular frequency?</li>
                    </ul>
                    """))

                    next_step_button(ask_missing_timestamps_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to missing timestamps or gaps."))
                    next_step_button(ask_missing_timestamps_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_set_index_check():
        move_to_check(7)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you set the date/time column as the index?</b><br>
            <span style="color: gray;">
            A datetime index makes time-based operations like plotting, resampling, and frequency checks easier.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b>
                    """))

                    code_block("""
your_data = your_data.set_index(date_column)
""")

                    display(widgets.HTML("""
                    <b>Optional check:</b>
                    """))

                    code_block("""
your_data.index
""")

                    display(widgets.HTML("""
                    <b>Why this matters:</b><br>
                    Once the date/time column is the index, it becomes easier to use tools like <code>.resample()</code>, <code>.asfreq()</code>, and time-based plotting.
                    """))

                    next_step_button(ask_frequency_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to checking the time frequency."))
                    next_step_button(ask_frequency_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_duplicate_timestamp_check():
        move_to_check(6)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked for duplicate timestamps?</b><br>
            <span style="color: gray;">
            Duplicate timestamps may be valid, but only if you expect multiple observations per time period.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b>
                    """))

                    code_block("""
your_data[date_column].duplicated().sum()
""")

                    display(widgets.HTML("""
                    <b>Inspect duplicate timestamps:</b>
                    """))

                    code_block("""
your_data[your_data[date_column].duplicated(keep=False)]
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>If you expect one row per time period, duplicate timestamps may be a problem.</li>
                        <li>If you have multiple stores, products, customers, or sensors per timestamp, duplicates may be valid.</li>
                        <li>If duplicates are valid, you may need a grouped or panel time series approach.</li>
                        <li>If duplicates are accidental, you may need to remove or aggregate them.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example aggregation:</b>
                    """))

                    code_block("""
your_data = (
    your_data
    .groupby(date_column)
    .mean(numeric_only=True)
    .reset_index()
)
""")

                    next_step_button(ask_set_index_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to setting the date/time index."))
                    next_step_button(ask_set_index_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_sort_datetime_check():
        move_to_check(5)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you sorted the data by date/time?</b><br>
            <span style="color: gray;">
            Time series models and plots depend on the observations being in the correct order.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b>
                    """))

                    code_block("""
your_data = your_data.sort_values(date_column)
""")

                    display(widgets.HTML("""
                    <b>Optional check:</b>
                    """))

                    code_block("""
your_data[date_column].head()
your_data[date_column].tail()
""")

                    display(widgets.HTML("""
                    <b>Why this matters:</b><br>
                    If your dates are out of order, trends, lags, rolling averages, train/test splits, and plots can all be misleading.
                    """))

                    next_step_button(ask_duplicate_timestamp_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to duplicate timestamps."))
                    next_step_button(ask_duplicate_timestamp_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_convert_datetime_check():
        move_to_check(4)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you converted the date/time column to datetime?</b><br>
            <span style="color: gray;">
            If dates are stored as text, Python may not sort, resample, or filter them correctly.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b>
                    """))

                    code_block("""
your_data[date_column] = pd.to_datetime(your_data[date_column])
""")

                    display(widgets.HTML("""
                    <b>Useful check:</b>
                    """))

                    code_block("""
your_data[date_column].dtype
""")

                    display(widgets.HTML("""
                    <b>Common issue:</b><br>
                    If your dates are in a non-standard format, you may need to specify the format explicitly.
                    """))

                    code_block("""
your_data[date_column] = pd.to_datetime(
    your_data[date_column],
    format="%Y-%m-%d"
)
""")

                    display(widgets.HTML("""
                    <span style="color: gray;">
                    Hint: make sure you have imported pandas as <code>pd</code>.
                    </span>
                    """))

                    next_step_button(ask_sort_datetime_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to sorting by date/time."))
                    next_step_button(ask_sort_datetime_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_identify_datetime_check():
        move_to_check(3)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you identified the date/time column?</b><br>
            <span style="color: gray;">
            This is the column that tells you when each observation happened.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Store the date/time column name in a variable:
                    """))

                    code_block("""
date_column = "your_date_column"
""")

                    display(widgets.HTML("""
                    <b>Example:</b>
                    """))

                    code_block("""
date_column = "date"
""")

                    display(widgets.HTML("""
                    <b>Useful check:</b><br>
                    Look at the column names again:
                    """))

                    code_block("""
your_data.columns
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    The date/time column might be called <code>date</code>, <code>timestamp</code>, <code>created_at</code>, <code>month</code>, <code>week</code>, or something similar.
                    """))

                    next_step_button(ask_convert_datetime_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to converting it to datetime."))
                    next_step_button(ask_convert_datetime_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_column_header_check():
        move_to_check(2)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Do your column headers make sense?</b><br>
            <span style="color: gray;">
            Clear column names make it easier to identify the date column, target column, and any useful features.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    You can rename columns using <code>.rename()</code>.
                    """))

                    code_block("""
your_data = your_data.rename(columns={
    "old_column_name": "new_column_name"
})
""")

                    display(widgets.HTML("""
                    <b>Example:</b>
                    """))

                    code_block("""
your_data = your_data.rename(columns={
    "Date of Sale": "date",
    "Daily Sales (£)": "sales",
    "Store ID": "store_id"
})
""")

                    display(widgets.HTML("""
                    <b>Why this helps:</b><br>
                    In time series work, clear names like <code>date</code>, <code>sales</code>, and <code>store_id</code> make it easier to avoid mistakes later.
                    """))

                    next_step_button(ask_identify_datetime_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to identifying the date/time column."))
                    next_step_button(ask_identify_datetime_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_head_check():
        move_to_check(1)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you had a brief look at your data?</b><br>
            <span style="color: gray;">
            A quick look at the first few rows can help you understand the time structure before cleaning.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b>
                    """))

                    code_block("""
your_data.head()
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Can you identify the date/time column?</li>
                        <li>Can you identify the value you want to forecast?</li>
                        <li>Does each row represent one time period?</li>
                        <li>Are there multiple records per time period?</li>
                        <li>Do the values look sensible?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example interpretation:</b><br>
                    <i>
                    Each row appears to represent one day of sales.
                    The <code>date</code> column tells us when the observation happened.
                    The <code>sales</code> column looks like the value we may want to forecast.
                    The next checks should confirm that the dates are stored correctly, sorted correctly, and have no missing time periods.
                    </i>
                    """))

                    next_step_button(ask_column_header_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to the column header check."))
                    next_step_button(ask_column_header_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    title = widgets.HTML("<b>Time series data cleaning checklist</b>")

    layout = widgets.HBox(
        [
            widgets.VBox([title, main_output], layout=widgets.Layout(width="70%")),
            widgets.VBox([checklist_output], layout=widgets.Layout(width="30%"))
        ],
        layout=widgets.Layout(width="100%", align_items="flex-start")
    )

    display(layout)

    update_checklist()
    ask_head_check()



def data_eda_timeseries():
    """
    Guided EDA checklist for time series problems.

    Check 1:
    - Have you plotted the time series?

    Check 2:
    - Have you checked the overall trend?

    Check 3:
    - Have you checked for seasonality?

    Check 4:
    - Have you checked for cycles or repeating patterns?

    Check 5:
    - Have you checked noise/volatility?

    Check 6:
    - Have you explored outliers and unusual events?

    Check 7:
    - Have you checked rolling statistics?

    Check 8:
    - Have you checked stationarity? Optional theory check.

    Check 9:
    - Have you checked autocorrelation? Optional theory check.

    Check 10:
    - Have you explored lag relationships? Optional feature insight.

    Check 11:
    - Have you checked train/test split timing?

    Check 12:
    - Have you written down your EDA conclusions?
    """
    from IPython.display import display
    import ipywidgets as widgets

    main_output = widgets.Output()
    checklist_output = widgets.Output()

    checklist_items = {
        1: "Plot the time series",
        2: "Check trend",
        3: "Check seasonality",
        4: "Check cycles",
        5: "Check noise/volatility",
        6: "Explore outliers and unusual events",
        7: "Check rolling statistics",
        8: "Check stationarity (optional theory check)",
        9: "Check autocorrelation (optional theory check)",
        10: "Explore lag relationships (optional feature insight)",
        11: "Check train/test split timing",
        12: "Write EDA conclusions"
    }

    checklist_status = {
        1: "current",
        2: "pending",
        3: "pending",
        4: "pending",
        5: "pending",
        6: "pending",
        7: "pending",
        8: "pending",
        9: "pending",
        10: "pending",
        11: "pending",
        12: "pending"
    }

    def update_checklist():
        with checklist_output:
            checklist_output.clear_output()

            html = """
            <div style="
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 12px;
                background-color: #fafafa;
                width: 330px;
            ">
            <b>Time series EDA checklist</b><br><br>
            """

            for number, label in checklist_items.items():
                status = checklist_status[number]

                if status == "complete":
                    icon = "✅"
                    style = "color: green;"
                elif status == "current":
                    icon = "➡️"
                    style = "color: #1f77b4; font-weight: bold;"
                else:
                    icon = "⬜"
                    style = "color: gray;"

                html += f"""
                <div style="{style}; margin-bottom: 8px;">
                    {icon} Check {number}: {label}
                </div>
                """

            html += "</div>"
            display(widgets.HTML(html))

    def move_to_check(check_number):
        for number in checklist_status:
            if number < check_number:
                checklist_status[number] = "complete"
            elif number == check_number:
                checklist_status[number] = "current"
            else:
                checklist_status[number] = "pending"

        update_checklist()

    def code_block(code):
        display(widgets.HTML(f"<pre><code>{code}</code></pre>"))

    def next_step_button(next_function):
        button = widgets.Button(description="Next step", button_style="info")

        def on_continue_clicked(button):
            next_function()

        button.on_click(on_continue_clicked)
        display(button)

    def ask_eda_conclusion_check():
        move_to_check(12)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you written down your time series EDA conclusions?</b><br>
            <span style="color: gray;">
            Before modelling, summarise what you have learned about the time series structure.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_finish_button():
                finish_button = widgets.Button(
                    description="Finish EDA checklist",
                    button_style="success"
                )

                def on_finish_clicked(button):
                    with output:
                        output.clear_output()

                        checklist_status[12] = "complete"
                        update_checklist()

                        display(widgets.HTML("""
                        <b>Time series EDA checklist complete.</b><br>
                        You have explored the main time-based patterns, risks, and modelling implications in your dataset.
                        """))

                finish_button.on_click(on_finish_clicked)
                display(finish_button)

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested EDA conclusion questions:</b><br>
                    <ul>
                        <li>Is there an overall trend?</li>
                        <li>Is there seasonality?</li>
                        <li>Are there cycles or longer repeating patterns?</li>
                        <li>How noisy or volatile is the series?</li>
                        <li>Are there outliers or unusual events?</li>
                        <li>Do rolling statistics change over time?</li>
                        <li>Does the series look stationary or non-stationary?</li>
                        <li>Are past values likely to be useful for forecasting?</li>
                        <li>Does your train/test split respect time order?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example conclusion:</b><br>
                    <i>
                    The series shows an upward trend and clear weekly seasonality. 
                    There are several unusual spikes that may correspond to promotions or holidays. 
                    The rolling mean changes over time, suggesting the series may not be stationary. 
                    Recent lag values appear likely to be useful for forecasting, so lag-based patterns should be considered during modelling. 
                    The train/test split should preserve time order so that future values are not used to predict the past.
                    </i>
                    """))

                    show_finish_button()

            def on_yes_clicked(button):
                show_guidance()

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_train_test_timing_check():
        move_to_check(11)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked train/test split timing?</b><br>
            <span style="color: gray;">
            Even when using AutoGluon, your test data should usually come after your training data in time.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Use the earlier part of the series for training and the later part for testing.
                    """))

                    code_block("""
split_point = int(len(your_data) * 0.8)

train_data = your_data.iloc[:split_point]
test_data = your_data.iloc[split_point:]
""")

                    display(widgets.HTML("""
                    <b>Check the split dates:</b>
                    """))

                    code_block("""
print(train_data.index.min(), train_data.index.max())
print(test_data.index.min(), test_data.index.max())
""")

                    display(widgets.HTML("""
                    <b>What to avoid:</b><br>
                    Do not randomly shuffle a basic time series forecast split.
                    """))

                    code_block("""
# Avoid this for basic time series forecasting
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(
    your_data,
    test_size=0.2,
    shuffle=True
)
""")

                    display(widgets.HTML("""
                    <b>Why this matters:</b><br>
                    <ul>
                        <li>Time series forecasting is about predicting future values.</li>
                        <li>The test set should represent future data the model has not seen.</li>
                        <li>Random splitting can leak future patterns into training and make performance look better than it really is.</li>
                    </ul>
                    """))

                    next_step_button(ask_eda_conclusion_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to your EDA conclusions."))
                    next_step_button(ask_eda_conclusion_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_lag_relationship_check():
        move_to_check(10)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you explored lag relationships? <span style="color: gray;">(optional feature insight)</span></b><br>
            <span style="color: gray;">
            AutoGluon can handle time series patterns without you manually engineering every lag, but this check can help you understand whether past values are useful.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Create a one-period lag and compare it with the target.
                    """))

                    code_block("""
your_data["lag_1"] = your_data[target_column].shift(1)

your_data[[target_column, "lag_1"]].corr()
""")

                    display(widgets.HTML("""
                    <b>Several lag examples:</b>
                    """))

                    code_block("""
for lag in [1, 7, 14, 30]:
    your_data[f"lag_{lag}"] = your_data[target_column].shift(lag)

your_data[[target_column, "lag_1", "lag_7", "lag_14", "lag_30"]].corr()
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>If recent values are strongly related to current values, lag information may be useful for forecasting.</li>
                        <li>A strong <code>lag_7</code> relationship in daily data may suggest weekly seasonality.</li>
                        <li>A strong <code>lag_12</code> relationship in monthly data may suggest yearly seasonality.</li>
                        <li>Lag features should only use past values, not future values.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Important:</b><br>
                    This is an optional insight check. If students are using AutoGluon, they do not necessarily need to manually build these lag features before modelling.
                    """))

                    next_step_button(ask_train_test_timing_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to train/test split timing."))
                    next_step_button(ask_train_test_timing_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_autocorrelation_check():
        move_to_check(9)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked autocorrelation? <span style="color: gray;">(optional theory check)</span></b><br>
            <span style="color: gray;">
            AutoGluon does not require you to manually inspect autocorrelation, but this can help you understand how strongly the series relates to its past values.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b>
                    """))

                    code_block("""
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(your_data[target_column].dropna(), lags=40)
""")

                    display(widgets.HTML("""
                    <b>Alternative:</b>
                    """))

                    code_block("""
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(your_data[target_column].dropna())
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Are recent values related to past values?</li>
                        <li>Do certain lags show strong relationships?</li>
                        <li>Are there repeating peaks that suggest seasonality?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example interpretation:</b><br>
                    Strong autocorrelation at short lags suggests recent values may be useful for forecasting future values.
                    Peaks at regular intervals may suggest seasonal behaviour.
                    """))

                    display(widgets.HTML("""
                    <b>Important:</b><br>
                    This is an optional theory check. It is more essential for traditional statistical forecasting than for AutoGluon workflows.
                    """))

                    next_step_button(ask_lag_relationship_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to lag relationships."))
                    next_step_button(ask_lag_relationship_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_stationarity_check():
        move_to_check(8)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked stationarity? <span style="color: gray;">(optional theory check)</span></b><br>
            <span style="color: gray;">
            AutoGluon does not require you to manually prove stationarity, but this check can help you understand whether the series changes over time.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b>
                    """))

                    code_block("""
from statsmodels.tsa.stattools import adfuller

result = adfuller(your_data[target_column].dropna())

print("ADF statistic:", result[0])
print("p-value:", result[1])
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>A stationary series has statistical properties that are relatively stable over time.</li>
                        <li>If the p-value is below <code>0.05</code>, this is often treated as evidence that the series is stationary.</li>
                        <li>If the p-value is above <code>0.05</code>, the series may be non-stationary.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Important:</b><br>
                    This is an optional theory check. ARIMA-style models often care more about stationarity than AutoGluon does.
                    """))

                    next_step_button(ask_autocorrelation_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to autocorrelation."))
                    next_step_button(ask_autocorrelation_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_rolling_statistics_check():
        move_to_check(7)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked rolling statistics?</b><br>
            <span style="color: gray;">
            Rolling statistics help you see whether the average or variation changes over time.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Try a rolling mean and rolling standard deviation.
                    """))

                    code_block("""
your_data[target_column].rolling(window=12).mean().plot()
your_data[target_column].rolling(window=12).std().plot()
""")

                    display(widgets.HTML("""
                    <b>For daily data, you might try:</b>
                    """))

                    code_block("""
your_data[target_column].rolling(window=30).mean().plot()
your_data[target_column].rolling(window=30).std().plot()
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Is the average changing over time?</li>
                        <li>Is the variation changing over time?</li>
                        <li>Does the series become more or less stable?</li>
                        <li>Are some periods more volatile than others?</li>
                    </ul>
                    """))

                    next_step_button(ask_stationarity_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to the optional stationarity check."))
                    next_step_button(ask_stationarity_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_outlier_event_check():
        move_to_check(6)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you explored outliers and unusual events?</b><br>
            <span style="color: gray;">
            Outliers in time series may be errors, rare events, promotions, holidays, weather events, or system issues.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b>
                    """))

                    code_block("""
your_data[target_column].describe()
your_data[target_column].plot()
""")

                    display(widgets.HTML("""
                    <b>Optional boxplot:</b>
                    """))

                    code_block("""
your_data[target_column].plot(kind="box")
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Sudden spikes or drops.</li>
                        <li>Values that look impossible.</li>
                        <li>Values that may correspond to real-world events.</li>
                        <li>Periods where the data appears frozen or flat.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example interpretation:</b><br>
                    There is a large spike around one date. This may be a real event, such as a promotion, holiday, weather event, or system issue, so it should be investigated before being removed.
                    """))

                    next_step_button(ask_rolling_statistics_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to rolling statistics."))
                    next_step_button(ask_rolling_statistics_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_noise_volatility_check():
        move_to_check(5)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked noise or volatility?</b><br>
            <span style="color: gray;">
            This helps you understand whether the series is smooth, jumpy, stable, or changing in variability.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Plot the series and rolling standard deviation.
                    """))

                    code_block("""
your_data[target_column].plot()
your_data[target_column].rolling(window=12).std().plot()
""")

                    display(widgets.HTML("""
                    <b>For daily data, you might try:</b>
                    """))

                    code_block("""
your_data[target_column].rolling(window=30).std().plot()
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Is the series smooth or highly jumpy?</li>
                        <li>Does the amount of variation change over time?</li>
                        <li>Are some periods more volatile than others?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example interpretation:</b><br>
                    The series becomes more volatile in later periods, suggesting that prediction uncertainty may be higher there.
                    """))

                    next_step_button(ask_outlier_event_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to outliers and unusual events."))
                    next_step_button(ask_outlier_event_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_cycles_check():
        move_to_check(4)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked for cycles or repeating patterns?</b><br>
            <span style="color: gray;">
            Cycles are repeated rises and falls that do not necessarily happen at a fixed calendar interval.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b>
                    """))

                    code_block("""
your_data[target_column].plot()
your_data[target_column].rolling(window=30).mean().plot()
""")

                    display(widgets.HTML("""
                    <b>How to think about it:</b><br>
                    <ul>
                        <li>Seasonality is regular and calendar-based.</li>
                        <li>Cycles are repeated movements that may be less regular.</li>
                        <li>Look for longer waves in the data.</li>
                        <li>Ask whether the series rises and falls in repeated phases.</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example interpretation:</b><br>
                    The series appears to rise and fall in broad waves, but not at a fixed monthly or yearly interval. This may suggest a cycle rather than strict seasonality.
                    """))

                    next_step_button(ask_noise_volatility_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to noise and volatility."))
                    next_step_button(ask_noise_volatility_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_seasonality_check():
        move_to_check(3)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked for seasonality?</b><br>
            <span style="color: gray;">
            Seasonality means a pattern repeats at a regular calendar interval, such as daily, weekly, monthly, quarterly, or yearly.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    First, plot the series:
                    """))

                    code_block("""
your_data[target_column].plot()
""")

                    display(widgets.HTML("""
                    <b>Monthly seasonality:</b>
                    """))

                    code_block("""
your_data.groupby(your_data.index.month)[target_column].mean().plot(kind="bar")
""")

                    display(widgets.HTML("""
                    <b>Day-of-week seasonality:</b>
                    """))

                    code_block("""
your_data.groupby(your_data.index.dayofweek)[target_column].mean().plot(kind="bar")
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Are certain months consistently higher or lower?</li>
                        <li>Are certain weekdays consistently higher or lower?</li>
                        <li>Does the pattern repeat at a regular calendar interval?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example interpretation:</b><br>
                    The average target value is higher in December, suggesting possible yearly seasonality.
                    """))

                    next_step_button(ask_cycles_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to cycles or repeating patterns."))
                    next_step_button(ask_cycles_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_trend_check():
        move_to_check(2)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you checked the overall trend?</b><br>
            <span style="color: gray;">
            Trend tells you whether the series is generally increasing, decreasing, or staying broadly flat over time.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    Use a rolling mean to smooth short-term noise.
                    """))

                    code_block("""
your_data[target_column].plot()
your_data[target_column].rolling(window=12).mean().plot()
""")

                    display(widgets.HTML("""
                    <b>For daily data, you might try:</b>
                    """))

                    code_block("""
your_data[target_column].rolling(window=30).mean().plot()
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Is the series generally increasing?</li>
                        <li>Is the series generally decreasing?</li>
                        <li>Does it stay broadly flat?</li>
                        <li>Does it change direction over time?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example interpretation:</b><br>
                    The rolling mean appears to increase over time, suggesting an upward trend in the target variable.
                    """))

                    next_step_button(ask_seasonality_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to seasonality."))
                    next_step_button(ask_seasonality_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_plot_series_check():
        move_to_check(1)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML("""
            <b>Have you plotted the time series?</b><br>
            <span style="color: gray;">
            A plot gives you the quickest first view of how the target changes over time.
            </span>
            """)

            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def show_guidance():
                with output:
                    output.clear_output()

                    display(widgets.HTML("""
                    <b>Suggested code:</b><br>
                    First, set your target column:
                    """))

                    code_block("""
target_column = "your_target_column"
""")

                    display(widgets.HTML("""
                    Then plot the series:
                    """))

                    code_block("""
your_data[target_column].plot()
""")

                    display(widgets.HTML("""
                    <b>What to look for:</b><br>
                    <ul>
                        <li>Can you see the series over time?</li>
                        <li>Does it generally go up, down, or stay flat?</li>
                        <li>Are there sudden jumps, drops, or gaps?</li>
                        <li>Are there repeating patterns?</li>
                    </ul>
                    """))

                    display(widgets.HTML("""
                    <b>Example interpretation:</b><br>
                    The series appears to increase over time, with some repeated peaks and dips. This suggests there may be both trend and seasonality to investigate.
                    """))

                    next_step_button(ask_trend_check)

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML("<b>Great.</b> Let’s move on to checking the overall trend."))
                    next_step_button(ask_trend_check)

            def on_no_clicked(button):
                show_guidance()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    title = widgets.HTML("<b>Time series EDA checklist</b>")

    layout = widgets.HBox(
        [
            widgets.VBox([title, main_output], layout=widgets.Layout(width="70%")),
            widgets.VBox([checklist_output], layout=widgets.Layout(width="30%"))
        ],
        layout=widgets.Layout(width="100%", align_items="flex-start")
    )

    display(layout)

    update_checklist()
    ask_plot_series_check()


def modelling_low_code_timeseries():
    """
    Guided low-code time series modelling checklist using AutoGluon.

    This follows the class workflow:
    - Prepare the data format
    - Convert to AutoGluon TimeSeriesDataFrame
    - Set prediction length
    - Create a time-aware train/test split
    - Set up and fit TimeSeriesPredictor
    - Compare models
    - Generate forecasts
    - Plot predictions
    - Compare against a naive baseline
    - Write modelling conclusions
    """
    from IPython.display import display
    import ipywidgets as widgets
    import html

    main_output = widgets.Output()
    checklist_output = widgets.Output()

    checklist_items = {
        1: "Check AutoGluon row format",
        2: "Convert date column to datetime",
        3: "Sort data by date",
        4: "Add or confirm item_id column",
        5: "Confirm AutoGluon schema",
        6: "Convert to TimeSeriesDataFrame",
        7: "Set prediction length",
        8: "Create time-based train/test split",
        9: "Set up TimeSeriesPredictor",
        10: "Fit predictor on training data",
        11: "Compare models with leaderboard",
        12: "Generate forecasts",
        13: "Plot forecast against test data",
        14: "Compare best model against naive baseline",
        15: "Write modelling conclusions"
    }

    checklist_status = {
        number: "pending"
        for number in checklist_items
    }
    checklist_status[1] = "current"

    def update_checklist():
        with checklist_output:
            checklist_output.clear_output()

            checklist_html = """
            <div style="
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 12px;
                background-color: #fafafa;
                width: 340px;
            ">
            <b>Low-code time series modelling checklist</b><br><br>
            """

            for number, label in checklist_items.items():
                status = checklist_status[number]

                if status == "complete":
                    icon = "✅"
                    style = "color: green;"
                elif status == "current":
                    icon = "➡️"
                    style = "color: #1f77b4; font-weight: bold;"
                else:
                    icon = "⬜"
                    style = "color: gray;"

                checklist_html += f"""
                <div style="{style}; margin-bottom: 8px;">
                    {icon} Check {number}: {label}
                </div>
                """

            checklist_html += "</div>"
            display(widgets.HTML(checklist_html))

    def move_to_check(check_number):
        for number in checklist_status:
            if number < check_number:
                checklist_status[number] = "complete"
            elif number == check_number:
                checklist_status[number] = "current"
            else:
                checklist_status[number] = "pending"

        update_checklist()

    def code_block(code):
        escaped_code = html.escape(code.strip("\n"))
        display(widgets.HTML(f"<pre><code>{escaped_code}</code></pre>"))

    def next_step_button(next_function):
        button = widgets.Button(description="Next step", button_style="info")

        def on_continue_clicked(button):
            next_function()

        button.on_click(on_continue_clicked)
        display(button)

    def finish_button():
        button = widgets.Button(description="Finish checklist", button_style="success")

        def on_finish_clicked(button):
            with main_output:
                main_output.clear_output()

                checklist_status[15] = "complete"
                update_checklist()

                display(widgets.HTML("""
                <b>Low-code time series modelling checklist complete.</b><br>
                You have followed the main AutoGluon modelling workflow from data formatting through to model comparison and interpretation.
                """))

        button.on_click(on_finish_clicked)
        display(button)

    def render_check(
        check_number,
        question_html,
        guidance_function,
        next_function=None,
        yes_message="<b>Great.</b> Let’s move on to the next step."
    ):
        move_to_check(check_number)

        with main_output:
            main_output.clear_output()

            question = widgets.HTML(question_html)
            yes_button = widgets.Button(description="Yes", button_style="success")
            no_button = widgets.Button(description="No", button_style="warning")
            output = widgets.Output()

            def on_yes_clicked(button):
                with output:
                    output.clear_output()
                    display(widgets.HTML(yes_message))

                    if next_function is not None:
                        next_step_button(next_function)
                    else:
                        finish_button()

            def on_no_clicked(button):
                with output:
                    output.clear_output()
                    guidance_function()

                    if next_function is not None:
                        next_step_button(next_function)
                    else:
                        finish_button()

            yes_button.on_click(on_yes_clicked)
            no_button.on_click(on_no_clicked)

            display(question)
            display(widgets.HBox([yes_button, no_button]))
            display(output)

    def ask_format_check():
        def guidance():
            display(widgets.HTML("""
            <b>Why this matters:</b><br>
            AutoGluon time series data should usually be in <b>long format</b>, with one row per timestamp per item.
            If your data is already formatted this way from the previous checks, you do not need to use <code>.melt()</code>.
            """))

            display(widgets.HTML("""
            <b>Long format — usually ready for AutoGluon:</b>
            <pre><code>item_id    timestamp    target
store_1    2024-01-01   120
store_1    2024-01-02   135
store_2    2024-01-01    90
store_2    2024-01-02   102</code></pre>
            """))

            display(widgets.HTML("""
            <b>Wide format — may need <code>.melt()</code>:</b>
            <pre><code>timestamp    store_1    store_2    store_3
2024-01-01   120        90         140
2024-01-02   135        102        155</code></pre>
            """))

            display(widgets.HTML("""
            <b>Suggested code if your data is wide:</b>
            """))

            code_block("""
your_data_long = your_data.melt(
    id_vars=date_column,
    var_name=item_id_column,
    value_name=target_column
)

your_data = your_data_long
""")

            display(widgets.HTML("""
            <b>Key point:</b><br>
            Only use <code>.melt()</code> if your data is wide. If your data already has one row per timestamp per item, skip this and continue.
            """))

        render_check(
            1,
            """
            <b>Is your data already in the right row format for AutoGluon?</b><br>
            <span style="color: gray;">
            Check whether your data is already in long format, or whether it needs reshaping with <code>.melt()</code>.
            </span>
            """,
            guidance,
            ask_datetime_check
        )

    def ask_datetime_check():
        def guidance():
            display(widgets.HTML("""
            <b>Suggested code:</b>
            """))

            code_block("""
date_column = "your_date_column"

your_data[date_column] = pd.to_datetime(your_data[date_column])
""")

            display(widgets.HTML("""
            <b>Why this matters:</b><br>
            Dates often arrive as text. Converting them to <code>datetime</code> lets Python recognise them as real dates, which is needed for sorting, time-based splitting, frequency handling, and forecasting.
            """))

            display(widgets.HTML("""
            <b>Example:</b>
            """))

            code_block("""
date_column = "date"

your_data[date_column] = pd.to_datetime(your_data[date_column])
""")

        render_check(
            2,
            """
            <b>Have you converted the date column to datetime?</b><br>
            <span style="color: gray;">
            The date column should be a real datetime column, not just text.
            </span>
            """,
            guidance,
            ask_sort_check
        )

    def ask_sort_check():
        def guidance():
            display(widgets.HTML("""
            <b>Suggested code:</b>
            """))

            code_block("""
your_data = your_data.sort_values(by=date_column)
""")

            display(widgets.HTML("""
            <b>Why this matters:</b><br>
            Time series data must be in chronological order. If dates are out of order, the model may learn from a scrambled sequence and your train/test split may not properly represent past predicting future.
            """))

            display(widgets.HTML("""
            <b>Optional check:</b>
            """))

            code_block("""
your_data[[date_column]].head()
your_data[[date_column]].tail()
""")

        render_check(
            3,
            """
            <b>Have you sorted the data by date?</b><br>
            <span style="color: gray;">
            Your rows should be ordered from earliest to latest.
            </span>
            """,
            guidance,
            ask_item_id_check
        )

    def ask_item_id_check():
        def guidance():
            display(widgets.HTML("""
            <b>Suggested code for a single time series:</b>
            """))

            code_block("""
item_id_column = "item_id"

your_data[item_id_column] = "single_series"
""")

            display(widgets.HTML("""
            <b>Why this matters:</b><br>
            AutoGluon expects an <code>item_id</code> because it can model many time series at once. Even if you only have one series, you still need to tell AutoGluon that all rows belong to the same series.
            """))

            display(widgets.HTML("""
            <b>For multiple time series:</b><br>
            Your item ID might already be a column such as <code>store_id</code>, <code>product_id</code>, <code>location</code>, or <code>sensor_id</code>.
            """))

            code_block("""
item_id_column = "store_id"
""")

        render_check(
            4,
            """
            <b>Have you added or confirmed the item_id column?</b><br>
            <span style="color: gray;">
            The item ID identifies which time series each row belongs to.
            </span>
            """,
            guidance,
            ask_schema_check
        )

    def ask_schema_check():
        def guidance():
            display(widgets.HTML("""
            <b>AutoGluon time series schema:</b>
            <pre><code>item_id      timestamp      target
series_1     2024-01-01     123
series_1     2024-01-02     130
series_2     2024-01-01      98</code></pre>
            """))

            display(widgets.HTML("""
            <b>Suggested code:</b>
            """))

            code_block("""
target_column = "your_target_column"

your_data[[item_id_column, date_column, target_column]].head()
""")

            display(widgets.HTML("""
            <b>Why this matters:</b><br>
            AutoGluon needs to know three things: which series each row belongs to, when the observation happened, and what value you want to forecast.
            """))

            display(widgets.HTML("""
            <b>Important:</b><br>
            For AutoGluon, the timestamp must be available as a <b>column</b> when converting to a <code>TimeSeriesDataFrame</code>. If you used it as an index during EDA, reset the index first.
            """))

            code_block("""
your_data = your_data.reset_index()
""")

        render_check(
            5,
            """
            <b>Does your dataset have the columns AutoGluon needs?</b><br>
            <span style="color: gray;">
            You should have an item ID column, a timestamp column, and a target column.
            </span>
            """,
            guidance,
            ask_timeseries_dataframe_check
        )

    def ask_timeseries_dataframe_check():
        def guidance():
            display(widgets.HTML("""
            <b>Suggested import:</b>
            """))

            code_block("""
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
""")

            display(widgets.HTML("""
            <b>Suggested code:</b>
            """))

            code_block("""
train_data = TimeSeriesDataFrame.from_data_frame(
    your_data,
    id_column=item_id_column,
    timestamp_column=date_column
)

train_data.head()
""")

            display(widgets.HTML("""
            <b>Why this matters:</b><br>
            AutoGluon does not use an ordinary pandas DataFrame directly for time series forecasting. 
            <code>TimeSeriesDataFrame</code> tells AutoGluon which column identifies the series and which column identifies the timestamp.
            """))

        render_check(
            6,
            """
            <b>Have you converted your data to a TimeSeriesDataFrame?</b><br>
            <span style="color: gray;">
            This is the AutoGluon-specific data structure used for time series modelling.
            </span>
            """,
            guidance,
            ask_prediction_length_check
        )

    def ask_prediction_length_check():
        def guidance():
            display(widgets.HTML("""
            <b>Suggested code:</b>
            """))

            code_block("""
prediction_length = 12
""")

            display(widgets.HTML("""
            <b>Why this matters:</b><br>
            <code>prediction_length</code> is the forecast horizon. It tells AutoGluon how many future time steps to predict.
            """))

            display(widgets.HTML("""
            <b>Examples:</b><br>
            <ul>
                <li>Daily data + <code>prediction_length = 12</code> means forecast the next 12 days.</li>
                <li>Monthly data + <code>prediction_length = 12</code> means forecast the next 12 months.</li>
                <li>Weekly data + <code>prediction_length = 4</code> means forecast the next 4 weeks.</li>
            </ul>
            """))

        render_check(
            7,
            """
            <b>Have you set the prediction length?</b><br>
            <span style="color: gray;">
            This defines how far into the future your forecast should go.
            </span>
            """,
            guidance,
            ask_train_test_split_check
        )

    def ask_train_test_split_check():
        def guidance():
            display(widgets.HTML("""
            <b>Suggested code:</b>
            """))

            code_block("""
train = train_data.slice_by_timestep(None, -prediction_length)
test = train_data.slice_by_timestep(-prediction_length, None)
""")

            display(widgets.HTML("""
            <b>Why this matters:</b><br>
            Forecasting should be tested on future data. This split trains on the earlier part of the series and holds back the final time periods for testing.
            """))

            display(widgets.HTML("""
            <b>Avoid random splitting for basic time series forecasting:</b>
            """))

            code_block("""
# Avoid this for basic time series forecasting
from sklearn.model_selection import train_test_split

train, test = train_test_split(
    train_data,
    test_size=0.2,
    shuffle=True
)
""")

        render_check(
            8,
            """
            <b>Have you created a time-based train/test split?</b><br>
            <span style="color: gray;">
            The test data should come after the training data in time.
            </span>
            """,
            guidance,
            ask_predictor_setup_check
        )

    def ask_predictor_setup_check():
        def guidance():
            display(widgets.HTML("""
            <b>Suggested code:</b>
            """))

            code_block("""
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target=target_column,
    eval_metric="MASE",
    freq="D"
)
""")

            display(widgets.HTML("""
            <b>Why this matters:</b><br>
            The predictor defines the main modelling setup: how far ahead to forecast, what value to forecast, how models are scored, and the time frequency.
            """))

            display(widgets.HTML("""
            <b>Parameter guide:</b><br>
            <ul>
                <li><code>prediction_length</code>: how far ahead to forecast.</li>
                <li><code>target</code>: the value being forecast.</li>
                <li><code>eval_metric</code>: how model quality is judged.</li>
                <li><code>freq</code>: the time frequency, such as <code>"D"</code> for daily.</li>
            </ul>
            """))

            display(widgets.HTML("""
            <b>MASE note:</b><br>
            MASE is a forecasting error metric. Lower is better.
            """))

        render_check(
            9,
            """
            <b>Have you set up the TimeSeriesPredictor?</b><br>
            <span style="color: gray;">
            This creates the AutoGluon forecasting object.
            </span>
            """,
            guidance,
            ask_fit_check
        )

    def ask_fit_check():
        def guidance():
            display(widgets.HTML("""
            <b>Suggested code:</b>
            """))

            code_block("""
predictor.fit(train, time_limit=30)
""")

            display(widgets.HTML("""
            <b>Why this matters:</b><br>
            The model should learn from the training data only. The test data is held back so you can evaluate whether the model can forecast unseen future values.
            """))

        render_check(
            10,
            """
            <b>Have you fitted the predictor on the training data?</b><br>
            <span style="color: gray;">
            Fit the model on <code>train</code>, not <code>test</code>.
            </span>
            """,
            guidance,
            ask_leaderboard_check
        )

    def ask_leaderboard_check():
        def guidance():
            display(widgets.HTML("""
            <b>Suggested code:</b>
            """))

            code_block("""
leaderboard = predictor.leaderboard()
leaderboard
""")

            display(widgets.HTML("""
            <b>Why this matters:</b><br>
            AutoGluon tries multiple forecasting models. The leaderboard helps you see which models performed best according to the chosen evaluation metric.
            """))

            display(widgets.HTML("""
            <b>What to look for:</b><br>
            <ul>
                <li>Which model performed best?</li>
                <li>Are the top models close together?</li>
                <li>Is the performance meaningfully better than simple baselines?</li>
            </ul>
            """))

        render_check(
            11,
            """
            <b>Have you compared models with the leaderboard?</b><br>
            <span style="color: gray;">
            The leaderboard shows how different models performed.
            </span>
            """,
            guidance,
            ask_forecast_check
        )

    def ask_forecast_check():
        def guidance():
            display(widgets.HTML("""
            <b>Suggested code:</b>
            """))

            code_block("""
predictions = predictor.predict(train)

predictions.head()
""")

            display(widgets.HTML("""
            <b>Why this matters:</b><br>
            This uses the trained model to forecast the next <code>prediction_length</code> time steps after the training data.
            """))

        render_check(
            12,
            """
            <b>Have you generated forecasts?</b><br>
            <span style="color: gray;">
            Forecasts are the model’s predicted future values.
            </span>
            """,
            guidance,
            ask_plot_check
        )

    def ask_plot_check():
        def guidance():
            display(widgets.HTML("""
            <b>Suggested code:</b>
            """))

            code_block("""
predictor.plot(
    test,
    predictions,
    quantile_levels=[0.4, 0.6],
    max_history_length=200,
    item_ids=["single_series"]
)
""")

            display(widgets.HTML("""
            <b>For multiple series:</b><br>
            Replace <code>"single_series"</code> with one of your actual item IDs, or remove <code>item_ids</code> to let AutoGluon choose examples.
            """))

            code_block("""
predictor.plot(
    test,
    predictions,
    quantile_levels=[0.4, 0.6],
    max_history_length=200
)
""")

            display(widgets.HTML("""
            <b>Why this matters:</b><br>
            Metrics are useful, but a plot helps you see whether the forecast behaves sensibly against the held-back test period.
            """))

            display(widgets.HTML("""
            <b>What to look for:</b><br>
            <ul>
                <li>Does the forecast follow the actual values?</li>
                <li>Is it consistently too high or too low?</li>
                <li>Does it miss sudden spikes or drops?</li>
                <li>Are the uncertainty bands reasonable?</li>
            </ul>
            """))

        render_check(
            13,
            """
            <b>Have you plotted the forecast against the test data?</b><br>
            <span style="color: gray;">
            This helps you visually inspect forecast quality.
            </span>
            """,
            guidance,
            ask_naive_baseline_check
        )

    def ask_naive_baseline_check():
        def guidance():
            display(widgets.HTML("""
            <b>Suggested code:</b><br>
            Generate forecasts from the best model and a seasonal naive baseline.
            """))

            code_block("""
best_forecast = predictor.predict(train)
naive_forecast = predictor.predict(train, model="SeasonalNaive")
""")

            display(widgets.HTML("""
            <b>Select one series to inspect:</b>
            """))

            code_block("""
selected_item_id = "single_series"

actual = test.loc[selected_item_id][target_column]
best = best_forecast.loc[selected_item_id]["mean"]
naive = naive_forecast.loc[selected_item_id]["mean"]
""")

            display(widgets.HTML("""
            <b>Compare absolute errors:</b>
            """))

            code_block("""
import pandas as pd

comparison = pd.DataFrame({
    "actual": actual.values,
    "best": best.values,
    "naive": naive.values,
}, index=actual.index)

comparison["best_abs_error"] = (
    comparison["actual"] - comparison["best"]
).abs()

comparison["naive_abs_error"] = (
    comparison["actual"] - comparison["naive"]
).abs()

comparison
""")

            code_block("""
comparison[["best_abs_error", "naive_abs_error"]]
""")

            display(widgets.HTML("""
            <b>Why this matters:</b><br>
            A model is only useful if it improves on a simple baseline. Before trusting a more complex model, check whether it beats a simple one.
            """))

            display(widgets.HTML("""
            <b>How to interpret it:</b><br>
            <ul>
                <li>If <code>best_abs_error</code> is usually lower than <code>naive_abs_error</code>, the AutoGluon model is generally making better predictions.</li>
                <li>If the naive model is similar or better, the best model may not be adding enough value.</li>
            </ul>
            """))

        render_check(
            14,
            """
            <b>Have you compared the best model against a naive baseline?</b><br>
            <span style="color: gray;">
            This checks whether the AutoGluon model improves on a simple forecasting approach.
            </span>
            """,
            guidance,
            ask_conclusion_check
        )

    def ask_conclusion_check():
        def guidance():
            display(widgets.HTML("""
            <b>Suggested modelling conclusion questions:</b><br>
            <ul>
                <li>Which model performed best on the leaderboard?</li>
                <li>Did the model beat the naive baseline?</li>
                <li>Did the forecast follow the actual test data?</li>
                <li>Where did the model make the biggest errors?</li>
                <li>Is the model good enough for the forecasting goal?</li>
                <li>What would you try next?</li>
            </ul>
            """))

            display(widgets.HTML("""
            <b>Example conclusion:</b><br>
            <i>
            The AutoGluon model produced forecasts for the next prediction window and was compared against a SeasonalNaive baseline. 
            The forecast generally followed the test data, but some spikes were missed. 
            The best model should be trusted only if it improves meaningfully over the naive baseline and the forecast errors are acceptable for the project goal.
            </i>
            """))

        render_check(
            15,
            """
            <b>Have you written down your modelling conclusions?</b><br>
            <span style="color: gray;">
            The final step is to explain what the model results mean, not just run the code.
            </span>
            """,
            guidance,
            next_function=None,
            yes_message="<b>Great.</b> Finish the checklist when you are ready."
        )

    title = widgets.HTML("<b>Low-code time series modelling checklist</b>")

    layout = widgets.HBox(
        [
            widgets.VBox([title, main_output], layout=widgets.Layout(width="70%")),
            widgets.VBox([checklist_output], layout=widgets.Layout(width="30%"))
        ],
        layout=widgets.Layout(width="100%", align_items="flex-start")
    )

    display(layout)

    update_checklist()
    ask_format_check()


