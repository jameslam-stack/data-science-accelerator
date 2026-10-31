import traceback
from typing import Any, Callable, Optional

import ipywidgets as widgets
from IPython.display import display


class GuidedActivity:
    """
    Reusable guided coding activity for Jupyter notebooks.

    Each activity supplies:

    - a title;
    - an introduction;
    - a list of steps;
    - an optional starting namespace;
    - an optional preview function.

    Each step should contain:

    {
        "title": str,
        "instruction": str,
        "placeholder": str,
        "validator": callable,
    }

    The validator must accept the learner namespace and return True or False.
    """

    def __init__(
        self,
        title: str,
        introduction: str,
        steps: list[dict[str, Any]],
        namespace: Optional[dict[str, Any]] = None,
        preview: Optional[Callable[[dict[str, Any]], None]] = None,
    ):
        if not steps:
            raise ValueError("GuidedActivity requires at least one step.")

        self.title = title
        self.introduction = introduction
        self.steps = steps
        self.namespace = namespace if namespace is not None else {}
        self.preview = preview

        self.current_step = 0
        self.completed_steps = [False] * len(self.steps)

        self._validate_step_definitions()
        self._create_widgets()

    # ---------------------------------------------------------
    # SETUP
    # ---------------------------------------------------------

    def _validate_step_definitions(self) -> None:
        """
        Check that every step contains the required fields.
        """

        required_fields = {
            "title",
            "instruction",
            "validator",
        }

        for index, step in enumerate(self.steps):
            missing_fields = required_fields - step.keys()

            if missing_fields:
                missing_text = ", ".join(sorted(missing_fields))

                raise ValueError(
                    f"Step {index + 1} is missing required fields: "
                    f"{missing_text}"
                )

            if not callable(step["validator"]):
                raise TypeError(
                    f"The validator for step {index + 1} must be callable."
                )

    def _create_widgets(self) -> None:
        """
        Create the main widget containers used by the activity.
        """

        self.main_area = widgets.VBox(
            layout=widgets.Layout(
                width="100%",
            )
        )

        self.heading_area = widgets.HTML()
        self.navigation_area = widgets.HTML()

        self.preview_area = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                margin="0 0 14px 0",
            )
        )

        self.activity_area = widgets.VBox(
            layout=widgets.Layout(
                width="100%",
            )
        )

        self.output_area = widgets.Output(
            layout=widgets.Layout(
                border="1px solid #d1d5db",
                padding="10px",
                width="100%",
                min_height="60px",
            )
        )

    # ---------------------------------------------------------
    # HEADING
    # ---------------------------------------------------------

    def update_heading(self) -> None:
        """
        Update the activity title and introduction.
        """

        self.heading_area.value = f"""
        <div style="
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            background: #f8fafc;
        ">
            <div style="
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 6px;
                color: #0f172a;
            ">
                {self.title}
            </div>

            <div style="
                color: #475569;
                line-height: 1.5;
            ">
                {self.introduction}
            </div>
        </div>
        """

    # ---------------------------------------------------------
    # NAVIGATION
    # ---------------------------------------------------------

    def update_navigation(self) -> None:
        """
        Update the checklist and progress bar.
        """

        rows = []

        for index, step in enumerate(self.steps):
            if self.completed_steps[index]:
                symbol = "✅"
                background = "#ecfdf5"
                border = "#10b981"
                text_colour = "#065f46"

            elif index == self.current_step:
                symbol = "▶"
                background = "#eff6ff"
                border = "#3b82f6"
                text_colour = "#1e40af"

            else:
                symbol = "○"
                background = "#f9fafb"
                border = "#d1d5db"
                text_colour = "#6b7280"

            rows.append(
                f"""
                <div style="
                    padding: 9px 12px;
                    margin-bottom: 6px;
                    border-left: 4px solid {border};
                    background: {background};
                    color: {text_colour};
                    border-radius: 5px;
                ">
                    {symbol} {step["title"]}
                </div>
                """
            )

        completed_count = sum(self.completed_steps)
        percentage = int(
            completed_count / len(self.steps) * 100
        )

        self.navigation_area.value = f"""
        <div style="
            border: 1px solid #d1d5db;
            padding: 14px;
            border-radius: 8px;
            margin-bottom: 14px;
        ">
            <div style="
                font-size: 17px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #111827;
            ">
                Activity progress:
                {completed_count}/{len(self.steps)}
            </div>

            <div style="
                width: 100%;
                height: 8px;
                background: #e5e7eb;
                border-radius: 10px;
                margin-bottom: 12px;
                overflow: hidden;
            ">
                <div style="
                    width: {percentage}%;
                    height: 100%;
                    background: #2563eb;
                    transition: width 0.3s;
                "></div>
            </div>

            {''.join(rows)}
        </div>
        """

    # ---------------------------------------------------------
    # PREVIEW
    # ---------------------------------------------------------

    def update_preview(self) -> None:
        """
        Display optional supporting content, such as a dataset.
        """

        self.preview_area.clear_output()

        if self.preview is None:
            return

        with self.preview_area:
            try:
                self.preview(self.namespace)

            except Exception:
                print("The activity preview could not be displayed.")
                traceback.print_exc()

    # ---------------------------------------------------------
    # CODE EXECUTION
    # ---------------------------------------------------------

    def run_code(self, code: str) -> bool:
        """
        Execute learner code inside the activity namespace.

        Returns True when execution succeeds and False when it fails.
        """

        with self.output_area:
            self.output_area.clear_output()

            if not code.strip():
                print("Enter some Python code before running the step.")
                return False

            try:
                exec(code, self.namespace)

            except Exception:
                traceback.print_exc()
                return False

            return True

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    def validate_current_step(self) -> bool:
        """
        Run the validator belonging to the current step.
        """

        step = self.steps[self.current_step]
        validator = step["validator"]

        try:
            return bool(validator(self.namespace))

        except Exception:
            with self.output_area:
                print("The step validator encountered an error.")
                traceback.print_exc()

            return False

    # ---------------------------------------------------------
    # STEP DISPLAY
    # ---------------------------------------------------------

    def show_step(self) -> None:
        """
        Render the current activity step.
        """

        self.update_heading()
        self.update_navigation()
        self.update_preview()
        self.output_area.clear_output()

        if self.current_step >= len(self.steps):
            self.show_completion()
            return

        step = self.steps[self.current_step]

        heading = widgets.HTML(
            f"""
            <div style="
                margin-bottom: 8px;
            ">
                <h3 style="
                    margin-bottom: 5px;
                    color: #111827;
                ">
                    {step["title"]}
                </h3>

                <div style="
                    color: #374151;
                    line-height: 1.5;
                ">
                    {step["instruction"]}
                </div>
            </div>
            """
        )

        code_input = widgets.Textarea(
            placeholder=step.get("placeholder", ""),
            layout=widgets.Layout(
                width="100%",
                height=step.get("input_height", "150px"),
            ),
        )

        run_button = widgets.Button(
            description="Run step",
            button_style="success",
            icon="play",
            tooltip="Run and validate this step",
        )

        next_button = widgets.Button(
            description=(
                "Finish activity"
                if self.current_step == len(self.steps) - 1
                else "Next step"
            ),
            button_style="info",
            icon="arrow-right",
            disabled=True,
            tooltip="Continue to the next step",
        )

        feedback = widgets.HTML()

        def handle_run(_button) -> None:
            executed = self.run_code(code_input.value)

            if not executed:
                feedback.value = """
                <div style="
                    color: #b91c1c;
                    margin-top: 8px;
                ">
                    The code could not be completed.
                    Review the output and try again.
                </div>
                """
                return

            if not self.validate_current_step():
                feedback.value = """
                <div style="
                    background: #fff7ed;
                    border-left: 4px solid #f97316;
                    padding: 10px;
                    margin-top: 8px;
                ">
                    The code ran, but it has not completed this step yet.
                    Review the instruction and try again.
                </div>
                """
                return

            self.completed_steps[self.current_step] = True

            feedback.value = """
            <div style="
                background: #ecfdf5;
                border-left: 4px solid #10b981;
                padding: 10px;
                margin-top: 8px;
            ">
                ✅ Step completed.
            </div>
            """

            run_button.disabled = True
            code_input.disabled = True
            next_button.disabled = False

            self.update_navigation()

        def handle_next(_button) -> None:
            self.current_step += 1
            self.show_step()

        run_button.on_click(handle_run)
        next_button.on_click(handle_next)

        button_row = widgets.HBox(
            [run_button, next_button],
            layout=widgets.Layout(
                gap="8px",
                margin="8px 0",
            ),
        )

        self.activity_area.children = [
            heading,
            code_input,
            button_row,
            feedback,
            widgets.HTML(
                """
                <div style="
                    font-weight: bold;
                    margin-top: 10px;
                    margin-bottom: 4px;
                ">
                    Output
                </div>
                """
            ),
            self.output_area,
        ]

    # ---------------------------------------------------------
    # COMPLETION
    # ---------------------------------------------------------

    def show_completion(self) -> None:
        """
        Show the activity completion screen.
        """

        self.update_navigation()

        restart_button = widgets.Button(
            description="Restart activity",
            icon="refresh",
            tooltip="Start this activity again",
        )

        completion_message = widgets.HTML(
            """
            <div style="
                padding: 16px;
                background: #ecfdf5;
                border-left: 5px solid #10b981;
                border-radius: 6px;
            ">
                <b>✅ Activity complete</b><br>
                You completed all the steps.
            </div>
            """
        )

        def handle_restart(_button) -> None:
            self.reset()
            self.show_step()

        restart_button.on_click(handle_restart)

        self.activity_area.children = [
            completion_message,
            restart_button,
        ]

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------

    def reset(
        self,
        clear_namespace: bool = False,
    ) -> None:
        """
        Reset progress to the first step.

        Set clear_namespace=True to remove variables created by the learner.
        """

        self.current_step = 0
        self.completed_steps = [False] * len(self.steps)

        if clear_namespace:
            self.namespace.clear()

    # ---------------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------------

    def display(self) -> None:
        """
        Display the complete guided activity.
        """

        self.main_area.children = [
            self.heading_area,
            self.navigation_area,
            self.preview_area,
            self.activity_area,
        ]

        display(self.main_area)
        self.show_step()