import json
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

BAR_COLORS = ("#1f77b4", "#7f7f7f")


def resource_path(name):
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / name


def load_json(name):
    with open(resource_path(name), encoding="utf-8") as f:
        return json.load(f)


class DemoApp(tk.Tk):
    def __init__(self, cohorts, model):
        super().__init__()
        self.title("Sleep Cognition ML")
        self.geometry("820x700")
        self.cohorts = {cohort["name"]: cohort["configurations"] for cohort in cohorts}
        self.model = model

        controls = ttk.Frame(self, padding=12)
        controls.pack(fill="x")

        ttk.Label(controls, text="Dataset").grid(row=0, column=0, sticky="w")
        self.cohort_box = ttk.Combobox(
            controls, values=list(self.cohorts), state="readonly", width=34
        )
        self.cohort_box.grid(row=1, column=0, padx=(0, 12))
        self.cohort_box.bind("<<ComboboxSelected>>", self.update_configurations)

        ttk.Label(controls, text="Configuration").grid(row=0, column=1, sticky="w")
        self.config_box = ttk.Combobox(controls, state="readonly", width=44)
        self.config_box.grid(row=1, column=1, padx=(0, 12))

        ttk.Button(controls, text="Run", command=self.render).grid(row=1, column=2)

        self.summary = ttk.Label(self, padding=12, justify="left")
        self.summary.pack(fill="x")

        self.build_live_panel()

        self.canvas = tk.Canvas(self, background="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.cohort_box.current(0)
        self.update_configurations()
        self.render()

    def build_live_panel(self):
        panel = ttk.LabelFrame(self, text="Live model: age and BMI", padding=12)
        panel.pack(side="bottom", fill="x", padx=12, pady=(0, 12))

        ttk.Label(panel, text="Age (years)").grid(row=0, column=0, sticky="w")
        self.age_entry = ttk.Entry(panel, width=10)
        self.age_entry.grid(row=1, column=0, padx=(0, 12))

        ttk.Label(panel, text="BMI (optional)").grid(row=0, column=1, sticky="w")
        self.bmi_entry = ttk.Entry(panel, width=10)
        self.bmi_entry.grid(row=1, column=1, padx=(0, 12))

        ttk.Button(panel, text="Predict", command=self.predict).grid(row=1, column=2)

        self.prediction = ttk.Label(panel, justify="left", wraplength=760)
        self.prediction.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def predict(self):
        model = self.model
        try:
            age = float(self.age_entry.get())
        except ValueError:
            self.prediction.configure(text="Enter a numeric age.")
            return
        bmi_text = self.bmi_entry.get().strip()
        try:
            bmi = float(bmi_text) if bmi_text else model["bmi_median"]
        except ValueError:
            self.prediction.configure(text="Enter a numeric BMI or leave it blank.")
            return

        z_age = (age - model["age_mean"]) / model["age_scale"]
        z_bmi = (bmi - model["bmi_mean"]) / model["bmi_scale"]
        logit = (
            model["intercept"] + model["age_coefficient"] * z_age + model["bmi_coefficient"] * z_bmi
        )
        probability = 1 / (1 + math.exp(-logit))
        base_rate = model["positives"] / model["patients"]

        text = (
            f"Estimated probability within this dataset: {probability:.1%}. "
            f"The dataset's base rate is {base_rate:.1%}. "
            f"This logistic regression uses only age and BMI "
            f"(cross-validated AUROC {model['cv_auroc_mean']:.3f} ± {model['cv_auroc_std']:.3f}, "
            f"{model['patients']} patients). "
            f"It is a research demonstration and not medical advice."
        )
        outside_age = not model["age_range"][0] <= age <= model["age_range"][1]
        outside_bmi = bool(bmi_text) and not model["bmi_range"][0] <= bmi <= model["bmi_range"][1]
        if outside_age or outside_bmi:
            text += " The values entered fall outside the typical range of the training data, so the estimate is unreliable."
        self.prediction.configure(text=text)

    def update_configurations(self, _event=None):
        configurations = self.cohorts[self.cohort_box.get()]
        self.config_box.configure(values=[item["name"] for item in configurations])
        self.config_box.current(0)

    def selected_configuration(self):
        for item in self.cohorts[self.cohort_box.get()]:
            if item["name"] == self.config_box.get():
                return item
        return None

    def render(self):
        configuration = self.selected_configuration()
        self.canvas.delete("all")
        self.summary.configure(
            text=(
                f"Dataset: {self.cohort_box.get()}\n"
                f"Model: {configuration['model']}\n"
                f"AUROC: {configuration['auroc']}"
            )
        )
        rows = configuration.get("importance")
        if not rows:
            self.canvas.create_text(
                10,
                10,
                anchor="nw",
                text="Feature importance is shown for the 20-seed and logistic regression views. Select one of those to see it.",
            )
            return
        self.canvas.create_text(
            10,
            8,
            anchor="nw",
            text=configuration["importance_label"],
            font=("TkDefaultFont", 10, "bold"),
        )
        peak = max(row[1] for row in rows)
        top, step, label_edge, bar_width = 36, 28, 240, 360
        for index, row in enumerate(rows):
            y = top + index * step
            length = bar_width * row[1] / peak
            color = BAR_COLORS[0] if index < 3 else BAR_COLORS[1]
            self.canvas.create_text(label_edge, y + 10, anchor="e", text=row[0])
            self.canvas.create_rectangle(
                label_edge + 10, y, label_edge + 10 + length, y + 20, fill=color, outline=""
            )
            value = f"{row[1]:.4f}" if len(row) == 2 else f"{row[1]:.4f} ± {row[2]:.4f}"
            self.canvas.create_text(label_edge + 18 + length, y + 10, anchor="w", text=value)


def main():
    cohorts = load_json("demo_results.json")["cohorts"]
    model = load_json("age_bmi_model.json")
    DemoApp(cohorts, model).mainloop()


if __name__ == "__main__":
    main()
