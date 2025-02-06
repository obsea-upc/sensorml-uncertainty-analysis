import importlib
import json
import rich
import tkinter as tk
from argparse import ArgumentParser

import pandas as pd
import os
from tkinter import ttk, W, E, messagebox, filedialog
from PIL import Image, ImageTk
import traceback
import SensorML

valid_time_formats = ["%Y-%m-%d", "%Y/%m/%d", "%Y/%m/%d %H:%m:%s"]

p07_climate_and_forecast = pd.read_csv("vocabs/P07.csv")
p06_units = pd.read_csv("vocabs/P06.csv")
p02_broad_variables = pd.read_csv("vocabs/P02.csv")
l05_sensor_types = pd.read_csv("vocabs/L05.csv")
l22_sensor_models = pd.read_csv("vocabs/L22.csv")
l35_manufacturers = pd.read_csv("vocabs/L35.csv")


GREEN = "#3D9140"
RED = "#DC143C"
GREY = "azure4"
YELLOW = "#FF6103"
LIGHT_BLUE = "azure2"
LIGHT_YELLOW = "blanchedalmond"
TICK = u"\u2713"
WARN = u"\u26A0"

class VocabComboBox:
    def __init__(self, root, name, df, row, columnspan=4, padx=10, pady=10, width=60):
        """
        Creates a combobox to select a value from a vocabulary
        :param root:
        :param name:
        :param df:
        :param row:
        :param columnspan:
        :param padx:
        :param pady:
        :param width:
        """
        self.items = df["prefLabel"].to_list()
        self.name = name
        self.value = tk.StringVar()
        self.root = root
        self.df = df
        # Create a combobox
        tk.Label(self.root, text=name, borderwidth=1).grid(row=row, column=0, padx=padx, pady=pady, sticky=E)
        self.combobox = ttk.Combobox(self.root, textvariable=self.value, width=width)
        self.combobox['values'] = self.items  # Set the initial list
        self.combobox.grid(row=row, column=1, columnspan=columnspan, padx=padx, pady=pady, sticky=W)
        self.combobox.bind("<KeyRelease>", self.update_list)  # Trigger filtering as user types
        self.combobox.bind("<KeyPress>", self.validate_list)  # Trigger filtering as user types

        self.combobox.bind("<<ComboboxSelected>>", self.validate_list)  # Handle selection

        self.valtext = tk.Label(self.root, text=WARN, foreground=YELLOW)
        self.valtext.grid(row=row, column=1+columnspan, columnspan=columnspan, padx=padx, pady=pady, sticky=W)

        # self.vocab_label = tk.Label(self.root, text=name)
        # self.vocab_label.grid(row=row + 1, column=0, padx=padx, pady=0, sticky=E)
        self.vocab_code = tk.Label(self.root, text="code:", foreground=GREY)
        self.vocab_code.grid(row=row + 1, column=1, padx=padx, pady=pady, sticky=W)
        self.vocab_uri = tk.Label(self.root, text="uri:", foreground=GREY)
        self.vocab_uri.grid(row=row + 1, column=2, padx=padx, pady=pady, columnspan=520, sticky=W)

    def update_list(self, event):
        """Filter the drop-down list based on user input and open it automatically."""
        typed_text = self.value.get().lower()
        filtered_items = [item for item in self.items if typed_text in item.lower()]

        # Update the combobox's values with the filtered list
        self.combobox['values'] = filtered_items if filtered_items else ["No match found"]

        # Open the drop-down list automatically
        self.combobox.event_generate('<Down>')
        # validate it afterwards
        self.validate_list(event)

        # Return the focus back to the entry field
        self.root.after(100, lambda: self.combobox.focus())


    def on_select(self, event):
        """Handle selection from the drop-down list."""

        self.validate_list(event)

    def validate_list(self, event):

        value = self.value.get()
        df = self.df
        row = df[df["prefLabel"] == value]
        if len(row) == 1:
            code = row["id"].values[0].split("::")[-1]
            uri = row["uri"].values[0]
            self.vocab_uri.config(text=f"uri: {uri}")
            self.vocab_code.config(text=f"code: {code}")
        else:
            self.vocab_uri.config(text=f"uri: not fuond")
            self.vocab_code.config(text=f"code: not found")

        if not value:
            self.valtext.config(text="", fg=GREEN)  # clear text

        if value in df["prefLabel"].to_list():
            self.valtext.config(text=TICK, fg=GREEN)
        else:
            self.valtext.config(text=WARN, fg=YELLOW)


    def __repr__(self):
        return f"""{self.name}
    label: {self.value.get()}
     code: {self.vocab_code.cget("text")}
      uri: {self.vocab_uri.cget("text")}"""

    def process_label(self, label):
        print(label.cget("text"))
        try:
            l = label.cget("text").split(": ")[1]
        except IndexError:
            return self.value.get()

        if l == "not found":
            return self.value.get()
        return l


    def get(self):
        return {
            "label": self.value.get(),
            "code": self.process_label(self.vocab_code),
            "uri": self.process_label(self.vocab_uri)
        }

    def vocab_lookup(self, d):
        """fills a vocab element based on one of the values"""
        assert(isinstance(d, dict)), f"expected dict, got {type(d)}"

        # Force missing elements
        for e in ["uri", "label", "code"]:
            if e not in d.keys():
                d[e] = ""

        df = self.df
        if d["uri"]:
            row = df[df["uri"] == d["uri"]]
        elif d["label"]:
            row = df[df["prefLabel"] == d["label"]]
        elif d["code"]:
            row = df[df["code"] == d["id"]]
        else:
            raise ValueError("value not found!")

        return {
            "label": row["prefLabel"].values[0],
            "code": row["id"].values[0],
            "uri": row["uri"].values[0]
        }


    def put(self, doc):
        doc = self.vocab_lookup(doc)
        self.value.set(doc["label"])
        self.validate_list(None)



class TextField:
    def __init__(self, root, label, row, padx=10, pady=10, validation="string"):
        self.root = root
        self.label = label

        tk.Label(root, text=label).grid(row=row, column=0, padx=padx, pady=pady, sticky=E)
        self.entry = tk.Entry(root)
        self.entry.grid(row=row, column=1, padx=padx, pady=pady, sticky=W)

        self.validation = tk.Label(root, text=WARN, fg=YELLOW) # east
        self.validation.grid(row=row, column=2, padx=padx, pady=pady, sticky=W)

        self.validation_handler = None
        if validation == "string":
            self.validation_handler = self.validate_string
        elif validation in ["date", "datetime", "time", "timestamp"]:
            self.validation_handler = self.validate_datetime
        elif validation == "float":
            self.validation_handler = self.validate_float

        self.entry.bind("<KeyRelease>", self.validate)  # Trigger filtering as user types

    def get(self):
        return self.entry.get()

    def put(self, value):
        self.entry.delete(0, tk.END)  # Clears the text inside the Entry
        self.entry.insert(0, value)
        self.validate(None)


    def validate(self, event):
        """call validate via the validation handler"""
        validated = self.validation_handler(self.entry.get())
        if validated:
            self.validation.config(text=u"\u2713 valid", fg=GREEN)
        else:
            self.validation.config(text=u"\u26A0 not valid", fg=RED)

    @staticmethod
    def validate_float(value):
        """will be true if string not empty"""
        try:
            float(value)
        except ValueError:
            return False
        return True

    # Validation methods
    @staticmethod
    def validate_string(value):
        if value:
            return True

    @staticmethod
    def validate_datetime(value):
        t = None
        for time_format in valid_time_formats:
            try:
                t = pd.to_datetime(value, format=time_format)
                break
            except ValueError:
                pass
        return not pd.isnull(t)



    def __repr__(self):
        return f"{self.label}: {self.entry.get()}"


class FloatField(TextField):
    def __init__(self, root, label, row, padx=10, pady=10):
        TextField.__init__(self, root, label, row, padx, pady, validation="float")

    def get(self):
        return float(self.entry.get())

    def put(self, value):
        self.entry.delete(0, tk.END)  # Clears the text inside the Entry
        self.entry.insert(0, str(value))
        self.validate(None)


class CalibrationArray:
    def __init__(self, root, row, padx=5, pady=5):
        # Create the main window
        self.pady = pady
        self.padx = padx
        self.root = root
        self.calibration_rows = [] # list of dicts containing each row

        self.row_count = row
        self.row_count += 1

        l = tk.Label(root, text="Calibration Values", font=("Helvetica", 10, "bold"))

        l.grid(row=self.row_count, column=0, columnspan=2, padx=padx, pady=20, sticky="s")

        # Add button to add rows
        btn_add_row = tk.Button(root, text="+ Add Row", command=self.add_calibration_row)
        btn_add_row.grid(row=self.row_count, column=2, columnspan=2,  padx=padx, pady=pady, sticky=W)
        btn_del_row = tk.Button(root, text="- Delete Row", command=self.del_calibration_row)
        btn_del_row.grid(row=self.row_count, column=3, columnspan=2,  padx=padx, pady=pady, sticky=W)

        self.row_count += 1

        self.col_names = ["reference", "measurement", "correction", "expanded uncertainty"]
        for i, text in enumerate(self.col_names):
            tk.Label(root, text=text).grid(row=self.row_count, column=i, padx=padx, pady=pady)
        self.row_count += 1


    def add_calibration_row(self):
        """Add a new row with 4 input fields."""
        row = {}

        for i, name in enumerate(self.col_names):
            entry = tk.Entry(self.root, width=10)
            if name == "correction":
                entry.config(bg=LIGHT_BLUE)
            entry.grid(row=self.row_count, column=i, padx=self.padx, pady=self.pady)
            entry.bind("<KeyRelease>", self.update_array)  # Trigger filtering as user types
            row[name] = entry

        self.row_count += 1
        self.calibration_rows.append(row)

    def del_calibration_row(self):
        """Add a new row with 4 input fields."""

        if len(self.calibration_rows) > 0:
            row = self.calibration_rows[-1]
            self.calibration_rows = self.calibration_rows[:-1]
            for key, entry in row.items():
                entry.destroy()
            self.row_count -= 1


    @staticmethod
    def validate_float_entry(entry):
        """Returns True if entry has a float or null string, otherwise False"""
        v = entry.get()
        if not v:
            entry.configure(fg="black")
            return False
        try:
            v = float(v)
        except ValueError:
            # Set red!
            entry.configure(fg=RED)
            return False

        entry.configure(fg="black")
        return v


    @staticmethod
    def get_decimal_precision(inp):
        if isinstance(inp, float):
            inp = str(inp)

        elif isinstance(inp, str):
            pass
        else:
            raise ValueError("Unimplemented")


        if "." not in inp:
            return 0
        return len(inp.split(".")[1])


    def update_array(self, event):
        data = {}
        for name in self.col_names:
            data[name] = []  # empty list

        for row in self.calibration_rows:
            meas = self.validate_float_entry(row["measurement"])
            ref = self.validate_float_entry(row["reference"])
            corr = self.validate_float_entry(row["correction"])

            # If both meas and reference are float (they have been filled) AND the event was triggered by MEAS or FLOAT
            # Entry objects, automatically fill the correction
            if (isinstance(meas, float) and isinstance(ref, float) and
                    id(event.widget) in [id(row["measurement"]), id(row["reference"])]):
                corr = ref - meas
                decimals = max(self.get_decimal_precision(meas), self.get_decimal_precision(ref))
                corr_f = str(round(float(corr), decimals))
                row["correction"].delete(0, tk.END)
                row["correction"].insert(0, corr_f)
                row["correction"].config(bg=LIGHT_BLUE)

            elif id(event.widget) == id(row["correction"]):
                row["correction"].config(bg=LIGHT_YELLOW)


            self.validate_float_entry(row["expanded uncertainty"])

    def get(self):
        array = []
        for row in self.calibration_rows:
            simple_row = []
            for field in row.values():
                v = float(field.get())
                simple_row.append(v)
            array.append(simple_row)
        return array

    def put(self, array):
        for i in range(len(array)):
            self.add_calibration_row()

        for i, row in enumerate(self.calibration_rows):
            ref, meas, corr, uncert = array[i]
            row["reference"].delete(0, tk.END)
            row["reference"].insert(0, str(ref))
            row["measurement"].delete(0, tk.END)
            row["measurement"].insert(0, str(meas))
            row["correction"].delete(0, tk.END)
            row["correction"].insert(0, str(corr))
            row["expanded uncertainty"].delete(0, tk.END)
            row["expanded uncertainty"].insert(0, str(uncert))



class CalibrationSheet:
    def __init__(self, root, row, padx=5, pady=5):
        self.lab = TextField(root, "calibration lab", row, validation="string", pady=pady)
        row += 1
        self.date = TextField(root, "calibration date", row, validation="date", pady=pady)
        row += 1
        self.k = FloatField(root, "Coverage factor (k)", row, pady=pady)

        self.array = CalibrationArray(root, row, padx=padx, pady=pady)

    def get(self):
        return {
            "lab": self.lab.get(),
            "date": self.date.get(),
            "coverageFactor": self.k.get(),
            "array": self.array.get()
        }

    def put(self, doc):
        self.lab.put(doc["lab"])
        self.date.put(doc["date"])
        self.k.put(doc["coverageFactor"])
        self.array.put(doc["array"])


class GenericTab:
    def __init__(self, notebook, tabname):
        self.notebook = notebook
        tab = ttk.Frame(notebook)
        self.tab = tab
        self.tab_id = len(notebook.tabs())
        self.tabname = tabname
        notebook.add(tab, text=f"Tab {self.tab_id}")
        notebook.tab(self.tab_id, text=tabname)


class VariableTab(GenericTab):
    def __init__(self, notebook, padx=10, pady=1):
        GenericTab.__init__(self, notebook, "CHANGEME")
        row = 0
        # Create a new tab frame

        # Add content to the new tab (just a label in this example)
        self.varcode = TextField(self.tab, "variable code", row, pady=pady)
        row += 1
        self.varname = VocabComboBox(self.tab, "variable name", p02_broad_variables, row, pady=pady)
        row += 2
        self.standard_name = VocabComboBox(self.tab, "standard name", p07_climate_and_forecast, row, pady=pady)
        row += 2
        self.units = VocabComboBox(self.tab, "units", p06_units, row, pady=pady)
        row += 2

        line = tk.Frame(self.tab, height=2, width=250, bg=GREY)
        cols, rows = self.tab.grid_size()  # grid_size() returns (columns, rows)
        line.grid(row=row, column=0, columnspan=cols, padx=padx, pady=pady, sticky="ew")  # Stretch across
        row += 1

        self.calibration = CalibrationSheet(self.tab, row, pady=pady)

        self.varcode.entry.bind("<KeyRelease>", self.update_tab_name)  # Trigger filtering as user types

    def update_tab_name(self, event):
        varcode = event.widget.get()
        if varcode:
            self.notebook.tab(self.tab, text=varcode)

    def get(self):
        return {
            "varcode": self.varcode.get(),
            "varname": self.varname.get(),
            "standardName": self.standard_name.get(),
            "units": self.units.get(),
            "calibration": self.calibration.get()
        }

    def put(self, doc):
        self.varcode.put(doc["varcode"])
        # Set tab name manually
        self.notebook.tab(self.tab, text=self.varcode.get())

        self.standard_name.put(doc["standardName"])
        self.varname.put(doc["varname"])
        self.units.put(doc["units"])
        self.calibration.put(doc["calibration"])




class SerialNumber(TextField):
    def __init__(self, tab, name, row):
        TextField.__init__(self, tab, name, row)

    def get(self):
        """Overload get to prevent leading SN to be treated as serial number"""
        sn = self.entry.get()
        if sn.startswith("SN"):
            return sn[2:]
        else:
            return sn


class GeneralInfoTab(GenericTab):
    def __init__(self, parent, pady=5):
        self.parent = parent
        GenericTab.__init__(self, parent.notebook, "General Info")
        self.sensor_name = TextField(self.tab, "name", 0)
        self.long_name = TextField(self.tab, "Long name", 2)
        self.serial_number = SerialNumber(self.tab, "serial no", 1)
        self.sensor_type = VocabComboBox(self.tab, "sensor type", l05_sensor_types, 3, pady=pady)
        self.manufacturer = VocabComboBox(self.tab, "manufacturer", l35_manufacturers, 7, pady=pady)
        self.sensor_model = VocabComboBox(self.tab, "model", l22_sensor_models, 5, pady=pady)

    def __repr__(self):
        lines = []
        lines.append(self.sensor_name.__repr__())
        lines.append(self.long_name.__repr__())
        lines.append(self.serial_number.__repr__())
        lines.append(self.sensor_type.__repr__())
        lines.append(self.sensor_model.__repr__())
        return "\n".join(lines)

    def put(self, doc):
        """Fills the GeneralInfoTab from a JSON doc"""
        self.sensor_name.put(doc["sensorName"])
        self.serial_number.put(doc["serialNumber"])
        self.long_name.put(doc["longName"])
        self.sensor_model.put(doc["model"])
        self.manufacturer.put(doc["manufacturer"])
        self.sensor_type.put(doc["sensorType"])


class SensorEditor:
    def __init__(self):
        root = tk.Tk()
        root.title("SensorML Editor")
        self.variables = []


        notebook = ttk.Notebook(root)  # Create a Notebook widget to hold the tabs
        notebook.pack(fill="both", expand=True)
        self.rowcount = 0
        self.root = root

        notebook.grid(row=self.rowcount, column=0, columnspan=2, sticky="nsew")  # Use grid for the notebook
        # Make the window resizeable
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        self.notebook = notebook


        # general Info tab
        self.general = GeneralInfoTab(self)

        self.rowcount+=1
        add_tab_button = tk.Button(root, text="Add variable", command=self.add_tab)
        add_tab_button.grid(row=self.rowcount, column=0, padx=10, pady=10, sticky="nsew")


        add_tab_button = tk.Button(root, text="Delete variable", command=self.delete_tab)
        add_tab_button.grid(row=self.rowcount, column=1, padx=10, pady=10, sticky="nsew")
        self.rowcount += 1

        btn_submit = tk.Button(root, text="Generate and save", command=self.generate)
        btn_submit.grid(row=self.rowcount, column=0, padx=10, pady=10)
        btn_submit = tk.Button(root, text="Open document",  command=self.open_doc)
        btn_submit.grid(row=self.rowcount, column=1, padx=10, pady=10)
        self.rowcount += 1


        image = Image.open(os.path.join("docs", "pics", "logos.png"))  # Replace with your image file
        width, height = image.size
        resize = 0.75
        image = image.resize((int(resize*width), int(resize*height)))  # Resize the image if needed
        img = ImageTk.PhotoImage(image)

        # Create a Label to display the image and use grid()
        image_label = tk.Label(self.root, image=img)
        image_label.grid(row=self.rowcount, column=0, padx=10, pady=10, sticky="nsew")

        self.rowcount += 1

        ack = tk.Label(self.root, text="This work has been funded by the MINKE project funded by the European Commission within the Horizon 2020 Programme (2014-2020) Grant Agreement No. 101008724.", fg=GREY)
        ack.grid(row=self.rowcount, column=0, padx=10, pady=10, sticky=W)
        self.root.mainloop()


    def add_tab(self):
        v = VariableTab(self.notebook)
        self.variables.append(v)
        return v

    def delete_tab(self):
        # Get the currently selected tab
        current_tab = self.root.nametowidget(self.notebook.select())
        # Keep all tabs that are different to this one

        if id(current_tab) != id(self.general.tab):
            self.notebook.forget(current_tab)
            # Delete from list by keeping other tabs
            self.variables = [v for v in self.variables if id(v.tab) != id(current_tab)]
        else:
            print("Can't delete general info tab!")

    def open_doc(self):
        importlib.reload(SensorML)
        file_path = filedialog.askopenfilename(defaultextension=".json",
                                                 filetypes=[("SensorML Files", "*.json"), ("All Files", "*.*")])
        rich.print(f"Opening file {file_path}...")
        doc = SensorML.to_gui(file_path)
        self.general.put(doc)
        for vardoc in doc["variables"]:
            var = self.add_tab()
            var.put(vardoc)

    def generate(self):
        """
        :return: Generate SensorML document
        """

        try:
            general = {
                "sensorName": self.general.sensor_name.get(),
                "longName": self.general.long_name.get(),
                "serialNumber": self.general.serial_number.get(),
                "model": self.general.sensor_model.get(),
                "sensorType": self.general.sensor_type.get(),
                "manufacturer": self.general.manufacturer.get()
            }
            variables = [v.get() for v in self.variables]

        except ValueError as e:
            print(traceback.format_exc())
            messagebox.showerror("Input Error", "Document validation failed!")
            return

        try:
            importlib.reload(SensorML)
            doc = SensorML.from_gui(general, variables)
            with open("temp.json", "w") as f:
                f.write(json.dumps(doc, indent=4))
            SensorML.validate_doc(doc)

        except AssertionError as e:
            messagebox.showerror("Input Error", e.__str__())
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("SensorML Files", "*.json"), ("All Files", "*.*")])
        with open(file_path, "w") as f:
            f.write(json.dumps(doc, indent=4))

        messagebox.showinfo('Info', 'Save completed!')


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-s", "--sensorml", type=str, help="SensorML document", default="")
    args = argparser.parse_args()

    if args.sensorml:
        with open(args.sensorml) as f:
            sensorml = json.load(f)

    sml = SensorEditor()

