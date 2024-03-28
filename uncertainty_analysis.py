#!/usr/bin/env python3
"""
This script shows an example on how a SensorML document can be used to apply uncertainty analysis

author: Enoc Martínez
institution: Universitat Politècnica de Catalunya (UPC)
email: enoc.martinez@upc.edu
license: MIT
created: 26/3/24
"""

import os
from argparse import ArgumentParser
import rich as r
import json
import jsonschema
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from SensorML import SensorML


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("sensorml", type=str, help="SensorML document", default="")
    argparser.add_argument("csv", type=str, help="CSV file", default="")
    args = argparser.parse_args()
    matplotlib.use('TkAgg')  # Set the backend to TkAgg
    with open(args.sensorml) as f:
        doc = json.load(f)

    # Load JSON Schema
    with open("ogcapi-connected-systems/sensorml/schemas/json/PhysicalSystem.json") as f:
        schema = json.load(f)

    base_path = os.path.join(os.getcwd(), "ogcapi-connected-systems", "sensorml", "schemas", "json")

    resolver = jsonschema.validators.RefResolver(base_uri='file://{}/'.format(base_path), referrer=schema)
    r.print(f"Validating '{doc['type']}' with id='{doc['id']}'...", end="")
    try:
        jsonschema.validate(instance=doc, schema=schema, resolver=resolver)
    except jsonschema.exceptions.ValidationError as e:
        r.print(e)
        r.print(f"[red]document {args.sensorml} not valid!")
        raise e
    r.print("[green]done")

    sml = SensorML(doc)
    df = pd.read_csv(args.csv)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    for variable in df.columns:
        if variable.lower().endswith("_qc"):
            continue

        calibration = sml.get_calibration(variable)
        if not calibration:
            r.print(f"[yellow]No calibration found for {variable}!")
            continue
        # if variable == "CNDC":
        #     continue

        raw_values = df[variable].values
        times = df.index.values
        corrected_values = calibration.correct(raw_values)

        r.print(f"Plotting variable {variable}")
        fig, axd = plt.subplot_mosaic([['left', 'right'], ['center', 'center'], ['bottom', 'bottom']])

        ax1 = axd["left"]
        ax2 = axd["right"]
        ax3 = axd["center"]
        ax4 = axd["bottom"]

        ax3.get_shared_x_axes().join(ax4, ax3)

        ax1.plot(times, raw_values, label="raw data")
        ax1.set_title(variable + " raw data")
        ax1.legend()
        ax1.grid()

        calibration.plot_calibration(ax2)
        ax2.set_title(f"{variable} calibration curve")
        ax2.grid()

        ax3.set_title(variable + " processed data")
        ax3.plot(times, corrected_values, label="corrected data")
        ax3.plot(times, raw_values, alpha=0.5, label="raw data")
        ax3.grid()

        if calibration.defined_uncertainty():
            uncertainty = calibration.calc_uncertainty(raw_values, times, stability=True)
            ax3.fill_between(times, raw_values + uncertainty, raw_values - uncertainty, alpha=0.5, label="uncertainty")

            ax4.fill_between(times, uncertainty, -uncertainty, alpha=0.3, label="uncertainty")
            ax4.set_title("Corrections + Uncertainty")
        else:
            ax4.set_title("Corrections")

        ax4.plot(times, corrected_values - raw_values, label="correction")
        ax4.grid()
        ax4.legend()


        ax3.legend()
    plt.show()
