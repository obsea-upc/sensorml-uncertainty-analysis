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
import rich
from emso_metadata_harmonizer.metadata.dataset import export_to_netcdf

from SensorML import SensorML
import emso_metadata_harmonizer as emh

def get_decimal_precision(values, limit=1000):
    decimals = 0

    if len(values) < limit:  # If we have more than limit values, just take some of them
        values = values[:limit]

    for v in values:
        try:
            s = str(v)
            decimals = max(len(s.split(".")[1]), decimals)
        except IndexError:
            pass
    return decimals


def correct_data_point(value, references,  measurements, corrections, uncertainties):
    for i in range(len(measurements) - 1):
        if measurements[i] < value < measurements[i + 1]:
            # measurements are X, corrections are Y
            # Interpolation implemented as  y = (x-x1)*(y2-y1)/(x2-x1) + y1
            x = value
            x1 = measurements[i - 1]
            x2 = measurements[i]
            y1 = corrections[i - 1]
            y2 = corrections[i]
            corr = (x - x1) * (y2 - y1) / (x2 - x1) + y1
            # Get the maximum uncertainty
            uncert = max(uncertainties[i], uncertainties[i + 1])
            # rich.print(f"   Raw value: {value}")
            # rich.print(f"     Segment: between {measurements[i]} and {measurements[i + 1]}")
            # rich.print(f"        corr: {corr}")
            # rich.print(f"  corr value: {value + corr}")
            # rich.print(f"      uncert: {uncert}")
            return corr, uncert
    return np.nan, np.nan


def calibrate(values: np.array, references: np.array, measurements: np.array, corrections: np.array,
              uncertainties: np.array) -> (np.array, np.array):
    # measurements are X, corrections are Y
    # Interpolation implemented as  y = (x-x1)*(y2-y1)/(x2-x1) + y1


    if np.count_nonzero(uncertainties):
        # By default, use the precision of the uncertainty
        p = get_decimal_precision(uncertainties)
    else:
        # However, if the manufacturer did not provide uncertainty, we need a workaround
        # Get the max precision of the references and the measurements
        p = max(get_decimal_precision(references), get_decimal_precision(measurements))

    result = np.zeros(len(values))
    result_u95 = np.zeros(len(values))

    for i in range(len(values)):
        corr, uncert95 = correct_data_point(values[i], references, measurements, corrections, uncertainties)
        result[i] = round(values[i] + corr, p)
        result_u95[i] = round(uncert95, p)

    if np.count_nonzero(result_u95):
        # If ALL values are 0 it means that we do not have uncertainty information, so let's put NaN to all
        result_u95.fill(np.nan)
    return result, result_u95


def multi_sensor_netcdf(wf):
    # Check if the WaterFrame is multi-sensor or not

    serial_numbers = []
    for varcode, meta in wf.vocabulary.items():
        if "sensor_serial_number" not in meta.keys():
            continue
        print(f"var: {varcode}, SN: {meta['sensor_serial_number']}")
        serials = meta["sensor_serial_number"]
        for s in serials:
            if s.strip():
                serial_numbers.append(s.strip)
    return serial_numbers



def open_netcdf(filename):
    wf = emh.metadata.dataset.load_data(filename)
    print(wf.data)
    return wf

def write_netcdf(wf):
    return emh.metadata.dataset.export_to_netcdf(wf, "output.nc")


def run_analysis(sensorml: str, dataset: str, output:str, plot=False):
    if dataset.endswith(".csv"):
        df = pd.read_csv(dataset)
        wf = None
    elif dataset.endswith(".nc"):
        wf = open_netcdf(dataset)
        df = wf.data
    else:
        raise ValueError(f"Unimplemented format {dataset.split('.')}")

    serial_numbers = multi_sensor_netcdf(wf)
    if multi_sensor_netcdf(wf):
        print("multisensor")
    else:
        pass

    with open(sensorml) as f:
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
        r.print(f"[red]document {sensorml} not valid!")
        raise e
    r.print("[green]done")

    sml = SensorML(doc)

    if "time" in df.columns:
        df = df.rename(columns={"time": "TIME"})

    df["TIME"] = pd.to_datetime(df["TIME"])
    df = df.set_index("TIME")

    for variable in df.columns:
        if variable.lower().endswith("_qc") or variable.lower() in ["depth", "latitude", "longitude", "sensor_id"]:
            continue

        calibration = sml.get_calibration(variable)
        if not calibration:
            r.print(f"[yellow]No calibration found for {variable}!")
            if wf: # adding raw data
                meta = wf.vocabulary[variable]  # original metadata
                # Add calibration notes
                meta["long_name"] += " (raw sensor values)"
                if "comment" not in meta.keys():
                    meta["comment"] = "Raw sensor values. "
                else:
                    meta["comment"] = "Raw sensor values. " + meta["comment"]
            continue

        # Getting the precision based in calibration
        r.print(f"[cyan]Calculating uncertainties for {variable}!")


        raw_values = df[variable].values
        times = df.index.values
        # corrected_values = calibration.correct(raw_values)
        # corrected_values, uncertainties = calibration.get_corrections_u(raw_values)
        corrected_values, u95 = calibrate(raw_values, calibration.references, calibration.measurements,
                                               calibration.corrections, calibration.uncertainties)

        df[variable + "_RAW"] = raw_values
        df[variable] = corrected_values
        df[variable + "_UNCERTAINTY"] = u95

        if wf:
            if "projects" in wf.metadata:
                wf.metadata["projects"] += " MINKE"
            else:
                wf.metadata["projects"] = "MINKE"

            wf.metadata["funding"] = "This work has been funded by the MINKE project funded by the European Commission within the Horizon 2020 Programme (2014-2020), grant Agreement No. 101008724."
            wf.data = df.reset_index()
            base = wf.vocabulary[variable].copy()  # original metadata

            # Add calibration notes
            meta = base.copy()
            meta["long_name"] += " (corrected values)"
            meta["comment"] = "values corrected with latest available calibration. " + meta["comment"]
            ancillary = meta["ancillary_variables"].split(" ")
            ancillary += [f"{variable}_{suffix}" for suffix in ["RAW", "UNCERTAINTY"]]

            meta["ancillary_variables"] = ", ".join(ancillary)
            wf.vocabulary[variable] = meta

            # Raw data
            meta = base.copy()
            meta["comment"] = "raw data. " + meta["comment"]
            meta["long_name"] += " (raw data)"
            wf.vocabulary[variable + "_RAW"] = meta

            meta = {}
            meta["long_name"] = base["long_name"] + " expanded uncertainty with 95% coverage (k=2)"
            meta["sdn_uom_urn"] = base["sdn_uom_urn"]
            meta["sdn_uom_name"] = base["sdn_uom_name"]
            meta["units"] = base["units"]
            meta["comment"] = "Expanded uncertainty with 95% coverage (k=2)"
            wf.vocabulary[variable + "_UNCERTAINTY"] = meta
            if plot:
                plot_variable(variable, times, raw_values, corrected_values, calibration, show=False)


    if wf:
        export_to_netcdf(wf, output, multisensor_metadata=True)
    else:
        df.to_csv(output)



        

def plot_variable(variable, times, raw_values, corrected_values, calibration, show=False):
    matplotlib.use('TkAgg')  # Set the backend to TkAgg
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

    if show:
        plt.show()


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("sensorml", type=str, help="SensorML document", default="")
    argparser.add_argument("dataset", type=str, help="input dataset in CSV or NetCDF format")
    argparser.add_argument("output", type=str, help="output in CSV or NetCDF format")
    argparser.add_argument("--plot", action="store_true", help="Plot the variables")
    args = argparser.parse_args()
    run_analysis(args.sensorml, args.dataset, args.output, plot=args.plot)
