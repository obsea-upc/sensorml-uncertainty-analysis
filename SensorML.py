#!/usr/bin/env python3
"""
Generic functions and classes to parse and process SensorML JSON documents

author: Enoc Martínez
institution: Universitat Politècnica de Catalunya (UPC)
email: enoc.martinez@upc.edu
license: MIT
created: 26/3/24
"""
import os

import jsonschema
import rich as r
import rich
import json
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
import traceback
import datetime

calibration_definition = "http://vocab.nerc.ac.uk/collection/W03/current/W030003"

def is_numeric(e: list):
    """
    Checks recursively if the elements within a list are numeric or objects
    :param e:
    :return:
    """
    for child in e:
        if type(child) == list:
            return is_numeric(child)
        elif type(child) == dict:
            return False
        elif type(child) in [str, float, int]:
            return True
        raise ValueError("This should never happen!")



class AbstractComponent:
    def __init__(self, doc, constructors: dict):
        """
        Processes generic info from an abstract component
        :param doc: dict object
        """
        assert type(doc) is dict, f"Expected dict, but got {type(doc)}"
        self.doc = doc
        for key, value in doc.items():
            if type(value) != dict and type(value) != list:
                self.__dict__[key] = value

            elif type(value) == dict:
                # If there's a specific constructor use it, otherwise use AbstractComponent
                if key in constructors.keys():
                    constructor = constructors[key]
                else:
                    constructor = AbstractComponent

                self.__dict__[key] = constructor(value, constructors)

            elif type(value) == list:

                if is_numeric(value): # if the list doesn't contain object go, just assign it
                    self.__dict__[key] = value
                    continue

                self.__dict__[key] = {}
                for v in value:
                    # If there's a specific constructor use it, otherwise use AbstractComponent
                    if key in constructors.keys():
                        constructor = constructors[key]
                    else:
                        constructor = AbstractComponent
                    child = constructor(v, constructors)

                    self.__dict__[key][child.get_id()] = child

    def get_id(self):
        """
        Returns the first identifiable element from the component: id or name or definition or label.
        :return: identifier
        """
        identifiable_types = ["id", "name", "definition", "label", "individualName", "organisationName"]
        for key in identifiable_types:
            if key in self.__dict__.keys():
                return self.__dict__[key]

        # If the object is a DataArray use "values" as identifier
        if "type" in self.doc.keys():
            if self.doc["type"] == "DataArray":
                return "values"

        r.print(f"[red]{self.doc}")
        raise ValueError(f"Object doesnt have any identifible types: {identifiable_types}")

    def __str__(self):
        return json.dumps(self.doc, indent=2)

    def __repr__(self):
        return json.dumps(self.doc, indent=2)


class Output(AbstractComponent):
    def __init__(self, doc, mapper):
        AbstractComponent.__init__(self, doc, mapper)

class Connection(AbstractComponent):
    def __init__(self, doc, mapper):
        self.source = doc.pop("source")
        self.destination = doc.pop("destination")
        AbstractComponent.__init__(self, doc, mapper)

    def get_id(self):
        return id(self)


class SensorCalibration:
    def __init__(self, calibration, component):
        """
        Parses a SensorML calibration sheet and stores into a python object.

        :param calibration: SensorML calibration history event (dict)
        """
        self.doc = calibration
        # self.resolution = value_from_array(calibration["capabilities"], label="resolution")
        # self.expanded_uncertainty = np.nan
        # self.confidence_level = np.nan
        # self.measuring_range = np.nan
        self.time = pd.to_datetime(calibration.time)

        # Assume that the calibration results are inside a DataArray in properties
        array = {}
        for p in calibration.properties.values():
            # assume this is the calibration array
            if p.type == "DataArray":
                # creating array as a dictionary with field names
                for field in p.elementType.fields.values():
                    array[field.name] = []
                # now parse the values fore each line
                n = 0
                for line in p.values:
                    n += 1
                    if len(line) != len(array):
                        r.print(f"[red]Error in line {n}, expected {len(array)} fields, but got {len(line)}")
                        exit()
                    for i in range(0, len(array)):
                        column = list(array.keys())[i]
                        array[column].append(line[i])

        self.values = {}
        for key, values in array.items():
            self.values[key] = np.array(values)

        # Now make sure that we have all required data within the array. Required values:
        for ref in ["reference", "measurement", "correction"]:
            if ref not in self.values.keys():
                r.print(f"[red]Missing required key in calibration array '{ref}'")
                raise ValueError(f"Missing required key in calibration array '{ref}'")
        # optional values
        __optional_values = ["expanded uncertainty"]
        for ref in ["reference", "measurement", "correction"]:
            if ref not in self.values.keys():
                r.print(f"[yellow]Missing optional key in calibration array '{ref}'")

        r.print(self.values)
        self.references = self.values["reference"]
        self.measurements = self.values["measurement"]
        self.corrections = self.values["correction"]
        self.uncertainties = self.values["expandedUncertainty"]

        if self.defined_uncertainty():
            self.uncertainty = self.values["expanded uncertainty"]

        # getting stability per year
        try:
            for cap in component.capabilities.values():
                if cap.id == "specifications":
                    specifications = cap
                    break

            for field in specifications.capabilities.values():
                if field.label == "stability":
                    stability = field.value
                    stability_units = field.uom.label

            if "day" in stability_units.lower():
                self.yearly_stability = stability*365
            elif "week" in stability_units.lower():
                self.yearly_stability = stability*(365/7)
            elif "month" in stability_units.lower():
                self.yearly_stability = stability*12
            elif "year" in stability_units.lower():
                self.yearly_stability = stability
            else:
                r.print(f"[red]Unknown '{stability_units}'! expected stability related to day, week, month or year!")
        except AttributeError:
            r.print(f"[red]Stability not found for {component.id}!")
            r.print(traceback.format_exc())
            self.yearly_stability = 0
        except UnboundLocalError:
            r.print(f"[red]Stability not defined for {component.id}!")
            r.print(traceback.format_exc())
            self.yearly_stability = 0

    def defined_uncertainty(self):
        """
        Checks if the current calibration defined the uncertanity
        :return:  True / False
        """
        if "expanded uncertainty" in self.values.keys():
            return True
        return False


    def plot_calibration(self, ax: Axes, step:float = 0.001):

        if "expanded uncertainty" in self.values.keys():
            u1 = self.values["measurement"] + self.values["expanded uncertainty"]
            u2 = self.values["measurement"] - self.values["expanded uncertainty"]
            ax.fill_between(self.values["reference"], u1, u2, alpha=0.2, color="blue", label="expanded uncertainty")

        ax.scatter(self.values["reference"], self.values["measurement"], label="measurements")
        ax.set_xlabel("reference")
        ax.set_ylabel("measurement")
        xmin = self.values["reference"][0]
        xmax = self.values["reference"][-1]
        x = np.arange(xmin, xmax, step)
        yinterp = np.interp(x, self.values["reference"], self.values["measurement"])
        ax.plot(x, yinterp, linestyle="--", color="orange", label="interpolation")
        ax.legend()
        return ax

    def get_correction(self, value, measurements, corrections):
        """
        Given a measurement value, calculate the correction to be applied according to the calibration using linear
        interpolation
        :param value:
        :param measurements: calibration measurements
        :param corrections: calibration corrections
        :return:
        """

        if value > self.measurements[-1] or value < self.measurements[0]:
            r.print(f"[yellow]Value {value} beyond calibration!")
            return value

        for i in range(len(self.measurements) - 1):
            if  self.measurements[i] < value < self.measurements[i + 1]:
                # measurements are X, corrections are Y
                # Interpolation implemented as  y = (x-x1)*(y2-y1)/(x2-x1) + y1
                x = value
                x1 = measurements[i-1]
                x2 = measurements[i]
                y1 = corrections[i-1]
                y2 = corrections[i]
                corr = (x-x1)*(y2-y1)/(x2-x1) + y1
                return corr
        raise ValueError(f"Data point beyond calibration! val={value} max measurement {self.measurements[-1]}")

    def get_corrections_u(self, value) -> (float, float):
        """
        Given a measurement value, calculate the correction to be applied according to the calibration using linear
        interpolation
        :param value:
        :param measurements: calibration measurements
        :param corrections: calibration corrections
        :param uncertainties:
        :return:
        """

        if value > self.measurements[-1] or value < self.measurements[0]:
            r.print(f"[yellow]Value {value} beyond calibration!")
            return value

        for i in range(len(self.measurements) - 1):
            if  self.measurements[i] < value < self.measurements[i + 1]:
                # measurements are X, corrections are Y
                # Interpolation implemented as  y = (x-x1)*(y2-y1)/(x2-x1) + y1
                x = value
                x1 = self.measurements[i-1]
                x2 = self.measurements[i]
                y1 = self.corrections[i-1]
                y2 = self.corrections[i]
                corr = (x-x1)*(y2-y1)/(x2-x1) + y1
                uncert = max(self.uncertainties[i], self.uncertainties[i+1])
                return corr, uncert
        raise ValueError(f"Data point beyond calibration! val={value} max measurement {self.measurements[-1]}")


    def calc_uncertainty(self, values: np.array, times: np.array, stability=True) -> np.array:
        """
        Returns the uncertainty for every point in the array using linear interpolation
        :param values: input array
        :return: np.array with uncertanity values
        """
        # create an empty array
        uncertainty = np.zeros(len(values))

        for i in range(len(values)): # for every point in the array
            value = values[i]
            u = self.get_correction(value, self.measurements, self.uncertainty)

            if stability and self.yearly_stability:
                # Add the drift to the uncertainty
                timedelta = times[i] - self.time
                seconds = timedelta.total_seconds()
                years = seconds / (3600*24*365)
                if seconds < 0:
                    pass
                else:
                    k = 2  # assuming initial distribution in gaussian k=2

                    # Calculate the uncertainty contribution of the stability, assuming rectangular distribution
                    u_stability = years*self.yearly_stability / np.sqrt(3)

                    # Combine the u_stability with the initial uncertainty
                    u = np.sqrt( np.power(u/k, 2) + np.power(u_stability, 2) )

                    # Assuming we still have a gaussian distribution
                    u = k*u

            uncertainty[i] = u

        return uncertainty


    def correct_u(self, values: np.array) -> np.array:
        """
        Adjusts the input array according to the calibration sheet using linear interpolation.
        :param values: input array
        :return: np.array with adjust values
        """
        # create an empty array
        out = np.zeros(len(values))
        uncert95 = np.zeros(len(values))

        for i in range(len(values)): # for every point in the array
            value = values[i]
            corr = self.get_corrections_u(value, self.measurements, self.corrections)
            out[i] = value - corr

        uncert99 = 3/2 * uncert95
        return out, uncert95, uncert99

class Component(AbstractComponent):
    def __init__(self, doc, mapper):
        AbstractComponent.__init__(self, doc, mapper)


    def get_calibration(self):
        """
        Looks for a calibration sheet within the component, if not found return empty dict
        :return: dict with calibration
        """
        if "history" not in self.__dict__.keys():
            return {}

        for history in self.history.values():  # loop through all calibration history
            try:
                if history.__dict__["definition"] == calibration_definition:
                    return history
            except KeyError:
                continue
        return {}

class_mapper_handler = {
    "outputs": Output,
    "output": Output,
    "components": Component,
    "connections": Connection
}

class SensorML(AbstractComponent):
    def __init__(self, doc):

        AbstractComponent.__init__(self, doc, class_mapper_handler)
        r.print(f"[orange1]Processing document for Sensor {self.label}")

        # Now let's match outputs with components

        self.output_component = {}
        self.calibrations = {}

        for name, value in self.outputs.items():
            r.print(f"Looking for component for output '{name}'...", end="")
            try:
                comp = self.components[name]
                r.print("[green]found!")
            except KeyError:
                r.print(f"[red]No components found")
                continue

            self.output_component[name] = comp

            calibration_config = comp.get_calibration()
            if not calibration_config:
                continue
            self.calibrations[name] = SensorCalibration(calibration_config, comp)

    def get_calibration(self, variable_name):
        if variable_name not in self.calibrations.keys():
            return None
        else:
            r.print(f"[green]found calibration for variable {variable_name}")
            return self.calibrations[variable_name]


def from_gui(generic: dict, variables: list):
    """
    Generate a SensorML document from the GUI output
    """
    required_vars = ["sensorName", "serialNumber"]

    for v in required_vars:
        assert generic[v], f"field {v} is required!"

    assert len(variables), f"At least one variable is required!"

    sensor_id = generic["sensorName"] + "-SN" + generic["serialNumber"]
    doc = {
        "$schema": "connected-systems/sensorml/schemas/json/PhysicalSystem.json",
        "type": "PhysicalSystem",
        "id": sensor_id,
        "uniqueId": sensor_id,
        "label": generic["sensorName"],
        "identifiers": [
            {
                "definition": "http://vocab.nerc.ac.uk/collection/W07/current/IDEN0006",
                "label": "shortName",
                "value": generic["sensorName"]
            },
            {
                "definition": "http://vocab.nerc.ac.uk/collection/W07/current/IDEN0002",
                "label": "longName",
                "value":  generic["longName"],
            },
            {
                "definition": "http://vocab.nerc.ac.uk/collection/W07/current/IDEN0005",
                "label": "serialNumber",
                "value": generic["serialNumber"]
            },
            {
                "definition": "http://vocab.nerc.ac.uk/collection/W07/current/IDEN0003",
                "label": "model",
                "value": generic["model"]["uri"]
            },
            {
                "definition": "http://vocab.nerc.ac.uk/collection/W07/current/IDEN0012",
                "label": "manufacturer",
                "value": generic["manufacturer"]["uri"]
            }
        ],
        "classifiers": [
            {
                "definition": "http://vocab.nerc.ac.uk/collection/W06/current/CLSS0002",
                "label": "instrumentType",
                "value": generic["sensorType"]["uri"]
            }
        ],
        #"contacts": [],
        "outputs": [],
        "components": []
    }

    for v in variables:
        rich.print(v)

        output =     {
          "type": "ObservableProperty",
          "id": v["varcode"],
          "label": v["varname"]["label"],
          "name": v["standardName"]["label"],
          "definition": v["standardName"]["uri"],
          "uom": {
            "code": v["units"]["label"],
            "href": v["units"]["uri"]
          }
        }
        doc["outputs"].append(output)

        uom = {"code": v["units"]["label"]}

        output_comp =     {
          "type": "ObservableProperty",
          "id": v["varcode"],
          "label": v["varname"]["label"],
          "name": v["standardName"]["label"],
          "definition": v["standardName"]["uri"],
          "uom": {
            "code": v["units"]["label"],
            "href": v["units"]["uri"]
          }
        }


        component =     {
          "id": v["varcode"],
          "type": "PhysicalComponent",
          "name": generic["sensorName"] + "-" + v["varcode"],
          "label": generic["sensorName"] + "-" + v["varcode"],
          "uniqueId": sensor_id + ":" +v["varcode"],
          #"capabilities": [],
          "history": [
            {
              "label": "calibration",
              "definition": "http://vocab.nerc.ac.uk/collection/W03/current/W030003",
              "time": v["calibration"]["date"],
              "description": "Calibration.",
              #"classifiers": [],
              "contacts": [
                {
                  "role": "http://vocab.nerc.ac.uk/collection/W08/current/CONT0002",
                  "organisationName": v["calibration"]["lab"],
                }
              ],
              "properties": [
              {
                  "type": "Quantity",
                  "label": "coverage factor",
                  "definition": "...",
                  "uom": {
                      "code": "dimensionless"
                  },
                  "value": v["calibration"]["coverageFactor"]
              },
                {
                  "type": "DataArray",
                  "elementType": {
                    "type": "DataRecord",
                    "name": "calibrationResults",
                    "fields": [
                      {
                        "type": "Quantity",
                        "name": "reference",
                        "label": "reference",
                        "definition": "...",
                        "uom": uom
                      },
                      {
                        "type": "Quantity",
                        "name": "measurement",
                        "label": "measurement",
                        "definition": "...",
                        "uom":uom
                      },
                      {
                        "type": "Quantity",
                        "name": "correction",
                        "label": "Correction",
                        "definition": "...",
                        "uom": uom
                      },
                      {
                        "type": "Quantity",
                        "name": "expandedUncertainty",
                        "label": "expanded uncertainty",
                        "definition": "...",
                        "uom": uom
                      }
                    ]
                  },
                  "encoding": { "type": "JSONEncoding" },
                  "values": v["calibration"]["array"]
                }
              ]
            }
          ],
          "outputs": [output_comp]
        }
        doc["components"].append(component)

    return doc

def process_by_mapping(sml_items: list, doc: dict, mappings: dict):
    """Maps complex SensorML items to the simpler GUI-JSON format"""
    for item in sml_items:
        for key, value in mappings.items():
            if item["label"] == key:
                if isinstance(doc[value], str):
                    doc[value] = item["value"]
                elif isinstance(doc[value], dict):
                    doc[value]["uri"] = item["value"]

    return doc

def to_gui(filename)->dict:
    """Opens a json doc and tries to convert from SensorML to the GUI simplified json"""
    with open(filename) as f:
        sml = json.load(f)

    doc = {
        "sensorName": "",
        "longName": "",
        "serialNumber": "",
        "model": {},
        "sensorType": {},
        "manufacturer": {},
        "variables": []
    }
    identifiers_mapping = { # SensorML key: simple doc key
        "shortName": "sensorName",
        "longName": "longName",
        "serialNumber": "serialNumber",
        "model": "model",
        "manufacturer": "manufacturer",
    }
    doc = process_by_mapping(sml["identifiers"], doc, identifiers_mapping)
    classifier_mapping = {
        "instrumentType": "sensorType"
    }
    doc = process_by_mapping(sml["classifiers"], doc, classifier_mapping)
    for component in sml["components"]:
        if len(component["outputs"]) != 1:
            raise ValueError("Each component should provide only one output")
        output = component["outputs"][0]

        variable = {
            "varcode": output["id"],
            "varname": {"label": output["label"]},  # P02
            "standardName": {"uri": output["definition"]}, # P07
            "units": {"uri": output["uom"]["href"]}, # P06
            "calibration": {}
        }
        doc["variables"].append(variable)

        calibration = {}
        for history in component["history"]:
            if history["label"] == "calibration":

                # Look for calibration facility role in contacts
                for c in history["contacts"]:
                    if c["role"] == "http://vocab.nerc.ac.uk/collection/W08/current/CONT0002":
                        calibration["lab"] = c["organisationName"]
                calibration["date"] = history["time"]

                # Process history components
                for prop in history["properties"]:
                    if "label" in prop.keys() and prop["label"] == "coverage factor":
                        calibration["coverageFactor"] = prop["value"]

                    if "type" in prop.keys() and prop["type"] == "DataArray" and prop["elementType"]["name"]  == "calibrationResults":
                        calibration["array"] = prop["values"]

        variable["calibration"] = calibration

    return doc


def validate_doc(doc):
    # Load JSON Schema
    with open("ogcapi-connected-systems/sensorml/schemas/json/PhysicalSystem.json") as f:
        schema = json.load(f)

    base_path = os.path.join(os.getcwd(), "ogcapi-connected-systems", "sensorml", "schemas", "json")

    resolver = jsonschema.validators.RefResolver(base_uri='file://{}/'.format(base_path), referrer=schema)
    r.print(f"Validating '{doc['type']}' with id='{doc['id']}'...", end="")
    try:
        jsonschema.validate(instance=doc, schema=schema, resolver=resolver)
    except jsonschema.exceptions.ValidationError as e:
        rich.print("[red]Not valid!")
        raise e




    r.print("[green]done")
