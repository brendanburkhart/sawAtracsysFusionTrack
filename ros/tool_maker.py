#!/usr/bin/env python

# Author: Anton Deguet, Brendan Burkhart
# Date: 2021-10-29

# (C) Copyright 2021-2022 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

import sys
import argparse
import rospy
import numpy as np
import json
import registration

import geometry_msgs.msg


class SAWToolDefinition:
    def __init__(self, tool_id, markers, pivot=None):
        self.id = tool_id
        self.markers = markers
        self.pivot = pivot

    @staticmethod
    def from_json(json_dict):
        def point_to_array(point):
            return np.array([point["x"], point["y"], point["z"]])

        assert json_dict.get("count", 0) == len(json_dict["fiducials"])
        pivot = point_to_array(json_dict["pivot"]) if "pivot" in json_dict else None
        markers = np.array([point_to_array(f) for f in json_dict["fiducials"]])

        tool_id = json_dict.get("id", None)

        return SAWToolDefinition(tool_id, markers, pivot)

    def to_json(self):
        def array_to_point(array):
            return {"x": array[0], "y": array[1], "z": array[2]}

        json_dict = {}

        if self.id is not None:
            json_dict["id"] = int(self.id)

        json_dict["count"] = len(self.markers)
        json_dict["fiducials"] = [array_to_point(m) for m in self.markers]

        if self.pivot is not None:
            json_dict["pivot"] = array_to_point(self.pivot)

        return json_dict

def error_threshold(geometry, factor=0.2):
    _, distance = registration.nearest_pair(geometry)
    return distance*factor

class SampleStream:
    def __init__(self, ros_topic, reference_geometry=None, marker_count=None):
        self.ros_topic = ros_topic
        self.reference_geometry = reference_geometry
        self.marker_count = marker_count

        self.collecting = False
        self.samples = []
        self.previous_sample = None

        if reference_geometry is not None:
            self.add_reference_geometry(reference_geometry)

        if reference_geometry is None and marker_count is None:
            raise ValueError(
                "either reference_geometry or marker_count must be specified"
            )

        self.data_subscriber = rospy.Subscriber(
            ros_topic, geometry_msgs.msg.PoseArray, self._sample_callback
        )

    def unregister(self):
        self.data_subscriber.unregister()

    def add_reference_geometry(self, geometry):
        self.reference_geometry = geometry
        self.marker_count = len(self.reference_geometry)
        self.maximum_marker_error = error_threshold(self.reference_geometry)

    def _display_sample_count(self):
        print("Number of samples collected: {}".format(len(self.samples)), end="\r")

    def _sample_callback(self, message):
        if not self.collecting or len(message.poses) != self.marker_count:
            self.previous_sample = None
            return

        point = lambda pose: [pose.position.x, pose.position.y, pose.position.z]
        sample = np.array([point(marker) for marker in message.poses])

        if self.previous_sample is not None and registration.nearest_neighbor(self.previous_sample, sample) is not None:
            sample = sample[registration.nearest_neighbor(self.previous_sample, sample)]
        self.previous_sample = sample

        if self.reference_geometry is not None:
            _, _, error = registration.iterative_closest_point(self.reference_geometry, sample, initial_order=True)
            if error > self.maximum_marker_error:
                self.previous_sample = None
                return
        
        self.samples.append(sample)
        self._display_sample_count()

    def collect_n(self, sample_count):
        self.samples = []
        self._display_sample_count()
        self.collecting = True

        while not rospy.is_shutdown() and len(self.samples) < sample_count:
            rospy.sleep(0.05)

        self.collecting = False
        return np.array(self.samples[:sample_count])

    def collect(self):
        print("Collection started, press 'enter' to stop")
        self.samples = []
        self._display_sample_count()
        self.collecting = True

        input()
        self.collecting = False
        print()

        return np.array(self.samples)

# apply PCA to align coordinate system to marker geometry
# Points data should have mean zero (i.e. be centered at origin).
def principal_component_analysis(points):
    # SVD for PCA
    _, sigma, Vt = np.linalg.svd(points, full_matrices=False)

    # Orientation should be (close to) +/-1
    basis_orientation = np.linalg.det(Vt)
    # Select positive orientation of basis
    if basis_orientation < 0.0:
        Vt[2, :] = -Vt[2, :]

    return np.matmul(points, Vt.T)


# given samples, determine tool geometry
def determine_geometry(samples):
    def align_markers(markers):
        order, (R, t), _ = registration.iterative_closest_point(samples[0], markers, initial_order=True)
        ordered_markers = markers[order]
        aligned_markers = np.matmul(ordered_markers, R.T) + t
        return aligned_markers

    # rotate/translate samples to align all markers (minimize RMS)
    aligned_samples = [align_markers(s) for s in samples]
    # average position of each marker
    averaged_markers = np.mean(aligned_samples, axis=0)
    # center of individual average marker positions
    centroid = np.mean(averaged_markers, axis=0)
    # center coordinate system at centroid
    markers = averaged_markers - centroid
    error = registration.rms_error(aligned_samples, averaged_markers)
    # align using PCA
    geometry = principal_component_analysis(markers)

    return geometry, error

# given samples where one pivot marker was kept relatively still,
# determine pivot marker and return its index within the reference geometry
def determine_pivot(records, geometry):
    _, (R, t), _ = registration.iterative_closest_point(geometry, records[0])
    pivot, error = registration.determine_pivot(records)
    pivot = np.matmul(R, pivot) + t

    return pivot, error

supported_units = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
}

# given marker points in meters, convert to desired units
def convert_units(marker_points, output_units):
    # Input marker pose data is always in meters
    input_units = "m"

    print("Converting units from {} to {}".format(input_units, output_units))
    return marker_points * supported_units[input_units] / supported_units[output_units]


def read_data(file_name):
    with open(file_name, "r") as f:
        json_dict = json.load(f)

    return SAWToolDefinition.from_json(json_dict)


def write_data(points, id, output_file_name, pivot=None):
    tool = SAWToolDefinition(id, points, pivot)

    with open(output_file_name, "w") as f:
        json_dict = tool.to_json()
        json.dump(json_dict, f, indent=4, sort_keys=True)
        f.write("\n")

    print("Generated tool geometry file {}".format(output_file_name))


def create_tool(args, minimum_samples=10):
    data_stream = SampleStream(args.topic, marker_count=args.num_markers)
    reference_geometry = None
    
    input(f"Press 'enter' to start collection using topic {args.topic}")
    while not rospy.is_shutdown() and reference_geometry is None:
        reference_snapshot = data_stream.collect_n(minimum_samples)
        if len(reference_snapshot) < minimum_samples:
            print("ERROR: failed to collect reference snapshot")
            return

        max_error = min([error_threshold(s) for s in reference_snapshot])
        measured_geometry, error = determine_geometry(reference_snapshot)
        if error < max_error:
            reference_geometry = measured_geometry

    data_stream.add_reference_geometry(reference_geometry)
    samples = data_stream.collect()
    if len(samples) < minimum_samples:
        print("Not enough samples collected ({} minimum)".format(minimum_samples))
        return

    geometry, error = determine_geometry(samples)
    print("RMS error of measured geometry: {:.3f}".format(1e3*error))
    geometry = convert_units(geometry, args.units)
    write_data(geometry, args.id, args.file)


def measure_pivot(args, minimum_samples=20):
    tool = read_data(args.file)
    geometry = 1e-3*tool.markers
    data_stream = SampleStream(args.topic, reference_geometry=geometry)

    input(f"Press 'enter' to start collection using topic {args.topic}")
    samples = data_stream.collect()
    if len(samples) < minimum_samples:
        print("Not enough samples collected ({} minimum)".format(minimum_samples))
        return

    samples = convert_units(samples, args.units)
    pivot, error = determine_pivot(samples, tool.markers)
    print("RMS error of measured pivot: {:.3f}".format(1e3*error))
    write_data(tool.markers, tool.id, args.file, pivot=pivot)

def main():
    # ros init node so we can use default ros arguments (e.g. __ns:= for namespace)
    rospy.init_node("tool_maker", anonymous=True)
    # strip ros arguments
    argv = rospy.myargv(argv=sys.argv)

    # parse arguments
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title="commands",
        help="see each command for additional help",
        dest="command",
        required=True,
    )

    # required arguments
    parser.add_argument(
        "-t",
        "--topic",
        type=str,
        required=True,
        help="topic to use to receive PoseArray without namespace. Use __ns:= to specify the namespace",
    )
    parser.add_argument(
        "-f", "--file", type=str, required=True, help="output file name"
    )

    pivot_parser = subparsers.add_parser(
        "pivot", aliases=["p"], help="determine tool pivot"
    )
    pivot_parser.set_defaults(command_func=measure_pivot)
    create_parser = subparsers.add_parser(
        "create", aliases=["c"], help="create a new tool definition"
    )
    create_parser.set_defaults(command_func=create_tool)
    create_parser.add_argument(
        "-n",
        "--num_markers",
        type=int,
        choices=range(3, 10),
        required=True,
        help="number of markers on the tool. Used to filter messages with incorrect number of markers",
    )
    create_parser.add_argument(
        "-i", "--id", type=int, required=False, help="specify optional tool id"
    )
    parser.add_argument(
        "-u",
        "--units",
        type=str,
        choices=supported_units.keys(),
        default="mm",
        required=False,
        help="units to use for output data in tool config",
    )

    args = parser.parse_args()
    args.command_func(args)


if __name__ == "__main__":
    main()
