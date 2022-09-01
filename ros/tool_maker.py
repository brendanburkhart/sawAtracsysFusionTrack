#!/usr/bin/env python

# Author: Anton Deguet
# Date: 2021-10-29

# (C) Copyright 2021 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

import sys
import argparse
import rospy
import math
import numpy as np
import json
import scipy.spatial
import scipy.optimize

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
        markers = [point_to_array(f) for f in json_dict["fiducials"]]

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

# collect marker poses from specified topic
def get_pose_data(ros_topic, expected_marker_count):
    records = []
    collecting = False

    def display_sample_count():
        print("Number of samples collected: %i" % len(records), end="\r")

    def pose_array_callback(msg):
        # skip if not recording marker pose messages
        if not collecting:
            return

        # make sure the number of poses matches the number of expected markers
        if len(msg.poses) != expected_marker_count:
            return

        record = np.array(
            [
                (marker.position.x, marker.position.y, marker.position.z)
                for marker in msg.poses
            ]
        )

        records.append(record)
        display_sample_count()

    pose_array_subscriber = rospy.Subscriber(
        ros_topic, geometry_msgs.msg.PoseArray, pose_array_callback
    )

    input("Press Enter to start collection using topic %s" % ros_topic)
    print("Collection started\nPress Enter to stop")
    display_sample_count()
    collecting = True

    input("")
    collecting = False
    pose_array_subscriber.unregister()

    return records


def order_record(reference, record):
    # each record has n poses but we don't know if they are sorted by markers
    # find correspondence to reference marker that minimizes pair-wise distance
    correspondence = scipy.spatial.distance.cdist(record, reference).argmin(axis=0)

    # skip records where naive-correspondence isn't one-to-one
    if len(np.unique(correspondence)) != len(reference):
        return None

    # put record markers into the same order as the corresponding reference markers
    ordered_record = record[correspondence]
    return ordered_record


def kabsch_alignment(reference, record):
    centroid_A = np.mean(reference, axis=0)
    centroid_B = np.mean(record, axis=0)

    # Align centroids to remove translation
    A = reference - centroid_A
    B = record - centroid_B
    covariance = np.matmul(np.transpose(A), B)
    u, s, vh = np.linalg.svd(covariance)
    d = math.copysign(1, np.linalg.det(np.matmul(u, vh)))
    C = np.diag([1, 1, d])

    R = np.matmul(u, np.matmul(C, vh))
    # T = centroid_A - np.matmul(R, centroid_B)

    aligned_record = np.matmul(B, np.transpose(R))  # + T

    return aligned_record


# Apply PCA to align markers, and if is_planar to project to plane.
# Points data should have mean zero (i.e. be centered at origin).
# planar_threshold is maximium relative variance along third axis that is considerd planar
def principal_component_analysis(points, is_planar, planar_threshold=1e-2):
    # SVD for PCA
    _, sigma, Vt = np.linalg.svd(points, full_matrices=False)

    # Orientation should be (close to) +/-1
    basis_orientation = np.linalg.det(Vt)
    # Select positive orientation of basis
    if basis_orientation < 0.0:
        Vt[2, :] = -Vt[2, :]

    # Three markers will always be planar, so we can ignore minor computation errors
    marker_count = np.size(is_planar)
    is_planar = is_planar or marker_count == 3

    # Project markers to best-fit plane
    if is_planar:
        print("Planar flag enabled, projecting markers onto plane...")
        # Remove 3rd (smallest) principal componenent to collapse points to plane
        Vt[2, :] = 0

    planarity = sigma[2] / sigma[1]
    if is_planar and planarity > planar_threshold:
        print("WARNING: planar flag is enabled, but markers don't appear to be planar!")
    elif not is_planar and planarity < planar_threshold:
        print(
            "Markers appear to be planar. If so, add '--planar' flag for better results"
        )

    return np.matmul(points, Vt.T)


def process_marker_records(records, is_planar):
    # make sure markers are in same order in each record
    ordered_records = [order_record(records[0], r) for r in records]
    ordered_records = [r for r in ordered_records if r is not None]
    # rotate/translate records to align all markers (minimize RMS)
    aligned_records = [kabsch_alignment(ordered_records[0], r) for r in ordered_records]
    # average position of each marker
    averaged_marker_poses = np.mean(aligned_records, axis=0)
    # center of individual average marker positions
    centroid = np.mean(averaged_marker_poses, axis=0)
    # center coordinate system at centroid
    points = averaged_marker_poses - centroid
    # align using PCA and project to plane if is_planar flag is set
    points = principal_component_analysis(points, is_planar)

    return points


def find_pivot(records, geometry, threshold=0.05):
    # make sure markers are in same order in each record
    ordered_records = [order_record(geometry, r) for r in records]
    ordered_records = [r for r in ordered_records if r is not None]
    # standard deviation of each marker's position
    stdev = np.std(records, axis=0)
    pivot_index = np.argmin(stdev)

    # Check only pivot marker is fixed
    for i, s in enumerate(stdev):
        relative_stdev = stdev[pivot_index] / s
        if i != pivot_index and relative_stdev > threshold:
            return None

    return pivot_index


supported_units = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
}


def convert_units(marker_points, output_units):
    # Input marker pose data is always in meters
    input_units = "m"

    print("Converting units from {} to {}".format(input_units, output_units))
    return marker_points * supported_units[input_units] / supported_units[output_units]


def read_data(file_name):
    with open(file_name, "r") as f:
        tool = SAWToolDefinition.from_json(f.read())

    return tool

def write_data(points, id, output_file_name, pivot=None):
    tool = SAWToolDefinition(id, points, pivot)

    with open(output_file_name, "w") as f:
        json.dump(tool.to_json(), f, indent=4, sort_keys=True)
        f.write("\n")

    print("Generated tool geometry file {}".format(output_file_name))

def main():
    # ros init node so we can use default ros arguments (e.g. __ns:= for namespace)
    rospy.init_node("tool_maker", anonymous=True)
    # strip ros arguments
    argv = rospy.myargv(argv=sys.argv)

    # parse arguments
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument(
        "-t",
        "--topic",
        type=str,
        required=True,
        help="topic to use to receive PoseArray without namespace. Use __ns:= to specify the namespace",
    )
    parser.add_argument(
        "-n",
        "--number-of-markers",
        type=int,
        choices=range(3, 10),
        required=True,
        help="number of markers on the tool. Used to filter messages with incorrect number of markers",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="output file name"
    )

    # optional arguments
    parser.add_argument(
        "-p",
        "--planar",
        action="store_true",
        help="indicates all markers lie in a plane",
    )
    parser.add_argument(
        "-v",
        "--pivot",
        action="store_true",
        help="compute pivot",
    )
    parser.add_argument(
        "-i", "--id", type=int, required=False, help="specify optional id"
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

    args = parser.parse_args(argv[1:])  # skip argv[0], script name

    # Arbitrary number to make sure we have enough records to average out noise etc.
    minimum_records_required = 10

    # create the callback that will collect data
    records = get_pose_data(args.topic, args.number_of_markers)
    if len(records) < minimum_records_required:
        print("Not enough records ({} minimum)".format(minimum_records_required))
        return

    if args.pivot:
        tool = read_data(args.output)
        pivot_index = find_pivot(records, tool.markers)
        write_data(tool.markers, args.id or tool.id, args.output, pivot=tool.markers[pivot_index])
    else:
        points = process_marker_records(records, args.planar)
        points = convert_units(points, args.units)
        write_data(points, args.id, args.output)

if __name__ == "__main__":
    main()
