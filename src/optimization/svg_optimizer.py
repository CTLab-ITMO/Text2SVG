import logging
import math
import re
import xml.etree.ElementTree as etree
from xml.dom import minidom

import cssutils
import numpy as np
import svgelements
import svgpathtools
from svgpathtools.parser import parse_transform

cssutils.log.setLevel(logging.ERROR)

cnt = 1

from .geometry import (
    Circle,
    Ellipse,
    Path,
    Polygon,
    Rect
)


def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = etree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def my_float(x):
    # TODO: атрибут может быть в процентах/пикселях и т.п.
    return None if x is None else float(x)


def remove_namespaces(s):
    """
    Remove XML namespaces: {some-namespace}tag -> tag
    """
    return re.sub('{.*}', '', s)


def lowercase_attributes(elem):
    """
    Recursively convert all attribute names of this element (and children)
    to lowercase. The attribute *values* and text remain unchanged.
    """
    old_keys = list(elem.attrib.keys())
    for old_key in old_keys:
        new_key = old_key.lower()
        if new_key != old_key:
            elem.attrib[new_key] = elem.attrib.pop(old_key)
    for child in elem:
        lowercase_attributes(child)


def parse_common_attrib(node, transform):
    """
    Applies any transform attributes from the node (e.g. 'transform', 'gradienttransform') 
    by multiplying it into the 'transform' matrix and removing them from node.attrib.
    """
    for t in ['transform', 'gradienttransform']:
        if t in node.attrib:
            transform = transform @ parse_transform(node.attrib[t])
            node.attrib.pop(t)
    return transform


def is_shape(tag):
    """
    Returns True if the tag is one of the basic shape elements.
    """
    return tag in [
        'path',
        'polygon',
        'line',
        'polyline',
        'circle',
        'rect',
        'ellipse',
    ]


def is_linear_gradient(tag):
    return tag in ['linearGradient']


def is_radial_gradient(tag):
    return tag in ['radialGradient']


def transform_points(new_transform, points) -> np.ndarray:
    """
    Apply a 3x3 transform matrix to a list of 2D points.
    """
    points = np.array(points, dtype=np.float32).reshape(-1, 2)
    # Append a column of 1s, then do matrix multiply
    points = (
        np.concatenate((points, np.ones([points.shape[0], 1])), axis=1)
        @ np.transpose(new_transform, (1, 0))
    )
    # Divide by the last coordinate (homogeneous normalization)
    points = points / points[:, 2:3]
    # Take the first two coords
    points = points[:, :2]
    return points


def optimize_path(path_str, transform=np.eye(3)):
    """
    Parse the string path using svgpathtools, convert arcs to cubic, 
    remove near-zero length segments, etc. Transform all points, 
    then return a list of 'Path' objects with control-point data.
    """
    path = svgpathtools.parse_path(path_str)
    if len(path) == 0:
        return []
    ret_paths = []
    subpaths = path.continuous_subpaths()
    for subpath in subpaths:
        # Check if the subpath should be closed
        if subpath.isclosed():
            if (
                len(subpath) > 1
                and isinstance(subpath[-1], svgpathtools.Line)
                and subpath[-1].length() < 1e-5
            ):
                subpath.remove(subpath[-1])
                subpath[-1].end = subpath[0].start  # Force closing the path
                subpath.end = subpath[-1].end
                assert subpath.isclosed()
        else:
            beg = subpath[0].start
            end = subpath[-1].end
            if abs(end - beg) < 1e-5:
                subpath[-1].end = beg  # Force closing the path
                subpath.end = subpath[-1].end
                assert subpath.isclosed()

        num_control_points = []
        points = []

        for i, e in enumerate(subpath):
            # The start point of each segment
            if i == 0:
                points.append((e.start.real, e.start.imag))
            else:
                # Must begin from the end of the previous segment
                assert (e.start.real == points[-1][0])
                assert (e.start.imag == points[-1][1])
            # Identify type of segment
            if isinstance(e, svgpathtools.Line):
                num_control_points.append(0)
            elif isinstance(e, svgpathtools.QuadraticBezier):
                num_control_points.append(1)
                points.append((e.control.real, e.control.imag))
            elif isinstance(e, svgpathtools.CubicBezier):
                num_control_points.append(2)
                points.append((e.control1.real, e.control1.imag))
                points.append((e.control2.real, e.control2.imag))
            elif isinstance(e, svgpathtools.Arc):
                # Convert arcs to multiple cubic segments
                start = e.theta * math.pi / 180.0
                stop = (e.theta + e.delta) * math.pi / 180.0
                sign = 1.0
                if stop < start:
                    sign = -1.0

                epsilon = 0.00001
                while (sign * (stop - start) > epsilon):
                    arc_to_draw = stop - start
                    # Limit to quarter-pi increments
                    if arc_to_draw > 0.0:
                        arc_to_draw = min(arc_to_draw, 0.5 * math.pi)
                    else:
                        arc_to_draw = max(arc_to_draw, -0.5 * math.pi)
                    alpha = arc_to_draw / 2.0
                    cos_alpha = math.cos(alpha)
                    sin_alpha = math.sin(alpha)
                    cot_alpha = 1.0 / math.tan(alpha)
                    phi = start + alpha
                    cos_phi = math.cos(phi)
                    sin_phi = math.sin(phi)
                    lambda_ = (4.0 - cos_alpha) / 3.0
                    mu = sin_alpha + (cos_alpha - lambda_) * cot_alpha
                    last = sign * (stop - (start + arc_to_draw)) <= epsilon
                    num_control_points.append(2)

                    rx = e.radius.real
                    ry = e.radius.imag
                    cx = e.center.real
                    cy = e.center.imag
                    rot = e.phi * math.pi / 180.0
                    cos_rot = math.cos(rot)
                    sin_rot = math.sin(rot)

                    # First control point
                    x = lambda_ * cos_phi + mu * sin_phi
                    y = lambda_ * sin_phi - mu * cos_phi
                    xx = x * cos_rot - y * sin_rot
                    yy = x * sin_rot + y * cos_rot
                    points.append((cx + rx * xx, cy + ry * yy))

                    # Second control point
                    x = lambda_ * cos_phi - mu * sin_phi
                    y = lambda_ * sin_phi + mu * cos_phi
                    xx = x * cos_rot - y * sin_rot
                    yy = x * sin_rot + y * cos_rot
                    points.append((cx + rx * xx, cy + ry * yy))

                    # If it’s not the last arc piece, add an intermediate end point
                    if not last:
                        points.append(
                            (
                                cx + rx * math.cos(rot + start + arc_to_draw),
                                cy + ry * math.sin(rot + start + arc_to_draw),
                            )
                        )
                    start += arc_to_draw

            # The end point of the segment
            if i != len(subpath) - 1:
                points.append((e.end.real, e.end.imag))
            else:
                if subpath.isclosed():
                    # Must end at the beginning of first segment
                    assert (e.end.real == points[0][0])
                    assert (e.end.imag == points[0][1])
                else:
                    points.append((e.end.real, e.end.imag))

        # Transform the points
        points = transform_points(transform, points)

        # Construct our Path object (with num_control_points, points, closed?)
        ret_paths.append(Path(np.array(num_control_points), points, subpath.isclosed()))

    return ret_paths


def parse_style_attribute(s):
    """
    Given a 'style' string, parse it into a dict: key -> value
    """
    style_dict = {}
    for e in s.split(';'):
        key_value = e.split(':')
        if len(key_value) == 2:
            key = key_value[0].strip()
            value = key_value[1].strip()
            style_dict[key] = value
    return style_dict


def optimize_linear_gradient(
    node,
    transform,
    cubic_only=False,
    normalize_points=False,
    normalize_scale=1.0,
    normalize_to_int=False,
):
    tag = remove_namespaces(node.tag)
    new_transform = parse_common_attrib(node, transform)

    x1 = float(node.attrib['x1'])
    y1 = float(node.attrib['y1'])
    x2 = float(node.attrib['x2'])
    y2 = float(node.attrib['y2'])
    points = [(x1, y1), (x2, y2)]
    points = transform_points(new_transform, points)

    if normalize_points:
        points = points * normalize_scale
        if normalize_to_int:
            points = points.astype(np.int32)

    x1, y1 = points[0]
    x2, y2 = points[1]
    node.set('x1', f"{x1}")
    node.set('y1', f"{y1}")
    node.set('x2', f"{x2}")
    node.set('y2', f"{y2}")


def optimize_radial_gradient(
    node,
    transform,
    cubic_only=False,
    normalize_points=False,
    normalize_scale=1.0,
    normalize_to_int=False,
):
    # fx/fy not yet handled here
    tag = remove_namespaces(node.tag)
    new_transform = parse_common_attrib(node, transform)

    cx_ = node.attrib['cx']
    cy_ = node.attrib['cy']
    r_ = node.attrib['r']
    cx = parse_float(cx_)
    cy = parse_float(cy_)
    r = parse_float(r_)
    points = [(cx, cy)]
    points = transform_points(new_transform, points)

    if normalize_points:
        points = points * normalize_scale
        r = r * normalize_scale
        if normalize_to_int:
            points = points.astype(np.int32)
            r = int(r)

    cx, cy = points[0]
    # Only override if not percentage
    if '%' not in cx_:
        node.set('cx', f"{cx}")
    if '%' not in cy_:
        node.set('cy', f"{cy}")
    if '%' not in r_:
        node.set('r', f"{r}")


def optimize_shape(
    node,
    transform,
    cubic_only=False,
    normalize_points=False,
    normalize_scale=1.0,
    normalize_to_int=False,
):
    tag = remove_namespaces(node.tag)
    new_transform = parse_common_attrib(node, transform)

    attrs_to_normalize = ['stroke-width']

    # If there's a stroke but no stroke-width, default to 1
    if 'stroke' in node.attrib and 'stroke-width' not in node.attrib:
        node.set('stroke-width', '1')

    if tag == 'path':
        d = node.attrib['d']
        paths: list[Path] = optimize_path(d, new_transform)
        update_data_in_path_node(
            node,
            paths,
            cubic_only,
            normalize_points=normalize_points,
            normalize_scale=normalize_scale,
            normalize_to_int=normalize_to_int,
        )
    elif tag in ['circle', 'ellipse']:
        # Convert circle or ellipse to path
        cx = float(node.attrib['cx'])
        cy = float(node.attrib['cy'])

        if tag == 'circle':
            r = float(node.attrib['r'])
            circle_path_segments = svgelements.Circle(cx, cy, r).segments()
        else:
            rx = float(node.attrib['rx'])
            ry = float(node.attrib['ry'])
            circle_path_segments = svgelements.Ellipse(cx, cy, rx, ry).segments()

        circle_as_path = str(svgelements.Path(circle_path_segments))
        paths: list[Path] = optimize_path(circle_as_path, new_transform)

        node.tag = "path"
        update_data_in_path_node(
            node,
            paths,
            cubic_only,
            normalize_points=normalize_points,
            normalize_scale=normalize_scale,
            normalize_to_int=normalize_to_int,
        )
        # Remove the circle/ellipse attributes
        node.attrib.pop('r', None)
        node.attrib.pop('cx', None)
        node.attrib.pop('rx', None)
        node.attrib.pop('cy', None)
        node.attrib.pop('ry', None)

    elif tag in ['rect']:
        # Convert rect (with optional rx/ry) to path
        x = my_float(node.attrib.get('x', 0))
        y = my_float(node.attrib.get('y', 0))
        width = my_float(node.attrib.get('width', 0))
        height = my_float(node.attrib.get('height', 0))
        rx = my_float(node.attrib.get('rx', None))
        ry = my_float(node.attrib.get('ry', None))

        if rx is None and ry is not None:
            rx = ry
        elif rx is not None and ry is None:
            ry = rx
        elif rx is None and ry is None:
            rx = ry = 0

        rect_path_segments = svgelements.Rect(x, y, width, height, rx, ry).segments()
        rect_as_path = str(svgelements.Path(rect_path_segments))
        paths: list[Path] = optimize_path(rect_as_path, new_transform)

        node.tag = "path"
        update_data_in_path_node(
            node,
            paths,
            cubic_only,
            normalize_points=normalize_points,
            normalize_scale=normalize_scale,
            normalize_to_int=normalize_to_int,
        )
        # Remove the rect attributes
        node.attrib.pop('x', None)
        node.attrib.pop('rx', None)
        node.attrib.pop('y', None)
        node.attrib.pop('ry', None)
        node.attrib.pop('width', None)
        node.attrib.pop('height', None)

    elif tag in ['line']:
        # Transform the two endpoints
        x1 = float(node.attrib.get('x1', 0))
        y1 = float(node.attrib.get('y1', 0))
        x2 = float(node.attrib.get('x2', 0))
        y2 = float(node.attrib.get('y2', 0))

        points = [(x1, y1), (x2, y2)]
        points = transform_points(new_transform, points)
        if normalize_points:
            points = points * normalize_scale
            if normalize_to_int:
                points = points.astype(np.int32)
        x1, y1 = points[0]
        x2, y2 = points[1]
        node.set('x1', f"{x1}")
        node.set('y1', f"{y1}")
        node.set('x2', f"{x2}")
        node.set('y2', f"{y2}")

    elif tag in ['polygon', 'polyline']:
        # Transform the array of points
        pts = node.attrib['points'].strip()
        pts = pts.split(' ')
        pts = [[float(y) for y in re.split(',| ', x)] for x in pts if x]
        points = transform_points(new_transform, pts)
        if normalize_points:
            points = points * normalize_scale
            if normalize_to_int:
                points = points.astype(np.int32)

        path_str = ''
        for j in range(0, points.shape[0]):
            path_str += '{} {}'.format(points[j, 0], points[j, 1])
            if j != points.shape[0] - 1:
                path_str += ' '
        node.set('points', path_str)
    else:
        print(f"---- Warning: tag {tag} is not supported")

    # Possibly adjust the normalization scale based on uniform transform
    if abs(abs(new_transform[0, 0]) - abs(new_transform[1, 1])) < 0.0001 and abs(new_transform[0, 0]) > 1e-8:
        normalize_scale = normalize_scale * abs(new_transform[0, 0])

    normalize_attributes(node, attrs_to_normalize, normalize_points, normalize_scale, normalize_to_int)


def normalize_style_tag(node, attrs_to_normalize, normalize_points, normalize_scale, normalize_to_int):
    """
    If the node is a <style> block with CSS rules, parse them with cssutils and 
    apply transformations to relevant attributes (e.g., stroke-width).
    """

    def parse_stylesheet(node):
        defs = {}
        sheet = cssutils.parseString(node.text)
        for rule in sheet:
            if hasattr(rule, 'selectorText') and hasattr(rule, 'style'):
                name = rule.selectorText
                if len(name) >= 2 and name[0] == '.':
                    # e.g. .classname {...}
                    defs[name] = parse_style_attribute(rule.style.getCssText())
                else:
                    print(f"---- WARNING: stylesheet has unknown style: {name}")
        return defs

    defs = parse_stylesheet(node)

    splitted_defs = {}
    for name in defs:
        # Allow for comma-delimited selectors (like .class1, .class2)
        names = name.replace(" ", "").split(",")
        for n in names:
            if n not in splitted_defs:
                splitted_defs[n] = {}
            splitted_defs[n].update(defs[name])

    out_node_text = []
    for name in splitted_defs:
        style_dict = splitted_defs[name]
        # If stroke present, ensure stroke-width
        if 'stroke' in style_dict and 'stroke-width' not in style_dict:
            style_dict['stroke-width'] = '1'
        # Normalize relevant attributes
        for attr in attrs_to_normalize:
            if attr in style_dict:
                a = parse_float(style_dict[attr])
                a = a * normalize_scale
                if normalize_to_int:
                    a = int(a)
                style_dict[attr] = str(a)
        style_as_string = ";".join([f"{key}:{style_dict[key]}" for key in style_dict])
        out_node_text.append(f"{name}{{{style_as_string}}}")
    node.text = '\n'.join(out_node_text)


def normalize_attributes(
    node,
    attrs_to_normalize,
    normalize_points,
    normalize_scale,
    normalize_to_int
):
    """
    Normalize numeric attributes like stroke-width, x, y, etc.
    If normalize_points is False, or an attribute is missing, do nothing.
    """
    if normalize_points:
        for attr in attrs_to_normalize:
            if attr in node.attrib:
                a = parse_float(node.attrib[attr])
                a = a * normalize_scale
                if normalize_to_int:
                    a = int(a)
                node.set(attr, str(a))
        # Also handle inline style="stroke-width:..."
        if 'style' in node.attrib:
            style_dict: dict = parse_style_attribute(node.attrib['style'])
            for attr in attrs_to_normalize:
                if attr in style_dict:
                    a = parse_float(style_dict[attr])
                    a = a * normalize_scale
                    if normalize_to_int:
                        a = int(a)
                    style_dict[attr] = str(a)
            node.set('style', ";".join([f"{key}:{style_dict[key]}" for key in style_dict]))


def update_data_in_path_node(
    node,
    paths,
    cubic_only=False,
    normalize_points=False,
    normalize_scale=1.0,
    normalize_to_int=False,
):
    """
    Rebuild the 'd' attribute in <path> from the list of 
    Path objects returned by optimize_path().
    """
    node.set('d', "")
    is_last_shape_closed = False
    for shape in paths:
        num_segments = shape.num_control_points.shape[0]
        num_control_points = shape.num_control_points.data
        points = shape.points.data
        num_points = shape.points.shape[0]

        # Start command
        cur_points = [points[0, 0], points[0, 1]]
        if normalize_points:
            cur_points = np.array(cur_points) * normalize_scale
            if normalize_to_int:
                cur_points = cur_points.astype(np.int32)
        path_str = 'M {} {}'.format(*cur_points)

        point_id = 1
        for j in range(0, num_segments):
            if num_control_points[j] == 0:
                # Line segment
                p = point_id % num_points
                if cubic_only:
                    # Convert line -> cubic with 2 control points
                    l0_0 = points[point_id - 1, 0]
                    l0_1 = points[point_id - 1, 1]
                    l1_0 = points[p, 0]
                    l1_1 = points[p, 1]

                    cur_template = ' C {} {} {} {} {} {}'
                    cur_points = [
                        l0_0 + 1/3*(l1_0 - l0_0),
                        l0_1 + 1/3*(l1_1 - l0_1),
                        l0_0 + 2/3*(l1_0 - l0_0),
                        l0_1 + 2/3*(l1_1 - l0_1),
                        l1_0,
                        l1_1
                    ]
                else:
                    cur_template = ' L {} {}'
                    cur_points = [points[p, 0], points[p, 1]]
                cur_to_add_to_point_id = 1
            elif num_control_points[j] == 1:
                # Quadratic
                p1 = (point_id + 1) % num_points
                if cubic_only:
                    # Convert quadratic -> cubic
                    q0_0 = points[point_id - 1, 0]
                    q0_1 = points[point_id - 1, 1]
                    q1_0 = points[point_id, 0]
                    q1_1 = points[point_id, 1]
                    q2_0 = points[p1, 0]
                    q2_1 = points[p1, 1]

                    cur_template = ' C {} {} {} {} {} {}'
                    cur_points = [
                        q0_0 + 2/3*(q1_0 - q0_0),
                        q0_1 + 2/3*(q1_1 - q0_1),
                        q2_0 + 2/3*(q1_0 - q2_0),
                        q2_1 + 2/3*(q1_1 - q2_1),
                        q2_0,
                        q2_1
                    ]
                else:
                    # Keep it as Q
                    cur_template = ' Q {} {} {} {}'
                    cur_points = [
                        points[point_id, 0],
                        points[point_id, 1],
                        points[p1, 0],
                        points[p1, 1]
                    ]
                cur_to_add_to_point_id = 2
            elif num_control_points[j] == 2:
                # Cubic
                p2 = (point_id + 2) % num_points
                cur_template = ' C {} {} {} {} {} {}'
                cur_points = [
                    points[point_id, 0], points[point_id, 1],
                    points[point_id + 1, 0], points[point_id + 1, 1],
                    points[p2, 0], points[p2, 1]
                ]
                cur_to_add_to_point_id = 3
            else:
                print(f"---- Warning: control points {num_control_points[j]} not supported")
                raise

            # Normalize the segment points if asked
            if normalize_points:
                cur_points = np.array(cur_points) * normalize_scale
                if normalize_to_int:
                    cur_points = cur_points.astype(np.int32)

            path_str += cur_template.format(*cur_points)
            point_id += cur_to_add_to_point_id
            is_last_shape_closed = shape.is_closed

        # If the shape is closed, add Z
        if is_last_shape_closed:
            path_str += "Z"

        # Append to existing 'd' in case multiple sub-paths
        old_D_value = node.get('d', default="")
        prefix_D = (old_D_value + " ") if old_D_value != "" else ""
        node.set('d', prefix_D + path_str)


def parse_float(s):
    """
    Attempt to parse a float from a string that may contain letters 
    (e.g. '100px', '12pt', '50%'). We keep only digits, '.' and minus sign.
    """
    return float(''.join(i for i in s if (not i.isalpha())))


def optimize_from_node(
    node,
    transform=np.eye(3),
    cubic_only=False,
    normalize_points=False,
    normalize_scale=1.0,
    normalize_to_int=False,
):
    """
    Recursively optimize the SVG DOM from this node downward.
    """
    transform = parse_common_attrib(node, transform)

    kwargs = {
        "transform": transform,
        "cubic_only": cubic_only,
        "normalize_points": normalize_points,
        "normalize_scale": normalize_scale,
        "normalize_to_int": normalize_to_int,
    }

    cur_tag = remove_namespaces(node.tag)

    # Common attributes to normalize, e.g. 'stroke-width', 'x', 'y', 'font-size', etc.
    attrs_to_normalize = ['stroke-width', 'x', 'y', 'x1', 'x2', 'y1', 'y2', 'font-size', 'line-height']
    if cur_tag != 'svg':
        attrs_to_normalize.extend(['width', 'height'])

    # If there's stroke but no stroke-width, set a default
    if 'stroke' in node.attrib and 'stroke-width' not in node.attrib:
        node.set('stroke-width', '1')

    # Process child nodes
    for child in node:
        tag = remove_namespaces(child.tag)
        if is_shape(tag):
            optimize_shape(child, **kwargs)
        elif is_linear_gradient(tag):
            optimize_linear_gradient(child, **kwargs)
        elif is_radial_gradient(tag):
            optimize_radial_gradient(child, **kwargs)
        else:
            optimize_from_node(child, **kwargs)

    # If it's <style>, handle CSS
    if cur_tag == 'style':
        normalize_style_tag(node, attrs_to_normalize, normalize_points, normalize_scale, normalize_to_int)
    else:
        # Otherwise, just normalize numeric attributes
        normalize_attributes(node, attrs_to_normalize, normalize_points, normalize_scale, normalize_to_int)


def optimize_from_root(
    root,
    cubic_only=False,
    normalize_points=False,
    normalize_scale=1.0,
    normalize_to_int=False,
):
    """
    Starting point to optimize an entire SVG tree.
    Adjusts the root's viewBox/width/height if necessary, 
    then calls optimize_from_node(root).
    """
    # root is assumed to have been processed by lowercase_attributes(), so we check 'viewbox'
    if 'viewbox' in root.attrib:
        # If there's a viewBox, parse it out
        view_box_array = root.attrib['viewbox'].replace(",", "").split()
        canvas_width = parse_float(view_box_array[2])
        canvas_height = parse_float(view_box_array[3])
        view_box = [
            parse_float(view_box_array[0]),
            parse_float(view_box_array[1]),
            parse_float(view_box_array[2]),
            parse_float(view_box_array[3]),
        ]
    else:
        # If no viewBox, rely on width/height
        view_box = [0, 0, 0, 0]
        if 'width' in root.attrib:
            canvas_width = parse_float(root.attrib['width'])
            view_box[2] = canvas_width
        else:
            print("---- Warning: Can't find canvas width.")
            canvas_width = 1.0
        if 'height' in root.attrib:
            canvas_height = parse_float(root.attrib['height'])
            view_box[3] = canvas_height
        else:
            print("---- Warning: Can't find canvas height.")
            canvas_height = 1.0

    # Ensure root has numeric width/height
    if 'width' not in root.attrib or '%' in root.attrib['width']:
        root.set('width', str(view_box[2]))
    if 'height' not in root.attrib or '%' in root.attrib['height']:
        root.set('height', str(view_box[3]))

    kwargs = {}
    if normalize_points:
        # Scale everything so that the largest side is normalize_scale units
        scale = normalize_scale / max(canvas_width, canvas_height)
        view_box = [x * scale for x in view_box]
        if normalize_to_int:
            view_box = [int(x) for x in view_box]
        root.set('viewbox', f"{view_box[0]} {view_box[1]} {view_box[2]} {view_box[3]}")
        kwargs["normalize_scale"] = scale

    optimize_from_node(
        root,
        cubic_only=cubic_only,
        normalize_points=normalize_points,
        normalize_to_int=normalize_to_int,
        **kwargs
    )
    return root


def optimize_svg_from_str(
    svg_str,
    cubic_only=False,
    normalize_points=False,
    normalize_scale=1.0,
    normalize_to_int=False,
):
    """
    Main entry point: pass in an SVG string, return an optimized Element.
    """
    root = etree.fromstring(svg_str)

    # Convert all attribute names to lowercase so "viewBox" -> "viewbox" etc.
    lowercase_attributes(root)

    return optimize_from_root(
        root,
        cubic_only=cubic_only,
        normalize_points=normalize_points,
        normalize_scale=normalize_scale,
        normalize_to_int=normalize_to_int,
    )


def optimize_svg_from_file(
    filename,
    cubic_only=False,
    normalize_points=False,
    normalize_scale=1.0,
    normalize_to_int=False,
):
    """
    Main entry point: pass in an SVG file, return an optimized Element.
    """
    tree = etree.parse(filename)
    root = tree.getroot()

    # Convert all attribute names to lowercase so "viewBox" -> "viewbox" etc.
    lowercase_attributes(root)

    return optimize_from_root(
        root,
        cubic_only=cubic_only,
        normalize_points=normalize_points,
        normalize_scale=normalize_scale,
        normalize_to_int=normalize_to_int,
    )


def postfix_svg_root(root):
    """
    Remove any namespace from the tags (e.g., '{http://www.w3.org/2000/svg}svg' -> 'svg')
    And set the xmlns attribute back to "http://www.w3.org/2000/svg".
    """
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
    root.set("xmlns", "http://www.w3.org/2000/svg")


# TODO: not supporting 'pt', '%' fully, or stroke-dasharray, filter elements, etc.
# TODO: arcs might need better approximation
# TODO: references/links are not well-supported
# TODO: The transform scale for stroke-width can be tricky if transform is non-uniform.
#       (E.g., scaleX != scaleY)
#       Might need more sophisticated logic to handle that properly.
