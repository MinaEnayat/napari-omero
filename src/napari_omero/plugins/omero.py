import sys
from functools import wraps

import napari
import numpy
from napari.layers.labels.labels import Labels as labels_layer
from napari.layers.points.points import Points as points_layer
from napari.layers.shapes.shapes import Shapes as shapes_layer
from qtpy.QtWidgets import QPushButton

import omero.clients
from napari_omero.utils import lookup_obj, obj_to_proxy_string
from napari_omero.plugins.loaders import parse_omero_shape, load_rois
from collections import defaultdict
from omero.cli import CLI, BaseControl, ProxyStringType
from omero.gateway import BlitzGateway, PixelsWrapper
from omero.model import (
    EllipseI,
    ImageI,
    LineI,
    PointI,
    PolygonI,
    PolylineI,
    RectangleI,
    RoiI,
)
from omero.rtypes import rdouble, rint, rstring

from .masks import save_labels

HELP = "Connect OMERO to the napari image viewer"

VIEW_HELP = "Usage: omero napari view Image:1"


def gateway_required(func):
    """Decorator which initializes a client and BlitzGateway.

    makes sure that all services of the Blitzgateway are closed again.
    """

    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        self.client = self.ctx.conn(*args)
        self.gateway = BlitzGateway(client_obj=self.client)
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.gateway is not None:
                self.gateway.close(hard=False)
                self.gateway = None
                self.client = None

    return _wrapper


class NapariControl(BaseControl):
    gateway = None
    client = None

    def _configure(self, parser):
        parser.add_login_arguments()
        sub = parser.sub()
        view = parser.add(sub, self.view, VIEW_HELP)

        obj_type = ProxyStringType("Image")

        view.add_argument("object", type=obj_type, help="Object to view")
        view.add_argument(
            "--eager",
            action="store_true",
            help=(
                "Use eager loading to load all planes immediately instead"
                "of lazy-loading each plane when needed"
            ),
        )

    @gateway_required
    def view(self, args):
        if isinstance(args.object, ImageI):
            try:
                img = lookup_obj(self.gateway, args.object)
            except NameError:
                self.ctx.die(110, f"No such {type}: {args.object.id}")

            self.ctx.out(f"View image: {img.name}")

            viewer = napari.Viewer()  # type: ignore

            add_buttons(viewer, img)

            viewer.open(
                f"omero://{obj_to_proxy_string(args.object)}",
                plugin="napari-omero",
            )
            set_dims_defaults(viewer, img)
            set_dims_labels(viewer, img)

            # add 'conn' and 'omero_image' to the viewer console
            viewer.update_console({"conn": self.gateway, "omero_image": img})
            napari.run()  # type: ignore


def add_buttons(viewer, img):
    """Add custom buttons to the viewer UI."""

    def handle_save_rois():
        save_rois(viewer, img)

    button = QPushButton("Save ROIs to OMERO")
    button.clicked.connect(handle_save_rois)
    viewer.window.add_dock_widget(button, name="Save OMERO", area="left")


def get_data(img, c=0):
    """
    Get 4D numpy array of pixel data, shape = (size_t, size_z, size_y, size_x).

    :param  img:        omero.gateway.ImageWrapper
    :c      int:        Channel index
    """
    size_z = img.getSizeZ()
    size_t = img.getSizeT()
    # get all planes we need in a single generator
    zct_list = [(z, c, t) for t in range(size_t) for z in range(size_z)]
    pixels = img.getPrimaryPixels()
    plane_gen = pixels.getPlanes(zct_list)

    t_stacks = []
    for _ in range(size_t):
        z_stack = [next(plane_gen) for _ in range(size_z)]
        t_stacks.append(numpy.array(z_stack))
    return numpy.array(t_stacks)


def set_dims_labels(viewer, image):
    """Set labels on napari viewer dims, based on dimensions of OMERO image.

    :param  viewer:     napari viewer instance
    :param  image:      omero.gateway.ImageWrapper
    """
    # dims (t, z, y, x) for 5D image
    dims = "TZ"

    for idx, label in enumerate(dims):
        viewer.dims.set_axis_label(idx, label)


def set_dims_defaults(viewer, image):
    """Set default Z/T index on napari viewer.

    Set Z/T slider index on napari viewer, according
    to default Z/T indecies of the OMERO image.

    :param  viewer:     napari viewer instance
    :param  image:      omero.gateway.ImageWrapper
    """
    # dims (t, z, y, x) for 5D image
    if image.getSizeT() > 1:
        viewer.dims.set_point(0, image.getDefaultT())
    if image.getSizeZ() > 1:
        viewer.dims.set_point(1, image.getDefaultZ())


def save_rois(viewer, image):
    """Save napari ROIs to OMERO.

    Usage: In napari, open console...
    >>> from napari_omero import *
    >>> save_rois(viewer, omero_image).
    """
    save_changes = False
    conn = image._conn
    img_id = image.id
    roi_service = conn.getRoiService()
    result = roi_service.findByImage(img_id, None)

    # Extract OMERO ROIs to compare with Napari layers for duplicate detection
    omero_rois = extract_omero_rois_coords(image, result)
    for layer in viewer.layers:
        if type(layer) is points_layer:
            for p in layer.data:
                points_layer_name = get_point_name(conn, image)
                print("Creating Points in roi", points_layer_name)
                # point = create_omero_point(p)
                # roi = create_roi(conn, image.id, [point])
                # print(f"Created ROI: {roi.id.val}")
        elif type(layer) is shapes_layer:
            if len(layer.data) == 0 or len(layer.shape_type) == 0:
                continue

            # Get the corresponding OMERO ROI layer name
            napari_layer_name = get_shapelayer_name(conn, image)
            if layer.name == napari_layer_name:
                shape_types = layer.shape_type
                if isinstance(shape_types, str):
                    shape_types = [
                        layer.shape_type for _ in range(len(layer.data))
                    ]

                # Collect existing OMERO shape IDs
                omero_shape_ids = {shape_id for roi in omero_rois.values() for shape_id in roi.keys()}

                napari_shape_ids = {shape_id for shape_id in layer.properties['shape_id'] if shape_id is not None}

                shapes_to_delete = omero_shape_ids - napari_shape_ids

                if shapes_to_delete:
                    # delete_rois(conn, result, shapes_to_delete)
                    save_changes = True

                # CHECK DUPLICATE, EDIT, CREATE NEW
                shapes_to_add = []
                napari_rois = layer.properties["roi_id"]
                napari_shapes = layer.properties["shape_id"]
                layer_data = group_rois_for_omero(layer.data, napari_rois, napari_shapes, shape_types)

                # Process each shape
                for data in layer_data:

                    shape_id = data['shape_id']
                    roi_id = data['roi_id']
                    napari_coords = data['coords']
                    napari_coords[:, 2:4] = numpy.round(napari_coords[:, 2:4], 6)

                    if shape_id is not None and shape_id in napari_shape_ids:

                        omero_shape_data = omero_rois[roi_id][shape_id]
                        omero_coords = numpy.array(omero_shape_data["coordinates"], dtype=numpy.float32)
                        omero_coords[:, 2:4] = numpy.round(omero_coords[:, 2:4], 6)
                        print("Omero coords", omero_coords)
                        print("napari coords", napari_coords)

                        same_coords = (
                            napari_coords.shape == omero_coords.shape
                            and numpy.allclose(napari_coords, omero_coords, atol=1e-5, equal_nan=True)
                        )
                        print("Same coords", same_coords)
                        if same_coords:
                            print("Duplicate")
                            continue
                        else:
                            print("We have to update and add")
                            save_changes = True
                            continue
                    else:
                        # New shape, prepare to create new ROI
                        shape = create_omero_shape(shape_type, data)
                        if shape is not None:
                            shapes_to_add.append(shape)

                    if shapes_to_add:
                        roi = create_roi(conn, img_id, [shape])
                        print(f"Created ROI: {roi.id.val}")
                    save_changes = True
                    # napari_shape_ids.clear()

            else:
                # Layer is not from OMERO, create new ROIs for all shapes
                shape_types = layer.shape_type

                if isinstance(shape_types, str):
                    shape_types = [
                        layer.shape_type for _ in range(len(layer.data))
                        ]

                for shape_type, data in zip(shape_types, layer.data):
                    shape = create_omero_shape(shape_type, data)
                    if shape is not None:
                        roi = create_roi(conn, image.id, [shape])
                        print(f"Created ROI: {roi.id.val}")
                save_changes = True

        elif type(layer) is labels_layer:
            print("Saving Labels...")
            save_labels(layer, image)

    return save_changes


def extract_omero_rois_coords(image, result):
    """Extract OMERO ROIs and their coordinates in (T, Z, Y, X) format."""
    omero_rois = {}

    # Loop over each ROI in the result
    for roi in result.rois:
        roi_id = roi.getId().getValue()
        if roi_id is None:
            continue

        omero_rois[roi_id] = {}

        # Loop over each shape in the ROI
        for shape in roi.copyShapes():
            shape_type = shape.__class__.__name__
            if shape is None:
                continue  # Skip invalid shapes

            shape_id = shape.getId().getValue()

            # Get Z and T indices
            theZ = shape.getTheZ()
            z_val = theZ.getValue() if theZ else None
            theT = shape.getTheT()
            t_val = theT.getValue() if theT else None

            # Handle OMERO Points separately
            if shape_type == "PointI":
                x = float(shape.getX().getValue())
                y = float(shape.getY().getValue())

                coords_4d = [[
                    (t_val if t_val is not None else None),
                    (z_val if z_val is not None else None),
                    y, x
                ]]
                meta_shape_type = "point"

            else:
                # Parse shape geometry using existing parser
                parsed = parse_omero_shape(shape)
                if parsed is None:
                    continue

                coords_2d, meta, _ = parsed
                coords_2d = numpy.round(coords_2d, 6)

                coords_4d = [[
                    (t_val if t_val is not None else None),
                    (z_val if z_val is not None else None),
                    y, x
                ] for y, x in coords_2d]
                meta_shape_type = meta["shape_type"]

            # Get optional text/comment for the shape
            text_value = shape.getTextValue()
            shape_text = text_value.getValue() if text_value else ""

            # Store data in the ROI dict
            omero_rois[roi_id][shape_id] = {
                "coordinates": coords_4d,
                "shape_text": shape_text,
                "shape_type": meta_shape_type,
            }

    return omero_rois


def get_point_name(conn, image):
    _, points_layer_meta = load_rois(conn, image, load_points=True)
    if points_layer_meta:
        return points_layer_meta.get("name", None)
    return None


def get_shapelayer_name(conn, image):
    _, roi_layer_meta = load_rois(conn, image, load_points=False)
    if roi_layer_meta:
        return roi_layer_meta.get("name", None)


def group_rois_for_omero(all_coords, roi_ids, shape_ids, shape_types):
    array_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    array_storage = defaultdict(lambda: defaultdict(dict))
    total_counts = defaultdict(lambda: defaultdict(int))
    shape_type_storage = defaultdict(dict)

    #  Count how many times each XY set appears
    for coords, roi_id, shape_id, shape_type in zip(all_coords, roi_ids, shape_ids, shape_types):
        arr = numpy.array(coords, dtype=numpy.float32).copy()
        yx_coords = tuple(map(tuple, numpy.round(arr[:, 2:4], 5)))

        if yx_coords not in array_storage[roi_id][shape_id]:
            array_storage[roi_id][shape_id][yx_coords] = arr.copy()

        array_counts[roi_id][shape_id][yx_coords] += 1
        total_counts[roi_id][shape_id] += 1
        shape_type_storage[roi_id][shape_id] = shape_type

    # Build final shapes for OMERO
    unique_shapes = []
    for roi_id, shape_dict in array_counts.items():
        for shape_id, counts in shape_dict.items():
            latest_arr = list(counts.keys())[-1]
            arr = array_storage[roi_id][shape_id][latest_arr].copy()

            # If multiple arrays → force T/Z = nan
            if total_counts[roi_id][shape_id] > 1:
                arr[:, 0:2] = numpy.nan

            unique_shapes.append({
                "roi_id": roi_id,
                "shape_id": shape_id,
                "shape_type": shape_type_storage[roi_id][shape_id],
                "coords": arr
            })

    return unique_shapes


def get_x(coordinate):
    return coordinate[-1]


def get_y(coordinate):
    return coordinate[-2]


def get_t(coordinate):
    return coordinate[0]


def get_z(coordinate):
    return coordinate[1]


def create_omero_point(data):
    point = PointI()
    point.x = rdouble(get_x(data))
    point.y = rdouble(get_y(data))
    point.theZ = rint(get_z(data))
    point.theT = rint(get_t(data))
    return point


def create_omero_shape(shape_type, data):
    # "line", "path", "polygon", "rectangle", "ellipse"
    # NB: assume all points on same plane.
    # Use first point to get Z and T index
    z_index = get_z(data[0])
    t_index = get_t(data[0])
    shape = None
    if shape_type == "line":
        shape = LineI()
        shape.x1 = rdouble(get_x(data[0]))
        shape.y1 = rdouble(get_y(data[0]))
        shape.x2 = rdouble(get_x(data[1]))
        shape.y2 = rdouble(get_y(data[1]))
    elif shape_type in ["path", "polygon"]:
        shape = PolylineI() if shape_type == "path" else PolygonI()
        # points = "10,20, 50,150, 200,200, 250,75"
        points = [f"{get_x(d)},{get_y(d)}" for d in data]
        if shape_type == "polygon" and points[0] != points[-1]:
            points.append(points[0])
        shape.points = rstring(", ".join(points))
    elif shape_type in ["rectangle", "ellipse"]:
        # corners go anti-clockwise starting top-left
        x1 = get_x(data[0])
        x2 = get_x(data[1])
        x3 = get_x(data[2])
        x4 = get_x(data[3])
        y1 = get_y(data[0])
        y2 = get_y(data[1])
        y3 = get_y(data[2])
        y4 = get_y(data[3])
        if shape_type == "rectangle":
            shape = RectangleI()
            shape.x = rdouble(x1)
            shape.y = rdouble(y1)
            shape.width = rdouble(x3 - x1)
            shape.height = rdouble(y3 - y1)
        elif shape_type == "ellipse":
            # Ellipse not rotated (ignore floating point rouding)
            shape = EllipseI()
            shape.x = rdouble((x1 + x3) / 2)
            shape.y = rdouble((y1 + y3) / 2)
            shape.radiusX = rdouble(abs(x3 - x1) / 2)
            shape.radiusY = rdouble(abs(y3 - y1) / 2)
    if shape is not None:
        shape.theZ = rint(z_index)
        shape.theT = rint(t_index)
    return shape


def create_roi(conn, img_id, shapes):
    updateService = conn.getUpdateService()
    roi = RoiI()
    roi.setImage(ImageI(img_id, False))
    for shape in shapes:
        roi.addShape(shape)
    return updateService.saveAndReturnObject(roi)


class NonCachedPixelsWrapper(PixelsWrapper):
    """Extend gateway.PixelWrapper to override _prepareRawPixelsStore."""

    def _prepareRawPixelsStore(self):
        """
        Creates RawPixelsStore and sets the id etc.

        This overrides the superclass behaviour to make sure that
        we don't re-use RawPixelStore in multiple processes since
        the Store may be closed in 1 process while still needed elsewhere.
        This is needed when napari requests may planes simultaneously,
        e.g. when switching to 3D view.
        """
        ps = self._conn.c.sf.createRawPixelsStore()
        ps.setPixelsId(self._obj.id.val, True, self._conn.SERVICE_OPTS)
        return ps


omero.gateway.PixelsWrapper = NonCachedPixelsWrapper
# Update the BlitzGateway to use our NonCachedPixelsWrapper
omero.gateway.refreshWrappers()


if __name__ == "__main__":
    # Register napari_omero as an OMERO CLI plugin
    cli = CLI()
    cli.register("napari", NapariControl, HELP)
    cli.invoke(sys.argv[1:])
