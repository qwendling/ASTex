# MIT License
#
# Copyright (c) 2021 Lukas Toenne
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

bl_info = {
    "name": "Tnb directions",
    "description": "Compute geodesic distances using a heat method and additional tools for tiling and blending (IGG, ICUBE, Univ. Strasbourg)",
    "author": "Joris Ravaglia, built uppon the work of Lukas Toenne",
    "version": (0, 1),
    "blender": (4, 0, 0),
    "location": "View3D > Object > Geodesic Distance",
    "warning": "Installs scipy and opencv",
    "doc_url": "https://github.com/lukas-toenne/geodoodle",
    "tracker_url": "https://github.com/lukas-toenne/geodoodle/issues",
    "support": "COMMUNITY",
    "category": "Mesh",
}

# Install required packages (requires administrator rights)
from .tnb_tools.tools import tnb_direction_install_package_in_blender
tnb_direction_install_package_in_blender.install_package_if_required('scipy', 'scipy')
tnb_direction_install_package_in_blender.install_package_if_required('cv2', 'opencv-python')

# Import Geodoodle tools (previous work of: Lukas Toenne - 2021)
from .geodoodle_tools import geodoodle_geometry_math
from .geodoodle_tools import geodoodle_layers
from .geodoodle_tools import geodoodle_ngon_mesh_refine
from .geodoodle_tools import geodoodle_operator
from .geodoodle_tools import geodoodle_surface_vector
from .geodoodle_tools import geodoodle_triangle_mesh
from .geodoodle_tools import geodoodle_util

# Import tnb tools
from .tnb_tools.gizmo     import tnb_tools_gizmo_manual_direction

from .tnb_tools.operators import tnb_tools_operator_compute_directions_from_geodesic_gradient
from .tnb_tools.operators import tnb_tools_operator_compute_horizontal_directions
from .tnb_tools.operators import tnb_tools_operator_export_distance_to_uv_borders_as_image
from .tnb_tools.operators import tnb_tools_operator_export_edges_voronoi_as_image
from .tnb_tools.operators import tnb_tools_operator_export_lissage_triangulation
from .tnb_tools.operators import tnb_tools_operator_export_scale_and_skew_as_image
from .tnb_tools.operators import tnb_tools_operator_export_texture_directions_as_csv
from .tnb_tools.operators import tnb_tools_operator_export_texture_directions_as_image
from .tnb_tools.operators import tnb_tools_operator_triangulate
from .tnb_tools.operators import tnb_tools_operator_view_directions
from .tnb_tools.operators import tnb_tools_operator_view_vertex_group_weights

from .tnb_tools.ui        import tnb_tools_ui


if "bpy" in locals():
    import importlib
    importlib.reload(geodoodle_util)
    importlib.reload(geodoodle_triangle_mesh)
    importlib.reload(geodoodle_geometry_math)
    importlib.reload(geodoodle_layers)
    importlib.reload(geodoodle_operator)
    importlib.reload(geodoodle_surface_vector)
    importlib.reload(geodoodle_ngon_mesh_refine)


def register():
    geodoodle_operator.register()

    tnb_tools_operator_compute_horizontal_directions.register()
    tnb_tools_gizmo_manual_direction.register()
    tnb_tools_operator_compute_directions_from_geodesic_gradient.register()
    tnb_tools_operator_view_directions.register()
    tnb_tools_operator_export_distance_to_uv_borders_as_image.register()
    tnb_tools_operator_export_edges_voronoi_as_image.register()
    tnb_tools_operator_export_lissage_triangulation.register()
    tnb_tools_operator_export_texture_directions_as_csv.register()
    tnb_tools_operator_export_scale_and_skew_as_image.register()
    tnb_tools_operator_export_texture_directions_as_image.register()
    tnb_tools_operator_view_vertex_group_weights.register()

    tnb_tools_operator_triangulate.register()

    tnb_tools_ui.register()

def unregister():
    geodoodle_operator.unregister()

    tnb_tools_operator_compute_horizontal_directions.unregister()
    tnb_tools_gizmo_manual_direction.unregister()
    tnb_tools_operator_compute_directions_from_geodesic_gradient.unregister()
    tnb_tools_operator_view_directions.unregister()
    tnb_tools_operator_export_distance_to_uv_borders_as_image.unregister()
    tnb_tools_operator_export_edges_voronoi_as_image.unregister()
    tnb_tools_operator_export_lissage_triangulation.unregister()
    tnb_tools_operator_export_texture_directions_as_csv.unregister()
    tnb_tools_operator_export_scale_and_skew_as_image.unregister()
    tnb_tools_operator_export_texture_directions_as_image.unregister()
    tnb_tools_operator_view_vertex_group_weights.unregister()
    
    tnb_tools_operator_triangulate.unregister()
    
    tnb_tools_ui.unregister()    


if __name__ == "__main__":
    register()
