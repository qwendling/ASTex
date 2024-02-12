from ..operators import tnb_tools_operator_compute_directions_from_geodesic_gradient
from ..operators import tnb_tools_operator_compute_horizontal_directions
from ..operators import tnb_tools_operator_export_distance_to_uv_borders_as_image
from ..operators import tnb_tools_operator_export_edges_voronoi_as_image
from ..operators import tnb_tools_operator_export_lissage_triangulation
from ..operators import tnb_tools_operator_export_scale_and_skew_as_image
from ..operators import tnb_tools_operator_export_texture_directions_as_csv
from ..operators import tnb_tools_operator_export_texture_directions_as_image
from ..operators import tnb_tools_operator_triangulate
from ..operators import tnb_tools_operator_view_directions
from ..operators import tnb_tools_operator_view_vertex_group_weights

import bpy


class VIEW3D_PT_tnb_ui(bpy.types.Panel):
    bl_space_type   = 'VIEW_3D'
    bl_region_type  = 'UI'
    bl_category     = 'TnB Tools'
    bl_label        = 'TnB_Tools_UI'

    def draw(self, context):
        layout = self.layout

        layout.operator(tnb_tools_operator_triangulate.TnbToolsOperatorTriangulate.bl_idname,
                        text='Triangulate mesh',
                        icon='MOD_TRIANGULATE')

        layout.operator(tnb_tools_operator_view_directions.TnbToolsOperatorViewDirections.bl_idname,
                        text='View TnB directions',
                        icon='INFO')
        
        layout.operator(tnb_tools_operator_view_vertex_group_weights.TnbToolsOperatorViewVertexGroupWeights.bl_idname,
                        text='View vertex group weights',
                        icon='INFO')
        
        layout.operator(tnb_tools_operator_compute_horizontal_directions.TnbToolsOperatorComputeHorizontalDirectionsAroundVertexOperator.bl_idname,
                        text='Horizontal dir. around vert.',
                        icon='ORIENTATION_NORMAL')
        
        layout.operator(tnb_tools_operator_compute_directions_from_geodesic_gradient.TnbToolsOperatorComputeDirectionsFromGeodesicGradient.bl_idname,
                        text='Dir. from geodesics',
                        icon='ORIENTATION_NORMAL')

        layout.operator(tnb_tools_operator_export_distance_to_uv_borders_as_image.TnbToolsOperatorExportDistToUvBorderAsImage.bl_idname,
                        text='Export geodesics to uv borders as image',
                        icon='EXPORT')
        
        layout.operator(tnb_tools_operator_export_edges_voronoi_as_image.TnbToolsOperatorExportEdgesVoronoiAsImage.bl_idname,
                        text='Export edge Voronoi',
                        icon='EXPORT')
        
        layout.operator(tnb_tools_operator_export_lissage_triangulation.TnbToolsOperatorExportSmoothedValuesOnFaces.bl_idname,
                        text='Lissage d\'une valeur',
                        icon='EXPORT')

        layout.operator(tnb_tools_operator_export_scale_and_skew_as_image.TnbToolsOperatorExportScaleAndSkewAsImage.bl_idname,
                        text='Export parameterization deformation as image',
                        icon='EXPORT')
        
        layout.operator(tnb_tools_operator_export_texture_directions_as_image.TnbToolsOperatorExportTextureDirectionsAsImage.bl_idname,
                        text='Export texture directions as image',
                        icon='EXPORT')
        
        layout.operator(tnb_tools_operator_export_texture_directions_as_csv.TnbToolsOperatorExportTextureDirectionsAsCsv.bl_idname,
                        text='Export texture directions as csv',
                        icon='EXPORT')


def register():
    bpy.utils.register_class(VIEW3D_PT_tnb_ui)


def unregister():
    bpy.utils.unregister_class(VIEW3D_PT_tnb_ui)