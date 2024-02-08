import cv2
import bmesh
import mathutils
import numpy as np
import bpy
import bpy_extras

from bpy.props import StringProperty
from bpy.props import IntProperty
from bpy.props import BoolProperty
from bpy.props import EnumProperty


class TnbToolsOperatorExportTextureDirectionsAsCsv(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):
    bl_idname       = "tnb_tools.operator_export_texture_directions_as_csv"
    bl_label        = "tnb_tools_export_texture_directions_as_csv"
    bl_description  = "Exports the expected (3D) directions of the texture as CSV"

    # ExportHelper mixin class uses this
    filename_ext = ".csv"

    filter_glob: bpy.props.StringProperty(
        default="*.csv",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )


    @classmethod
    def poll(cls, context):
        single_object_selected = (len(bpy.context.selected_objects) == 1)
        
        if single_object_selected is False:
            return False
        
        selected_object_is_mesh = (bpy.context.selected_objects[0].type == 'MESH')

        return selected_object_is_mesh
    
    
    def execute(self, context):
        self.output_tnb_direction_as_csv()

        return {'FINISHED'}
    

    @classmethod
    def compute_transfor_matrix(cls, selected_face):
        v_0 = selected_face.verts[0].co
        v_1 = selected_face.verts[1].co

        local_z = selected_face.normal
        local_x = v_1 - v_0
        local_y = local_x.cross(local_z)

        local_x.normalize()
        local_y.normalize()
        local_z.normalize()

        local_basis_in_world = mathutils.Matrix.Identity(3)
        for i in range(0, 3):
            local_basis_in_world[i][0] = local_x[i]
            local_basis_in_world[i][1] = local_y[i]
            local_basis_in_world[i][2] = local_z[i]

        return local_basis_in_world


    def output_tnb_direction_as_csv(self):
        if bpy.context.mode == 'EDIT_MESH':
            curr_edit_mesh = bmesh.from_edit_mesh(bpy.context.object.data)
        elif bpy.context.mode == 'OBJECT' and bpy.context.object.type == 'MESH':
            curr_edit_mesh = bmesh.new()
            curr_edit_mesh.from_mesh(bpy.context.object.data)
        else:
            return {'CANCELLED'}
        
        curr_edit_mesh_rotation_layer = curr_edit_mesh.faces.layers.float.get('tnb_rotation_from_v0_v1')
        
        if curr_edit_mesh_rotation_layer is None:
            self.report({'ERROR'}, f'Cannot save directions to image: it seems no directions were computed!')
            return

        with open(self.filepath, mode='w') as out_file:
            curr_edit_mesh.faces.ensure_lookup_table()
            for curr_face in curr_edit_mesh.faces:
                curr_rotation       = curr_edit_mesh.faces[curr_face.index][curr_edit_mesh_rotation_layer]
                rotation_matrix     = mathutils.Matrix.Rotation(curr_rotation, 3, (0.0, 0.0, 1.0))
                tnb_direction_3d    = TnbToolsOperatorExportTextureDirectionsAsCsv.compute_transfor_matrix(curr_face) @ rotation_matrix @ mathutils.Vector((1.0, 0.0, 0.0))
                out_file.write(f'{tnb_direction_3d[0]},{tnb_direction_3d[1]},{tnb_direction_3d[2]}\n')


def register():
    bpy.utils.register_class(TnbToolsOperatorExportTextureDirectionsAsCsv)


def unregister():
    bpy.utils.unregister_class(TnbToolsOperatorExportTextureDirectionsAsCsv)


if __name__ == "__main__":
    register()

    # test call
    # bpy.ops.geodoodle_tnb_direction.save_to_image('INVOKE_DEFAULT')