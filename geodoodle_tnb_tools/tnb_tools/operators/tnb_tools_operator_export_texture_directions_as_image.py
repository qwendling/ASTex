import os
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


class TnbToolsOperatorExportTextureDirectionsAsImage(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):
    bl_idname       = "tnb_tools.operator_export_texture_directions_as_image"
    bl_label        = "tnb_tools_export_texture_directions_as_image"
    bl_description  = "Exports the expected (3D) directions of the texture as an image"

    # ExportHelper mixin class uses this
    filename_ext = ".exr"

    filter_glob: bpy.props.StringProperty(
        default="*.exr",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    image_size: bpy.props.IntProperty(
        name="Image size (width = height)",
        description="Width and height of the output image",
        default=1000,
    )

    flip_image_vertically: bpy.props.BoolProperty(
        name="Flip image vert.",
        description="Wether or not to flip the image vertically",
        default=True,
    )


    @classmethod
    def poll(cls, context):
        single_object_selected = (len(bpy.context.selected_objects) == 1)
        
        if single_object_selected is False:
            return False
        
        selected_object_is_mesh = (bpy.context.selected_objects[0].type == 'MESH')

        return selected_object_is_mesh
    
    
    def execute(self, context):
        self.output_tnb_direction_image()

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


    def output_tnb_direction_image(self):
        try:        
            output_image = np.zeros((self.image_size, self.image_size, 3), np.float32)
            
        except:
            self.report({'ERROR'}, f'Could not save image with auto size (image too big? Width: {self.image_size}, height: {self.image_size})')
            return

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

        uv_layer = curr_edit_mesh.loops.layers.uv.verify()
        curr_edit_mesh.faces.ensure_lookup_table()
        
        for curr_face in curr_edit_mesh.faces:
            curr_rotation       = curr_edit_mesh.faces[curr_face.index][curr_edit_mesh_rotation_layer]
            rotation_matrix     = mathutils.Matrix.Rotation(curr_rotation, 3, (0.0, 0.0, 1.0))
            tnb_direction_3d    = TnbToolsOperatorExportTextureDirectionsAsImage.compute_transfor_matrix(curr_face) @ rotation_matrix @ mathutils.Vector((1.0, 0.0, 0.0))
            
            uv_in_image = list()
            for loop in curr_face.loops:
                loop_uv = loop[uv_layer]
                uv_in_image.append([int(loop_uv.uv[0] * self.image_size), int(loop_uv.uv[1] * self.image_size)])
            
            curr_color = (((tnb_direction_3d[0] + 1.0)/2.0) *255.0,
                          ((tnb_direction_3d[1] + 1.0)/2.0) *255.0,
                          ((tnb_direction_3d[2] + 1.0)/2.0) *255.0)
            
            cv2.fillPoly(output_image, pts=[np.asarray(uv_in_image)], color=(curr_color))
            
        if self.flip_image_vertically is True:
            output_image = cv2.flip(output_image, 0)

        cv2.imwrite(self.filepath, output_image)

        assert output_image.dtype == np.float32
        cv2.imwrite(self.filepath, output_image)

        outu_file_path = self.filepath
        out_file_dir = os.path.dirname(outu_file_path)
        out_file_basename = os.path.basename(outu_file_path)
        out_file_basename_no_ext = os.path.splitext(out_file_basename)[0]
        out_save_png = os.path.join(out_file_dir, f'{out_file_basename_no_ext}.png')

        titi = cv2.imread(self.filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        assert titi.dtype == np.float32
        titi = cv2.normalize(titi, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        cv2.imwrite(out_save_png, titi)


def register():
    bpy.utils.register_class(TnbToolsOperatorExportTextureDirectionsAsImage)


def unregister():
    bpy.utils.unregister_class(TnbToolsOperatorExportTextureDirectionsAsImage)


if __name__ == "__main__":
    register()

    # test call
    # bpy.ops.geodoodle_tnb_direction.save_to_image('INVOKE_DEFAULT')