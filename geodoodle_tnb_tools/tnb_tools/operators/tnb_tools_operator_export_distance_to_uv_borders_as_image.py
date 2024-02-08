import os
import cv2
import bmesh
import mathutils
import numpy as np
import bpy
import bpy_extras


class TnbToolsOperatorExportDistToUvBorderAsImage(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):
    bl_idname       = "tnb_tools.operator_export_dist_to_uv_border_as_image"
    bl_label        = "tnb_tools_export_dist_to_uv_border_as_image"
    bl_description  = "Exports the distance of each pixel to the closest uv border (uses distance transform)"

    # ExportHelper mixin class uses this
    filename_ext = ".exr"

    filter_glob: bpy.props.StringProperty(
        default="*.exr",
        options={'HIDDEN'},
        maxlen=1000,  # Max internal buffer length, longer would be clamped.
    )

    image_size: bpy.props.IntProperty(
        name="Image width",
        description="Width of the output image",
        default=1000,
    )

    flip_image_vertically: bpy.props.BoolProperty(
        name="Flip image vert.",
        description="Wether or not to flip the image vertically",
        default=True,
    )

    enum_choices_distance_def = [('L1', 'Manhattan dist.', 'Manhattan distance', '', 0),
                                 ('L2', 'Euclidean dist.', 'Euclidean distance', '', 1)]
    
    distance_def: bpy.props.EnumProperty(
        items=enum_choices_distance_def,
        name="Distance definition",
        description="Either use Manhattan distance or Euclidean distance",
        default='L2',
    )

    enum_choices_distance_unit = [('Pixels', 'pixels', 'Distance to UV border is computed in pixels', '', 0),
                                  ('UV',     'UV',     'Distance to UV border is computed in UV',     '', 1)]
    
    distance_unit: bpy.props.EnumProperty(
        items=enum_choices_distance_unit,
        name="Distance unit",
        description="Either use Manhattan distance or Euclidean distance",
        default='UV',
    )

    @classmethod
    def poll(cls, context):
        single_object_selected = (len(bpy.context.selected_objects) == 1)
        
        if single_object_selected is False:
            return False
        
        selected_object_is_mesh = (bpy.context.selected_objects[0].type == 'MESH')

        return selected_object_is_mesh
    
    
    def execute(self, context):
        self.save_dist_to_uv_border_as_image(context)

        return {'FINISHED'}
    

    def save_dist_to_uv_border_as_image(self, context):
        try:        
            in_out_image = np.zeros((self.image_size, self.image_size, 3), np.float32)
            
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
        
        uv_layer = curr_edit_mesh.loops.layers.uv.verify()
        curr_edit_mesh.faces.ensure_lookup_table()

        for curr_face in curr_edit_mesh.faces:
            cur_face_v_uv_in_image = list()
            for loop in curr_face.loops:
                loop_uv = loop[uv_layer]
                cur_face_v_uv_in_image.append( np.array([int(loop_uv.uv[0] * self.image_size), int(loop_uv.uv[1] * self.image_size)]) )

            cv2.fillPoly(in_out_image, pts=[np.asarray(cur_face_v_uv_in_image)], color=(255, 255, 255, 255))

        in_out_image_gray = cv2.cvtColor(in_out_image, cv2.COLOR_BGR2GRAY) 

        # Threshold the image to create a binary image 
        _, in_out_image_bw = cv2.threshold(in_out_image_gray, 250, 255, cv2.THRESH_BINARY) 
        
        # Conversion to uint8 to be able to use distance transform
        in_out_image_bw = np.uint8(in_out_image_bw)
        
        # Calculate the distance transform 
        if self.distance_def == 'L1':
            dist_transform_image = cv2.distanceTransform(in_out_image_bw, cv2.DIST_L1, 3)
        elif self.distance_def == 'L2':
            dist_transform_image = cv2.distanceTransform(in_out_image_bw, cv2.DIST_L2, 3)
        else:
            dist_transform_image = cv2.distanceTransform(in_out_image_bw, cv2.DIST_C, 3) # Should enver happen

        if self.distance_unit == 'UV':
            dist_transform_image /= self.image_size

        if self.flip_image_vertically is True:
            dist_transform_image = cv2.flip(dist_transform_image, 0)
        
        cv2.imwrite(self.filepath, dist_transform_image)

        outu_file_path = self.filepath
        out_file_dir = os.path.dirname(outu_file_path)
        out_file_basename = os.path.basename(outu_file_path)
        out_file_basename_no_ext = os.path.splitext(out_file_basename)[0]
        out_save_png = os.path.join(out_file_dir, f'{out_file_basename_no_ext}.png')

        titi = cv2.imread(self.filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        assert titi.dtype == np.float32
        titi = cv2.normalize(titi, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        cv2.imwrite(out_save_png, titi)


    def execute(self, context):
        self.save_dist_to_uv_border_as_image(context)

        return {'FINISHED'}
    

def register():
    bpy.utils.register_class(TnbToolsOperatorExportDistToUvBorderAsImage)


def unregister():
    bpy.utils.unregister_class(TnbToolsOperatorExportDistToUvBorderAsImage)


if __name__ == "__main__":
    register()

    # test call
    bpy.ops.tnb_tools.operator_export_dist_to_uv_border_as_image('INVOKE_DEFAULT')