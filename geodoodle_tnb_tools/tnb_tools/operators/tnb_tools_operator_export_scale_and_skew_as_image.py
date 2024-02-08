import os
import cv2
import bmesh
import mathutils
import numpy as np
import bpy
import bpy_extras


class TnbToolsOperatorExportScaleAndSkewAsImage(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):
    bl_idname       = "tnb_tools.operator_export_scale_and_skew_as_image"
    bl_label        = "tnb_tools_export_scale_and_skew_as_image"
    bl_description  = "Exports the deformation induced by the current parameterization (scalings and skewing) as an image"


    # ExportHelper mixin class uses this
    filename_ext = ".exr"

    filter_glob: bpy.props.StringProperty(
        default="*.exr",
        options={'HIDDEN'},
        maxlen=1000,  # Max internal buffer length, longer would be clamped.
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
        self.compute_scale_skew(context)
        self.output_tnb_scale_skew_images()

        return {'FINISHED'}
    

    def compute_scale_skew(self, context):
        if bpy.context.mode == 'EDIT_MESH':
            edit_object = bpy.context.edit_object
            mesh_edit   = bmesh.from_edit_mesh(edit_object.data)
        elif bpy.context.mode == 'OBJECT' and bpy.context.object.type == 'MESH':
            mesh_edit = bmesh.new()
            mesh_edit.from_mesh(bpy.context.object.data)
            
        
        mesh_edit_scale_x_layer = mesh_edit.faces.layers.float.get('tnb_scale_x')
        if mesh_edit_scale_x_layer is None:
            mesh_edit_scale_x_layer = mesh_edit.faces.layers.float.new('tnb_scale_x')

        mesh_edit_scale_y_layer = mesh_edit.faces.layers.float.get('tnb_scale_y')
        if mesh_edit_scale_y_layer is None:
            mesh_edit_scale_y_layer = mesh_edit.faces.layers.float.new('tnb_scale_y')

        mesh_edit_skew_top_right_layer = mesh_edit.faces.layers.float.get('tnb_skew_top_right')
        if mesh_edit_skew_top_right_layer is None:
            mesh_edit_skew_top_right_layer = mesh_edit.faces.layers.float.new('tnb_skew_top_right')

        mesh_edit_skew_bot_left_layer = mesh_edit.faces.layers.float.get('tnb_skew_bot_left')
        if mesh_edit_skew_bot_left_layer is None:
            mesh_edit_skew_bot_left_layer = mesh_edit.faces.layers.float.new('tnb_skew_bot_left')

        uv_layer = mesh_edit.loops.layers.uv.verify()

        for curr_face in mesh_edit.faces:
            v0 = curr_face.verts[0].co
            v1 = curr_face.verts[1].co
            v2 = curr_face.verts[2].co

            uv_in_image = list()
            for loop in curr_face.loops:
                loop_uv = loop[uv_layer]
                uv_in_image.append(np.array([loop_uv.uv[0], loop_uv.uv[1], 0.0]))

            v0_uv = uv_in_image[0]
            v1_uv = uv_in_image[1]
            v2_uv = uv_in_image[2]
            
            Ds = np.identity(3)
            Ds[:, 0] = v1 - v0
            Ds[:, 1] = v2 - v0
            Ds[:, 2] = curr_face.normal.normalized()

            Dm = np.identity(3)
            Dm[:, 0] = v1_uv - v0_uv
            Dm[:, 1] = v2_uv - v0_uv
            
            tmp_vec = np.cross((v1_uv - v0_uv), (v2_uv - v0_uv))
            if np.linalg.norm(tmp_vec) == 0:
                self.report({'ERROR'}, f'Cannot normalize a null vector')
                return
            
            tmp_vec_normalized = tmp_vec / np.linalg.norm(tmp_vec)
            Dm[:, 2] = tmp_vec_normalized
            inv_Dm = np.linalg.inv(Dm)
            
            curr_deformation_gradient = np.matmul(Ds, inv_Dm)
            
            U_mat, S_vec, V_mat = np.linalg.svd(curr_deformation_gradient,
                                                full_matrices=True,
                                                compute_uv=True,
                                                hermitian=False)

            S_mat = np.diag(S_vec)
            
            R_mat = np.matmul(U_mat, V_mat)
            det_R_mat = np.linalg.det(R_mat)
            L_mat = np.identity(3)
            L_mat[2, 2] = det_R_mat
            
            S_mat = np.matmul(S_mat, L_mat)

            det_U_mat = np.linalg.det(U_mat)
            det_V_mat = np.linalg.det(V_mat)
            
            if det_U_mat < 0 and det_V_mat > 0:
                U_mat = np.matmul(U_mat, L_mat)

            if det_U_mat > 0 and det_V_mat < 0:
                V_mat = V_mat.transpose()
                V_mat = np.matmul(V_mat, L_mat)
                V_mat = V_mat.transpose()
            
            R_mat = np.matmul(U_mat, V_mat)
            curr_face_sim = np.matmul(np.matmul(V_mat.transpose(), S_mat), V_mat)

            if abs(curr_face_sim[0,1] - curr_face_sim[1, 0]) > 1e-3:
                print('Erreur: matrice non symetrique!')
                print(curr_face_sim)

            curr_face[mesh_edit_scale_x_layer]          = curr_face_sim[0, 0]
            curr_face[mesh_edit_scale_y_layer]          = curr_face_sim[1, 1]
            curr_face[mesh_edit_skew_top_right_layer]   = curr_face_sim[0, 1]
            curr_face[mesh_edit_skew_bot_left_layer]    = curr_face_sim[1, 0]

        if bpy.context.mode == 'EDIT_MESH':
            bmesh.update_edit_mesh(edit_object.data)
        elif bpy.context.mode == 'OBJECT' and bpy.context.object.type == 'MESH':
            mesh_edit.to_mesh(bpy.context.object.data)


    def output_tnb_scale_skew_images(self):
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
        
        curr_edit_mesh_scale_x_layer        = curr_edit_mesh.faces.layers.float.get('tnb_scale_x')
        curr_edit_mesh_scale_y_layer        = curr_edit_mesh.faces.layers.float.get('tnb_scale_y')
        curr_edit_mesh_skew_top_right_layer = curr_edit_mesh.faces.layers.float.get('tnb_skew_top_right')
        curr_edit_mesh_skew_bot_left_layer  = curr_edit_mesh.faces.layers.float.get('tnb_skew_bot_left')
        uv_layer                            = curr_edit_mesh.loops.layers.uv.verify()
        
        if curr_edit_mesh_scale_x_layer is None or curr_edit_mesh_scale_y_layer is None or curr_edit_mesh_skew_top_right_layer is None or curr_edit_mesh_skew_bot_left_layer is None:
            self.report({'ERROR'}, f'Cannot save deformations to file: deformation values do not seem to exit!')
            return

        curr_edit_mesh.faces.ensure_lookup_table()

        for curr_face in curr_edit_mesh.faces:
            uv_in_image = list()
            for loop in curr_face.loops:
                loop_uv = loop[uv_layer]
                uv_in_image.append([int(loop_uv.uv[0] * self.image_size), int(loop_uv.uv[1] * self.image_size)])
            
            curr_color = (curr_face[curr_edit_mesh_scale_x_layer],
                          curr_face[curr_edit_mesh_skew_top_right_layer],
                          curr_face[curr_edit_mesh_scale_y_layer])
            
            cv2.fillPoly(output_image, pts=[np.asarray(uv_in_image)], color=(curr_color))
            

        if self.flip_image_vertically is True:
            output_image = cv2.flip(output_image, 0)

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
    bpy.utils.register_class(TnbToolsOperatorExportScaleAndSkewAsImage)


def unregister():
    bpy.utils.unregister_class(TnbToolsOperatorExportScaleAndSkewAsImage)


if __name__ == "__main__":
    register()

    # # test call
    # bpy.ops.geodoodle_tnb_direction.save_scale_skew_as_images('INVOKE_DEFAULT')