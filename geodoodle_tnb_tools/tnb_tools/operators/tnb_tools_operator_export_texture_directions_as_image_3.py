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


def global_get_barycentric(p, p0, p1, p2):
    v0 = np.asarray(p1) - np.asarray(p0)
    v1 = np.asarray(p2) - np.asarray(p0)
    v2 = np.asarray(p) - np.asarray(p0)
    d00 = np.dot(v0, v0);
    d01 = np.dot(v0, v1);
    d11 = np.dot(v1, v1);
    d20 = np.dot(v2, v0);
    d21 = np.dot(v2, v1);
    denom = d00 * d11 - d01 * d01;

    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0 - v - w;

    return np.asarray([u, v, w])
    

def global_get_direction(p_barycentric, face, circle_center_3d):
    face_verts = face.verts
    p_3d  = (p_barycentric[0]*np.asarray(face_verts[0].co)) 
    p_3d += (p_barycentric[1]*np.asarray(face_verts[1].co))
    p_3d += (p_barycentric[2]*np.asarray(face_verts[2].co))
    direction_at_p_uv = p_3d - circle_center_3d
    direction_at_p_uv /= np.linalg.norm(direction_at_p_uv)    

    return direction_at_p_uv


class MyTriangle:
    def __init__(self,
                 p0, p1, p2) -> None:
        self._p0 = p0
        self._p1 = p1
        self._p2 = p2

        self._bbox_bot_x = None
        self._bbox_bot_y = None
        self._bbox_top_x = None
        self._bbox_top_y = None

        self._bbox_bot_x, self._bbox_bot_y, self._bbox_top_x, self._bbox_top_y = self.set_bbox()


    def set_bbox(self):
        bot_x = np.min((self._p0[0], self._p1[0], self._p2[0]))
        bot_y = np.min((self._p0[1], self._p1[1], self._p2[1]))
        top_x = np.max((self._p0[0], self._p1[0], self._p2[0]))
        top_y = np.max((self._p0[1], self._p1[1], self._p2[1]))

        return bot_x, bot_y, top_x, top_y
    

    def sign_func(self, p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    

    def get_barycentric(self, p):
        v0 = np.asarray(self._p1) - np.asarray(self._p0)
        v1 = np.asarray(self._p2) - np.asarray(self._p0)
        v2 = np.asarray(p) - np.asarray(self._p0)
        d00 = np.dot(v0, v0);
        d01 = np.dot(v0, v1);
        d11 = np.dot(v1, v1);
        d20 = np.dot(v2, v0);
        d21 = np.dot(v2, v1);
        denom = d00 * d11 - d01 * d01;

        v = (d11 * d20 - d01 * d21) / denom;
        w = (d00 * d21 - d01 * d20) / denom;
        u = 1.0 - v - w;
    
        return np.asarray([u, v, w])


    def contains(self, pt):
        d1 = self.sign_func(pt, self._p0, self._p1)
        d2 = self.sign_func(pt, self._p1, self._p2)
        d3 = self.sign_func(pt, self._p2, self._p0)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)
    

    def draw(self, out_img, face, uv_layer, circle_center_3d):
        height, width, channels = out_img.shape
        for x in range(int(np.floor(self._bbox_bot_x)), int(np.ceil(self._bbox_top_x)+1)):
            for y in range(int(np.floor(self._bbox_bot_y)), int(np.ceil(self._bbox_top_y)+1)):
                p = np.asarray([x, y])
                if self.contains(p):
                    uv_in_image = list()
                    for loop in face.loops:
                        loop_uv = loop[uv_layer]
                        uv_in_image.append([int(loop_uv.uv[0] * width), int(loop_uv.uv[1] * width)])
                        
                    p_barycentric = global_get_barycentric(p, uv_in_image[0], uv_in_image[1], uv_in_image[2])
                    cur_direction = global_get_direction(p_barycentric, face, circle_center_3d)
                        
                    out_img[y, x] = ((cur_direction + 1) / 2.0) * 255.0


class TnbToolsOperatorExportTextureDirectionsAsImage3(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):
    bl_idname       = "tnb_tools.operator_export_texture_directions_as_image_3"
    bl_label        = "tnb_tools_export_texture_directions_as_image_3"
    bl_description  = "Exports the expected (3D) directions of the texture as an image"

    def get_vertex_groups_as_enum(self, context):
        obj = context.object

        available_vertex_groups = list()

        for i_vertex_group, curr_vertex_group in enumerate(obj.vertex_groups):
            curr_enum_identifier    = curr_vertex_group.name
            curr_enum_name          = curr_vertex_group.name
            curr_enum_desc          = ''
            curr_enum_number        = i_vertex_group

            available_vertex_groups.append((curr_enum_identifier, curr_enum_name, curr_enum_desc, curr_enum_number))

        return tuple(available_vertex_groups)
    

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

    available_vertex_groups: bpy.props.EnumProperty(
        items=get_vertex_groups_as_enum,
        name="vertex_groups",
        description="Select a vertex group corresponding to a geodesic distance"
    )


    @classmethod
    def poll(cls, context):
        single_object_selected = (len(bpy.context.selected_objects) == 1)
        
        if single_object_selected is False:
            return False
        
        selected_object_is_mesh = (bpy.context.selected_objects[0].type == 'MESH')

        return selected_object_is_mesh
    
    
    def execute(self, context):
        self.output_tnb_direction_image(context)

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


    def get_face_direction(self, cur_face, vertex_group):
        # Get the index (in the face) of the vertex with lesser weight
        v_id_min_weight = np.argmin([vertex_group.weight(cur_face.verts[0].index), vertex_group.weight(cur_face.verts[1].index), vertex_group.weight(cur_face.verts[2].index)])

        v_0 = cur_face.verts[v_id_min_weight].co
        w_0 = vertex_group.weight(cur_face.verts[v_id_min_weight].index)
        v_1 = cur_face.verts[(v_id_min_weight+1)%3].co
        w_1 = vertex_group.weight(cur_face.verts[(v_id_min_weight+1)%3].index)
        v_2 = cur_face.verts[(v_id_min_weight+2)%3].co
        w_2 = vertex_group.weight(cur_face.verts[(v_id_min_weight+2)%3].index)
        
        v_0_v_1 = v_1 - v_0
        v_0_v_2 = v_2 - v_0

        if w_1 == w_2:
            point_at_w_on_e_1 = v_1
            point_at_w_on_e_2 = v_2
        
        elif w_1 < w_2:
            # On cherche ou se trouve w_1 sur l'arete e_2 (v_0_v_2). C'est une interpolation lineaire
            factor = (w_1 - w_0) / (w_2 - w_0)
            point_at_w_on_e_1 = v_1
            point_at_w_on_e_2 = v_0 + factor * v_0_v_2
        
        else:
            factor = (w_2 - w_0) / (w_1 - w_0)
            point_at_w_on_e_1 = v_0 + factor * v_0_v_1
            point_at_w_on_e_2 = v_2
        
        isophote_vector     = (point_at_w_on_e_1 - point_at_w_on_e_2).normalized()
        geodesic_gradient   = ((cur_face.normal.normalized()).cross(isophote_vector)).normalized()

        return geodesic_gradient
    
        
    def output_tnb_direction_image(self, context):
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

        obj = context.object
        max_geodesic_distance = obj.data['max_geodesic_distance']
        vertex_group = obj.vertex_groups[self.available_vertex_groups]
        edit_object = bpy.context.edit_object
        mesh_edit = bmesh.from_edit_mesh(edit_object.data)
        uv_layer = mesh_edit.loops.layers.uv.verify()
        
        mesh_edit_rotation_layer    = mesh_edit.faces.layers.float.get('tnb_rotation_from_v0_v1')
        if mesh_edit_rotation_layer is None:
            mesh_edit_rotation_layer = mesh_edit.faces.layers.float.new('tnb_rotation_from_v0_v1')

        for cur_face in mesh_edit.faces:
            uv_in_image = list()
            for loop in cur_face.loops:
                loop_uv = loop[uv_layer]
                uv_in_image.append([int(loop_uv.uv[0] * self.image_size), int(loop_uv.uv[1] * self.image_size)])

            cur_triangle = MyTriangle(uv_in_image[0], uv_in_image[1], uv_in_image[2])

            # Compute approx geodesic gradient per face
            cur_face_direction = self.get_face_direction(cur_face, vertex_group)
            cur_face_direction_normalized = cur_face_direction / np.linalg.norm(cur_face_direction)

            # Get the index (in the face) of the vertex with lesser weight
            v_id_min_weight = np.argmin([vertex_group.weight(cur_face.verts[0].index), vertex_group.weight(cur_face.verts[1].index), vertex_group.weight(cur_face.verts[2].index)])
            v_0 = cur_face.verts[(v_id_min_weight+0)%3].co
            w_0 = vertex_group.weight(cur_face.verts[v_id_min_weight].index)
            v_1 = cur_face.verts[(v_id_min_weight+1)%3].co
            w_1 = vertex_group.weight(cur_face.verts[(v_id_min_weight+1)%3].index)
            v_2 = cur_face.verts[(v_id_min_weight+2)%3].co
            w_2 = vertex_group.weight(cur_face.verts[(v_id_min_weight+2)%3].index)

            # That this direction is valid at v0 which is at distance w0
            # Then, compute the center of the 3D circle
            circle_center = v_0

            cur_triangle.draw(output_image, cur_face, uv_layer, circle_center)
            
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
    bpy.utils.register_class(TnbToolsOperatorExportTextureDirectionsAsImage3)


def unregister():
    bpy.utils.unregister_class(TnbToolsOperatorExportTextureDirectionsAsImage3)


if __name__ == "__main__":
    register()

    # test call
    bpy.ops.tnb_tools.operator_export_texture_directions_as_image_3('INVOKE_DEFAULT')