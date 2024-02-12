import os
import cv2
import bmesh
import mathutils
import numpy as np
import bpy
import bpy_extras
import random


class MyColoredTriangle:
    def __init__(self,
                 p0, color0,
                 p1, color1,
                 p2, color2) -> None:
        self._p0 = p0
        self._p1 = p1
        self._p2 = p2
        
        self._c0 = color0
        self._c1 = color1
        self._c2 = color2

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
    

    def draw(self, out_img):
        for x in range(int(np.floor(self._bbox_bot_x)), int(np.ceil(self._bbox_top_x)+1)):
            for y in range(int(np.floor(self._bbox_bot_y)), int(np.ceil(self._bbox_top_y)+1)):
                p = np.asarray([x, y])
                if self.contains(p):
                    p_bary = self.get_barycentric(p)
                    out_img[y, x] = ((p_bary[0]*np.asarray(self._c0).astype(float)) + (p_bary[1]*np.asarray(self._c1).astype(float)) + (p_bary[2]*np.asarray(self._c2).astype(float))).astype(int)



class TnbToolsOperatorExportSmoothedValuesOnFaces(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):
    bl_idname       = "tnb_tools.operator_export_smoothed_values_on_faces"
    bl_label        = "tnb_tools_export_smoothed_values_on_faces"
    bl_description  = "Test"

    # ExportHelper mixin class uses this
    filename_ext = ".png"

    filter_glob: bpy.props.StringProperty(
        default="*.png",
        options={'HIDDEN'},
        maxlen=1000,  # Max internal buffer length, longer would be clamped.
    )

    image_size: bpy.props.IntProperty(
        name="Image width",
        description="Width of the output image",
        default=1000,
    )

    smooth_radius: bpy.props.FloatProperty(
        name="Smooth radius",
        description="Radius for smoothing values",
        default=0.005,
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
        self.save_edges_voronoi_as_image(context)

        return {'FINISHED'}


    def create_fake_layer(self, cur_edit_mesh):
        # Create a face layer
        face_value_layer = cur_edit_mesh.faces.layers.float.get('random_test_face')
        if face_value_layer is None:
            face_value_layer = cur_edit_mesh.faces.layers.float.new('random_test_face')

        # Fill it with random integers between 0 and 255
        for cur_face in cur_edit_mesh.faces:
            cur_face[face_value_layer] = random.randint(0, 255)

        # Create a vertex layer
        vert_value_layer = cur_edit_mesh.verts.layers.float.get('random_test_vert')
        if vert_value_layer is None:
            vert_value_layer = cur_edit_mesh.verts.layers.float.new('random_test_vert')

        # Set the value of each vertex to the mean of the neighbour faces
        for cur_vert in cur_edit_mesh.verts:
            cur_mean_val = 0.0
            cur_mean_cpt = 0
            for cur_nei_face in cur_vert.link_faces:
                cur_mean_cpt += 1
                cur_mean_val += cur_edit_mesh.faces[cur_nei_face.index][face_value_layer]
            
            cur_mean_val /= cur_mean_cpt

            cur_vert[vert_value_layer] = cur_mean_val

        return face_value_layer, vert_value_layer



    def bbox(self, p0, p1, p2):
        bot_x = np.min((p0[0], p1[0], p2[0]))
        bot_y = np.min((p0[1], p1[1], p2[1]))
        top_x = np.max((p0[0], p1[0], p2[0]))
        top_y = np.max((p0[1], p1[1], p2[1]))

        return bot_x, bot_y, top_x, top_y
    
    def toto_sign(self, p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    def toto_barycentric(self, p, a, b, c):
        v0 = np.asarray(b) - np.asarray(a)
        v1 = np.asarray(c) - np.asarray(a)
        v2 = np.asarray(p) - np.asarray(a)
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


    def toto_is_point_in_triangle(self, pt, v1, v2, v3):
        d1 = self.toto_sign(pt, v1, v2)
        d2 = self.toto_sign(pt, v2, v3)
        d3 = self.toto_sign(pt, v3, v1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def draw_triangle(self, p0, p1, p2, c0, c1, c2, out_img):
        bot_x, bot_y, top_x, top_y = self.bbox(p0, p1, p2)
        for x in range(int(np.floor(bot_x)), int(np.ceil(top_x)+1)):
            for y in range(int(np.floor(bot_y)), int(np.ceil(top_y)+1)):
                p = np.asarray([x, y])
                if self.toto_is_point_in_triangle(p, p0, p1, p2):
                    p_bary = self.toto_barycentric(p, p0, p1, p2)
                    # print(f'c0 {c0}')
                    # print(f'c1 {c1}')
                    # print(f'c2 {c2}')
                    # print(f'p_bary {p_bary[0]}')
                    # print(f'p_bary {p_bary[1]}')
                    # print(f'p_bary {p_bary[2]}')
                    # out_img[y, x] = (255, 0, 255)
                    # print(f'x {x} y {y} out_img[y, x] {out_img[y, x]}')
                    out_img[y, x] = ((p_bary[0]*np.asarray(c0).astype(float)) + (p_bary[1]*np.asarray(c1).astype(float)) + (p_bary[2]*np.asarray(c2).astype(float))).astype(int)


    def get_small_triangle(self, face, uv_layer):
        cur_face_v_uv_in_image = list()
        cur_face_v_index = list()
        for loop in face.loops:
            loop_uv = loop[uv_layer]
            cur_face_v_uv_in_image.append( np.array([int(loop_uv.uv[0] * self.image_size), int(loop_uv.uv[1] * self.image_size)]) )
        
        for cur_vert in face.verts:
            cur_face_v_index.append(cur_vert.index)

        # Compute incenter of triangle
        p0_in_image = np.asarray(cur_face_v_uv_in_image[0])
        p1_in_image = np.asarray(cur_face_v_uv_in_image[1])
        p2_in_image = np.asarray(cur_face_v_uv_in_image[2])
        p0p1_norm = np.linalg.norm(p0_in_image - p1_in_image)
        p0p2_norm = np.linalg.norm(p0_in_image - p2_in_image)
        p1p2_norm = np.linalg.norm(p1_in_image - p2_in_image)
        perimeter = (p0p1_norm + p0p2_norm + p1p2_norm)
        incenter_barycentric = np.array([p1p2_norm, p0p2_norm, p0p1_norm])
        incenter_barycentric /= perimeter
        incenter_uv_in_image = (incenter_barycentric[0] * p0_in_image) + (incenter_barycentric[1] * p1_in_image) + (incenter_barycentric[2] * p2_in_image)
        semi_perimeter       = 0.5 * perimeter
        incircle_radius      = np.sqrt( ( (semi_perimeter - p0p1_norm)*(semi_perimeter - p0p2_norm)*(semi_perimeter - p1p2_norm) ) / semi_perimeter )

        # Scale triangle around incenter
        if incircle_radius > self.smooth_radius_in_image:
            scale_factor = (incircle_radius - self.smooth_radius_in_image) / incircle_radius
            p0_scaled_in_mage = incenter_uv_in_image + ((p0_in_image - incenter_uv_in_image) * scale_factor)
            p1_scaled_in_mage = incenter_uv_in_image + ((p1_in_image - incenter_uv_in_image) * scale_factor)
            p2_scaled_in_mage = incenter_uv_in_image + ((p2_in_image - incenter_uv_in_image) * scale_factor)

            return cur_face_v_index[0], cur_face_v_index[1], cur_face_v_index[2], p0_in_image, p1_in_image, p2_in_image, p0_scaled_in_mage, p1_scaled_in_mage, p2_scaled_in_mage
        
        else:
            return cur_face_v_index[0], cur_face_v_index[1], cur_face_v_index[2], p0_in_image, p1_in_image, p2_in_image, incenter_uv_in_image, incenter_uv_in_image, incenter_uv_in_image


    def compute_faces_triangles_layers(self, cur_edit_mesh, uv_layer):
        face_small_tri_x_layer = cur_edit_mesh.faces.layers.float_vector.get('small_triangles_x')
        if face_small_tri_x_layer is None:
            face_small_tri_x_layer = cur_edit_mesh.faces.layers.float_vector.new('small_triangles_x')

        face_small_tri_y_layer = cur_edit_mesh.faces.layers.float_vector.get('small_triangles_y')
        if face_small_tri_y_layer is None:
            face_small_tri_y_layer = cur_edit_mesh.faces.layers.float_vector.new('small_triangles_y')

        face_large_tri_x_layer = cur_edit_mesh.faces.layers.float_vector.get('large_triangles_x')
        if face_large_tri_x_layer is None:
            face_large_tri_x_layer = cur_edit_mesh.faces.layers.float_vector.new('large_triangles_x')

        face_large_tri_y_layer = cur_edit_mesh.faces.layers.float_vector.get('large_triangles_y')
        if face_large_tri_y_layer is None:
            face_large_tri_y_layer = cur_edit_mesh.faces.layers.float_vector.new('large_triangles_y')

        for cur_face in cur_edit_mesh.faces:
            v0_id,             v1_id,             v2_id, \
            p0_in_image,       p1_in_image,       p2_in_image, \
            p0_small_in_image, p1_small_in_image, p2_small_in_image = self.get_small_triangle(cur_face, uv_layer)

            cur_face[face_small_tri_x_layer][0] = p0_small_in_image[0]
            cur_face[face_small_tri_x_layer][1] = p1_small_in_image[0]
            cur_face[face_small_tri_x_layer][2] = p2_small_in_image[0]

            cur_face[face_small_tri_y_layer][0] = p0_small_in_image[1]
            cur_face[face_small_tri_y_layer][1] = p1_small_in_image[1]
            cur_face[face_small_tri_y_layer][2] = p2_small_in_image[1]

            cur_face[face_large_tri_x_layer][0] = p0_in_image[0]
            cur_face[face_large_tri_x_layer][1] = p1_in_image[0]
            cur_face[face_large_tri_x_layer][2] = p2_in_image[0]

            cur_face[face_large_tri_y_layer][0] = p0_in_image[1]
            cur_face[face_large_tri_y_layer][1] = p1_in_image[1]
            cur_face[face_large_tri_y_layer][2] = p2_in_image[1]

        return face_large_tri_x_layer, face_large_tri_y_layer, face_small_tri_x_layer, face_small_tri_y_layer


    def save_edges_voronoi_as_image(self, context):
        try:        
            out_image = np.zeros((self.image_size, self.image_size, 3), np.float32)
            
        except:
            self.report({'ERROR'}, f'Could not save image with auto size (image too big? Width: {self.image_size}, height: {self.image_size})')
            return
        
        if bpy.context.mode == 'EDIT_MESH':
            cur_edit_mesh = bmesh.from_edit_mesh(bpy.context.object.data)
        elif bpy.context.mode == 'OBJECT' and bpy.context.object.type == 'MESH':
            cur_edit_mesh = bmesh.new()
            cur_edit_mesh.from_mesh(bpy.context.object.data)
            cur_edit_mesh.faces.ensure_lookup_table()
        else:
            return {'CANCELLED'}
        
        uv_layer = cur_edit_mesh.loops.layers.uv.verify()
        self.smooth_radius_in_image = self.smooth_radius * self.image_size
        
        face_value_layer, vert_value_layer = self.create_fake_layer(cur_edit_mesh)
        face_large_tri_x_layer, face_large_tri_y_layer, face_small_tri_x_layer, face_small_tri_y_layer = self.compute_faces_triangles_layers(cur_edit_mesh, uv_layer)
        
        all_triangles = list()

        for cur_face in cur_edit_mesh.faces:
            small_triangle_points = list()
            small_triangle_colors = list()
            for i_vert, cur_vert in enumerate(cur_face.verts):
                cur_small_triangle_point = np.asarray([cur_face[face_small_tri_x_layer][i_vert], cur_face[face_small_tri_y_layer][i_vert]])
                small_triangle_points.append(cur_small_triangle_point)

                cur_small_triangle_color = cur_face[face_value_layer]
                small_triangle_colors.append(cur_small_triangle_color)

            all_triangles.append( MyColoredTriangle(small_triangle_points[0], small_triangle_colors[0],
                                                    small_triangle_points[1], small_triangle_colors[1],
                                                    small_triangle_points[2], small_triangle_colors[2]) )
            
            # For each vertex of each edge of the current face
            for cur_edge in cur_face.edges:
                cur_edge_verts     = [cur_edge.verts[0], cur_edge.verts[1]]
                cur_edge_verts_opp = [cur_edge.verts[1], cur_edge.verts[0]]

                for cur_vert_in_cur_edge, cur_vert_opp_in_cur_edge in zip(cur_edge_verts, cur_edge_verts_opp):
                    # Find local index of vertex in current face
                    cur_vert_local_id_in_cur_face = 10
                    cur_vert_opp_local_id_in_cur_face = 10
                    for i_cur_vert_in_face, cur_vert_in_face in enumerate(cur_face.verts):
                        if cur_vert_in_cur_edge.index == cur_vert_in_face.index:
                            cur_vert_local_id_in_cur_face = i_cur_vert_in_face
                        if cur_vert_opp_in_cur_edge.index == cur_vert_in_face.index:
                            cur_vert_opp_local_id_in_cur_face = i_cur_vert_in_face

                    # Find local index of its counterpart in the neighbour face
                    nei_face = None
                    cur_vert_local_id_in_nei_face = 10
                    cur_vert_opp_local_id_in_nei_face = 10
                    for cur_nei_face in cur_edge.link_faces:
                        if cur_nei_face.index != cur_face.index:
                            nei_face = cur_nei_face
                            for i_cur_vert_in_nei_face, cur_vert_in_nei_face in enumerate(cur_nei_face.verts):
                                if cur_vert_in_nei_face.index == cur_vert_in_cur_edge.index:
                                    cur_vert_local_id_in_nei_face = i_cur_vert_in_nei_face
                                if cur_vert_in_nei_face.index == cur_vert_opp_in_cur_edge.index:
                                    cur_vert_opp_local_id_in_nei_face = i_cur_vert_in_nei_face

                    is_edge_border = nei_face is None
                    if not is_edge_border:
                        cur_face_v_uv = list()
                        nei_face_v_uv = list()
                        for cur_loop, nei_loop in zip(cur_face.loops, nei_face.loops):
                            cur_loop_uv = cur_loop[uv_layer]
                            cur_face_v_uv.append( np.array(cur_loop_uv.uv) )

                            nei_loop_uv = nei_loop[uv_layer]
                            nei_face_v_uv.append( np.array(nei_loop_uv.uv) )

                        is_seam = False
                        if np.linalg.norm(cur_face_v_uv[cur_vert_local_id_in_cur_face] - nei_face_v_uv[cur_vert_local_id_in_nei_face]) > 1e-11:
                            is_seam = True
                            
                        if not is_seam:
                            # Create a triangle with the
                            p0 = np.asarray([cur_face[face_large_tri_x_layer][cur_vert_local_id_in_cur_face], cur_face[face_large_tri_y_layer][cur_vert_local_id_in_cur_face]])
                            c0 = cur_vert_in_cur_edge[vert_value_layer]
                            p1 = np.asarray([cur_face[face_small_tri_x_layer][cur_vert_local_id_in_cur_face], cur_face[face_small_tri_y_layer][cur_vert_local_id_in_cur_face]])
                            c1 = cur_face[face_value_layer]
                            p2 = np.asarray([nei_face[face_small_tri_x_layer][cur_vert_local_id_in_nei_face], nei_face[face_small_tri_y_layer][cur_vert_local_id_in_nei_face]])
                            c2 = nei_face[face_value_layer]
                            all_triangles.append( MyColoredTriangle(p0, c0, p1, c1, p2, c2) )

                            # Create a triangle with the
                            p3 = np.asarray([cur_face[face_small_tri_x_layer][cur_vert_opp_local_id_in_cur_face], cur_face[face_small_tri_y_layer][cur_vert_opp_local_id_in_cur_face]])
                            c3 = cur_face[face_value_layer]
                            all_triangles.append( MyColoredTriangle(p1, c1, p2, c2, p3, c3) )

                            # Create a triangle with the
                            p4 = np.asarray([nei_face[face_small_tri_x_layer][cur_vert_opp_local_id_in_nei_face], nei_face[face_small_tri_y_layer][cur_vert_opp_local_id_in_nei_face]])
                            c4 = nei_face[face_value_layer]
                            all_triangles.append( MyColoredTriangle(p2, c2, p3, c3, p4, c4) )

                        else:
                            # Create a triangle with the
                            p0 = np.asarray([cur_face[face_large_tri_x_layer][cur_vert_local_id_in_cur_face], cur_face[face_large_tri_y_layer][cur_vert_local_id_in_cur_face]])
                            c0 = cur_vert_in_cur_edge[vert_value_layer]
                            p1 = np.asarray([cur_face[face_small_tri_x_layer][cur_vert_local_id_in_cur_face], cur_face[face_small_tri_y_layer][cur_vert_local_id_in_cur_face]])
                            c1 = cur_face[face_value_layer]
                            p3 = np.asarray([cur_face[face_small_tri_x_layer][cur_vert_opp_local_id_in_cur_face], cur_face[face_small_tri_y_layer][cur_vert_opp_local_id_in_cur_face]])
                            c3 = cur_face[face_value_layer]
                            all_triangles.append( MyColoredTriangle(p0, c0, p1, c1, p3, c3) )

                            p5 = np.asarray([cur_face[face_large_tri_x_layer][cur_vert_opp_local_id_in_cur_face], cur_face[face_large_tri_y_layer][cur_vert_opp_local_id_in_cur_face]])
                            c5 = cur_vert_opp_in_cur_edge[vert_value_layer]
                            all_triangles.append( MyColoredTriangle(p0, c0, p3, c3, p5, c5) )

                    else:
                        # edge is a border
                        p0 = np.asarray([cur_face[face_large_tri_x_layer][cur_vert_local_id_in_cur_face], cur_face[face_large_tri_y_layer][cur_vert_local_id_in_cur_face]])
                        c0 = cur_vert_in_cur_edge[vert_value_layer]
                        p1 = np.asarray([cur_face[face_small_tri_x_layer][cur_vert_local_id_in_cur_face], cur_face[face_small_tri_y_layer][cur_vert_local_id_in_cur_face]])
                        c1 = cur_face[face_value_layer]
                        p3 = np.asarray([cur_face[face_small_tri_x_layer][cur_vert_opp_local_id_in_cur_face], cur_face[face_small_tri_y_layer][cur_vert_opp_local_id_in_cur_face]])
                        c3 = cur_face[face_value_layer]
                        all_triangles.append( MyColoredTriangle(p0, c0, p1, c1, p3, c3) )

                        p5 = np.asarray([cur_face[face_large_tri_x_layer][cur_vert_opp_local_id_in_cur_face], cur_face[face_large_tri_y_layer][cur_vert_opp_local_id_in_cur_face]])
                        c5 = cur_vert_opp_in_cur_edge[vert_value_layer]
                        all_triangles.append( MyColoredTriangle(p0, c0, p3, c3, p5, c5) )


        for cur_triangle in all_triangles:
            cur_triangle.draw(out_image)

        if self.flip_image_vertically is True:
            out_image = cv2.flip(out_image, 0)

        assert out_image.dtype == np.float32
        cv2.imwrite(self.filepath, out_image)


    def execute(self, context):
        self.save_edges_voronoi_as_image(context)

        return {'FINISHED'}
    

def register():
    bpy.utils.register_class(TnbToolsOperatorExportSmoothedValuesOnFaces)


def unregister():
    bpy.utils.unregister_class(TnbToolsOperatorExportSmoothedValuesOnFaces)


if __name__ == "__main__":
    register()

    # test call
    bpy.ops.tnb_tools.operator_export_smoothed_values_on_faces('INVOKE_DEFAULT')