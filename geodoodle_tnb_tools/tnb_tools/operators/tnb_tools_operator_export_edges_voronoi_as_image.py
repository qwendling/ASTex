import os
import cv2
import bmesh
import mathutils
import numpy as np
import bpy
import bpy_extras


class TnbToolsOperatorExportEdgesVoronoiAsImage(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):
    bl_idname       = "tnb_tools.operator_export_edges_voronoi_as_image"
    bl_label        = "tnb_tools_export_edges_voronoi_as_image"
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
    
    sample_step: bpy.props.FloatProperty(
        name="Sample step",
        description="Distance between consecutive samples along an edge",
        default=0.05,
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

            
    def save_edges_voronoi_as_image(self, context):
        try:        
            out_image = np.zeros((self.image_size, self.image_size, 3), np.float32)
            
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

        all_sample_pairs_in_image = list()
        all_longest_edges = list()
        all_shortest_edges = list()

        for cur_face in curr_edit_mesh.faces:
            cur_face_v_uv_in_image = list()
            for loop in cur_face.loops:
                loop_uv = loop[uv_layer]
                cur_face_v_uv_in_image.append( np.array([int(loop_uv.uv[0] * self.image_size), int(loop_uv.uv[1] * self.image_size)]) )

            for cur_edge in cur_face.edges:
                is_curr_edge_seam = False
                seamed_face_1 = cur_face
                seamed_face_2 = None

                for nei_face in cur_edge.link_faces:
                    if cur_face.index != nei_face.index:
                        nei_face_v_uv_in_image = list()
                        for loop in nei_face.loops:
                            loop_uv = loop[uv_layer]
                            nei_face_v_uv_in_image.append( np.array([int(loop_uv.uv[0] * self.image_size), int(loop_uv.uv[1] * self.image_size)]) )

                        n_verts_equal_uv = 0
                        for i_vert_cur_face in range(3):
                            for i_vert_nei_face in range(3):
                                if np.linalg.norm(cur_face_v_uv_in_image[i_vert_cur_face] - nei_face_v_uv_in_image[i_vert_nei_face]) < 1e-9:
                                    n_verts_equal_uv += 1

                        is_curr_edge_seam = (n_verts_equal_uv != 2)
                        seamed_face_2 = nei_face

                if is_curr_edge_seam is True:
                    verts_i_seam_face_1 = list()
                    verts_i_seam_face_2 = list()
                    for i_vert_seam_face_1 in range(3):
                        for i_vert_seam_face_2 in range(3):
                            if np.linalg.norm(seamed_face_1.verts[i_vert_seam_face_1].co - seamed_face_2.verts[i_vert_seam_face_2].co) < 1e-9:
                                verts_i_seam_face_1.append(i_vert_seam_face_1)
                                verts_i_seam_face_2.append(i_vert_seam_face_2)

                    v0_tc_seam_edge_face_1 = np.asarray(seamed_face_1.loops[verts_i_seam_face_1[0]][uv_layer].uv)
                    v1_tc_seam_edge_face_1 = np.asarray(seamed_face_1.loops[verts_i_seam_face_1[1]][uv_layer].uv)
                    v0_tc_seam_edge_face_2 = np.asarray(seamed_face_2.loops[verts_i_seam_face_2[0]][uv_layer].uv)
                    v1_tc_seam_edge_face_2 = np.asarray(seamed_face_2.loops[verts_i_seam_face_2[1]][uv_layer].uv)

                    v0_tc_seam_edge_face_1_in_image = (np.asarray(v0_tc_seam_edge_face_1) * self.image_size).astype(int)
                    v1_tc_seam_edge_face_1_in_image = (np.asarray(v1_tc_seam_edge_face_1) * self.image_size).astype(int)
                    v0_tc_seam_edge_face_2_in_image = (np.asarray(v0_tc_seam_edge_face_2) * self.image_size).astype(int)
                    v1_tc_seam_edge_face_2_in_image = (np.asarray(v1_tc_seam_edge_face_2) * self.image_size).astype(int)

                    edge_face_1 = [v0_tc_seam_edge_face_1, v1_tc_seam_edge_face_1]
                    edge_face_1_in_image = [v0_tc_seam_edge_face_1_in_image, v1_tc_seam_edge_face_1_in_image]
                    if v0_tc_seam_edge_face_1[0] > v1_tc_seam_edge_face_1[0]:
                        edge_face_1 = [v1_tc_seam_edge_face_1, v0_tc_seam_edge_face_1]
                        edge_face_1_in_image = [v1_tc_seam_edge_face_1_in_image, v0_tc_seam_edge_face_1_in_image]
                    norm_edge_1 = np.linalg.norm(edge_face_1[0] - edge_face_1[1])

                    edge_face_2 = [v0_tc_seam_edge_face_2, v1_tc_seam_edge_face_2]
                    edge_face_2_in_image = [v0_tc_seam_edge_face_2_in_image, v1_tc_seam_edge_face_2_in_image]
                    if v0_tc_seam_edge_face_2[0] > v1_tc_seam_edge_face_2[0]:
                        edge_face_2 = [v1_tc_seam_edge_face_2, v0_tc_seam_edge_face_2]
                        edge_face_2_in_image = [v1_tc_seam_edge_face_2_in_image, v0_tc_seam_edge_face_2_in_image]
                    norm_edge_2 = np.linalg.norm(edge_face_2[0] - edge_face_2[1])

                    if norm_edge_1 > norm_edge_2:
                        longest_edge = edge_face_1
                        norm_longest_edge = norm_edge_1
                        longest_edge_in_image = edge_face_1_in_image
                        shortest_edge = edge_face_2
                        norm_shortest_edge = norm_edge_2
                        shortest_edge_in_image = edge_face_2_in_image
                    else:
                        longest_edge = edge_face_2
                        norm_longest_edge = norm_edge_2
                        longest_edge_in_image = edge_face_2_in_image
                        shortest_edge = edge_face_1
                        norm_shortest_edge = norm_edge_1
                        shortest_edge_in_image = edge_face_1_in_image

                    all_longest_edges.append(longest_edge_in_image)
                    all_shortest_edges.append(shortest_edge_in_image)

                    # Compute number of sample points on longest edge
                    half_step = self.sample_step * 0.5
                    all_t = np.arange(half_step, norm_longest_edge, step=self.sample_step) / norm_longest_edge

                    samples_longest_edge = list()
                    samples_shortest_edge = list()
                    for cur_t in all_t:
                        cur_sample_longest = longest_edge[0]   + cur_t * (longest_edge[1] - longest_edge[0])
                        cur_sample_longest_in_image = np.array(cur_sample_longest*self.image_size).astype(int)
                        samples_longest_edge.append(cur_sample_longest)

                        cur_sample_shortest = shortest_edge[0] + cur_t * (shortest_edge[1] - shortest_edge[0])
                        samples_shortest_edge.append(cur_sample_shortest)
                        cur_sample_shortest_in_image = np.array(cur_sample_shortest*self.image_size).astype(int)
                        
                        rand_color = (len(all_sample_pairs_in_image), len(all_sample_pairs_in_image), len(all_sample_pairs_in_image))
                        all_sample_pairs_in_image.append([cur_sample_longest_in_image, cur_sample_shortest_in_image, rand_color])
            
        bound_voronoi = (0, 0, self.image_size, self.image_size)
        voronoi_subdiv = cv2.Subdiv2D(bound_voronoi)
        for cur_sample_pair_in_image in all_sample_pairs_in_image:
            voronoi_subdiv.insert( (int(cur_sample_pair_in_image[0][0]), int(cur_sample_pair_in_image[0][1])) )
            voronoi_subdiv.insert( (int(cur_sample_pair_in_image[1][0]), int(cur_sample_pair_in_image[1][1])) )

        (voronoi_facets, voronoi_centers) = voronoi_subdiv.getVoronoiFacetList([])
        for cur_voronoi_facet, cur_voronoi_center in zip(voronoi_facets, voronoi_centers):
            if len(cur_voronoi_facet) != 0:
                pair_color = (0, 0, 0)
                for sample_longest_in_image, sample_shortest_in_image, color in all_sample_pairs_in_image:
                    if np.linalg.norm(sample_longest_in_image - cur_voronoi_center) < 1e-9 or np.linalg.norm(sample_shortest_in_image - cur_voronoi_center) < 1e-9:
                        pair_color = color
                cv2.fillPoly(out_image, pts=[np.array(cur_voronoi_facet).astype(int)], color=pair_color)

        for i_row in range(self.image_size):
            for i_col in range(self.image_size):
                cur_pix = np.asarray([i_col, i_row]).astype(float)
                cur_pix_color = out_image[i_row, i_col] # stores the index of the voronoi cell

                sample_longest_in_image, sample_shortest_in_image, color = all_sample_pairs_in_image[int(cur_pix_color[0])]
                dist_to_sample_longest = np.linalg.norm(cur_pix - sample_longest_in_image)
                dist_to_sample_shortest = np.linalg.norm(cur_pix - sample_shortest_in_image)

                if dist_to_sample_longest < dist_to_sample_shortest:
                    out_direction = sample_longest_in_image - cur_pix
                else:
                    out_direction = sample_shortest_in_image - cur_pix
                    
                out_direction_norm = np.linalg.norm(out_direction)
                if out_direction_norm != 0.0:
                    out_direction /= self.image_size
                else:
                    out_direction = np.asarray([0, 0, 0]).astype(float)
                
                out_image[i_row, i_col] = np.array([out_direction[0], out_direction[1], cur_pix_color[0]]).astype(float)

        # for cur_edge in all_longest_edges:
        #     cv2.line(out_image, np.array(cur_edge[0]).astype(int), np.array(cur_edge[1]).astype(int), (0, 255, 0, 255), 3)
            
        # for cur_edge in all_shortest_edges:
        #     cv2.line(out_image, np.array(cur_edge[0]).astype(int), np.array(cur_edge[1]).astype(int), (0, 0, 0, 255), 3)

        # for cur_voronoi_center in voronoi_centers:
        #     cv2.circle(out_image,
        #                (cur_voronoi_center).astype(int),
        #                radius=3,
        #                color=(100.1, 100.5, 100.2),
        #                thickness=5)
                    
        if self.flip_image_vertically is True:
            out_image = cv2.flip(out_image, 0)

        assert out_image.dtype == np.float32
        cv2.imwrite(self.filepath, out_image)


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
        self.save_edges_voronoi_as_image(context)

        return {'FINISHED'}
    

def register():
    bpy.utils.register_class(TnbToolsOperatorExportEdgesVoronoiAsImage)


def unregister():
    bpy.utils.unregister_class(TnbToolsOperatorExportEdgesVoronoiAsImage)


if __name__ == "__main__":
    register()

    # test call
    bpy.ops.tnb_tools.operator_export_edges_voronoi_as_image('INVOKE_DEFAULT')