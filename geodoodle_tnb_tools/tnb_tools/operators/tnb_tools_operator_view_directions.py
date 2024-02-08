import bmesh
import mathutils
import bpy
import gpu
from   gpu_extras.batch import batch_for_shader


def draw_3d_segments(color, segments, line_width):
    shader  = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch   = batch_for_shader(shader, 'LINES', {"pos": segments})

    prev_line_width = gpu.state.line_width_get()

    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(True)
    gpu.state.line_width_set(line_width)
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.depth_mask_set(False)
    gpu.state.line_width_set(prev_line_width)


def compute_transfor_matrix(selected_face):
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


class TnbToolsOperatorViewDirections(bpy.types.Operator):
    """Draw 3d Operator"""
    bl_idname       = "tnb_tools.operator_draw_directions"
    bl_label        = "tnb_tools_view_tnb_directions"
    bl_description  = "View the TnB directions"

    _mesh_object            = None
    _edit_mesh              = None
    _rotation_layer         = None
    _handle_3d              = None
    _3d_segments_beg_end    = None


    line_width: bpy.props.FloatProperty(
        name="Line width",
        description="Line width of drawn directions",
        default=5,
        min=1.0,
        max=20.0,
        step=1.0
    )


    @classmethod
    def poll(cls, context):
        return (len(context.selected_objects) == 1) and (context.selected_objects[0].type == 'MESH')
    
    
    def update_3d_segments(self):
        self._3d_segments_beg_end = list()

        for curr_face in self._edit_mesh.faces:
            curr_face_center    = curr_face.calc_center_median()
            curr_rotation_angle = curr_face[self._rotation_layer]

            curr_rotation_matrix    = mathutils.Matrix.Rotation(curr_rotation_angle, 3, (0.0, 0.0, 1.0))
            curr_tnb_direction_3d   = compute_transfor_matrix(curr_face) @ curr_rotation_matrix @ mathutils.Vector((1.0, 0.0, 0.0))


            v_0     = curr_face.verts[0].co
            v_1     = curr_face.verts[1].co
            v_2     = curr_face.verts[2].co
            
            v_0_v_1 = v_1 - v_0
            v_0_v_2 = v_2 - v_0
            v_1_v_2 = v_2 - v_1
            
            curr_perimeter_uv       = v_0_v_1.length + v_0_v_2.length + v_1_v_2.length
            curr_half_perimeter     = curr_perimeter_uv * 0.5
            curr_area_uv            = v_0_v_1.cross(v_0_v_2).length * 0.5
            curr_incircle_radius    = curr_area_uv / curr_half_perimeter

            segment_beg_local = curr_face_center
            segment_beg_local_homogeneous = mathutils.Vector((segment_beg_local[0], segment_beg_local[1], segment_beg_local[2], 1.0))
            segment_beg_world_homogeneous = self._mesh_object.matrix_world @ segment_beg_local_homogeneous

            segment_end_local = curr_face_center + (curr_tnb_direction_3d.normalized() * curr_incircle_radius)
            segment_beg_local_homogeneous = mathutils.Vector((segment_end_local[0], segment_end_local[1], segment_end_local[2], 1.0))
            segment_end_world_homogeneous = self._mesh_object.matrix_world @ segment_beg_local_homogeneous

            self._3d_segments_beg_end.append((segment_beg_world_homogeneous[0], segment_beg_world_homogeneous[1], segment_beg_world_homogeneous[2]))
            self._3d_segments_beg_end.append((segment_end_world_homogeneous[0], segment_end_world_homogeneous[1], segment_end_world_homogeneous[2]))

        return


    def reset_members(self):
        if self._handle_3d != None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle_3d, 'WINDOW')

        self._mesh_object           = None
        self._edit_mesh             = None
        self._rotation_layer        = None
        self._handle_3d             = None
        self._3d_segments_beg_end   = None


    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)


    def execute(self, context):
        if context.area.type == 'VIEW_3D':
            self._mesh_object = context.object

            if context.mode == 'EDIT_MESH':
                self._edit_mesh = bmesh.from_edit_mesh(bpy.context.edit_object.data)
            elif context.mode == 'OBJECT' and bpy.context.object.type == 'MESH':
                self._edit_mesh = bmesh.new()
                self._edit_mesh.from_mesh(bpy.context.object.data)
            else:
                self.reset_members()
                return {'CANCELLED'}
            
            self._rotation_layer = self._edit_mesh.faces.layers.float.get('tnb_rotation_from_v0_v1')
            if self._rotation_layer is None:
                self._rotation_layer = self._edit_mesh.faces.layers.float.new('tnb_rotation_from_v0_v1')

            self.update_3d_segments()

            curr_color      = (1.0, 0.3, 0.0, 1.0)
            curr_line_width = self.line_width
            args            = (curr_color, self._3d_segments_beg_end, curr_line_width)

            # draw in view space with 'POST_VIEW' and 'PRE_VIEW'
            self._handle_3d = bpy.types.SpaceView3D.draw_handler_add(draw_3d_segments, args, 'WINDOW', 'POST_VIEW')

            context.window_manager.modal_handler_add(self)
            
            return {'RUNNING_MODAL'}
        
        else:
            self.reset_members()
            return {'CANCELLED'}


    def modal(self, context, event):
        context.area.tag_redraw()

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.reset_members()
            return {'CANCELLED'}
        
        return {'PASS_THROUGH'}


def register():
    bpy.utils.register_class(TnbToolsOperatorViewDirections)


def unregister():
    bpy.utils.unregister_class(TnbToolsOperatorViewDirections)


if __name__ == "__main__":
    register()