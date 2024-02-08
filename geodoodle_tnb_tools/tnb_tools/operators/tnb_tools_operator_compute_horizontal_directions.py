import bpy
import bmesh
import mathutils
import numpy as np

class TnbToolsOperatorComputeHorizontalDirectionsAroundVertexOperator(bpy.types.Operator):
    bl_idname       = "tnb_tools.operator_h_dirs_around_vert"
    bl_label        = "tnb_tools_h_dirs_around_vert"
    bl_description  = "Horizontal directions around selected vertex"


    @classmethod
    def poll(cls, context):
        if context.mode == 'EDIT_MESH':
            curr_edit_mesh = bmesh.from_edit_mesh(context.object.data)
            selected_vertices = [v for v in curr_edit_mesh.verts if v.select]

            return len(selected_vertices) == 1
        
        else:
            return False
    

    def generate_horizontal_tnb_direction_around_vertex(self, context):
        curr_edit_mesh                  = bmesh.from_edit_mesh(context.object.data)
        curr_edit_mesh_rotation_layer   = curr_edit_mesh.faces.layers.float.get('tnb_rotation_from_v0_v1')

        if curr_edit_mesh_rotation_layer is None:
            curr_edit_mesh_rotation_layer = curr_edit_mesh.faces.layers.float.new('tnb_rotation_from_v0_v1')

        selected_vertices = [v for v in curr_edit_mesh.verts if v.select]

        selected_vertex = selected_vertices[0].co

        for curr_face in curr_edit_mesh.faces:
            v_0             = curr_face.verts[0].co
            v_1             = curr_face.verts[1].co
            reference_dir   = (v_1 - v_0).normalized()

            curr_face_normal = curr_face.normal.normalized()

            curr_face_center        = curr_face.calc_center_median()
            orientation_vector      = mathutils.Matrix.Rotation(np.deg2rad(90.0), 3, 'Z') @ (curr_face_center - selected_vertex)
            orientation_vector[2]   = 0
            orientation_vector.normalize()

            tnb_horizontal_direction = curr_face_normal.cross( mathutils.Vector((0.0, 0.0, 1.0))).normalized()

            if tnb_horizontal_direction.length == 0:
                tnb_horizontal_direction = orientation_vector.normalized()

            if tnb_horizontal_direction.dot(orientation_vector) < 0:
                tnb_horizontal_direction = tnb_horizontal_direction * -1.0

            dot_prod        = max(-1.0, min(1.0, tnb_horizontal_direction.dot(reference_dir)))
            rotation_angle  = np.arccos(dot_prod)

            if tnb_horizontal_direction.cross(reference_dir).dot(curr_face_normal) < 0:
                rotation_angle = rotation_angle * -1.0

            curr_edit_mesh.faces[curr_face.index][curr_edit_mesh_rotation_layer] = rotation_angle

        bmesh.update_edit_mesh(bpy.context.object.data)


    def execute(self, context):
        self.generate_horizontal_tnb_direction_around_vertex(context)

        return {'FINISHED'}
    

def register():
    bpy.utils.register_class(TnbToolsOperatorComputeHorizontalDirectionsAroundVertexOperator)


def unregister():
    bpy.utils.unregister_class(TnbToolsOperatorComputeHorizontalDirectionsAroundVertexOperator)


if __name__ == "__main__":
    register()