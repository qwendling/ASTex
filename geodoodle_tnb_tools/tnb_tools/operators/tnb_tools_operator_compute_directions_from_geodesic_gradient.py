import bpy
import bmesh
import numpy as np


class TnbToolsOperatorComputeDirectionsFromGeodesicGradient(bpy.types.Operator):
    bl_idname       = "tnb_tools.operator_dirs_from_geodesic"
    bl_label        = "tnb_tools_dirs_from_geodesic"
    bl_description  = "Directions from geodesic gradient"

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


    available_vertex_groups: bpy.props.EnumProperty(
        items=get_vertex_groups_as_enum,
        name="vertex_groups",
        description="Select a vertex group corresponding to a geodesic distance"
    )


    @classmethod
    def poll(cls, context):
        return (len(context.selected_objects) == 1) and (context.selected_objects[0].type == 'MESH')
    

    def generate_tnb_direction_from_geodesic_gradient(self, context):
        obj                         = context.object
        vertex_group                = obj.vertex_groups[self.available_vertex_groups]
        edit_object                 = bpy.context.edit_object
        mesh_edit                   = bmesh.from_edit_mesh(edit_object.data)
        mesh_edit_rotation_layer    = mesh_edit.faces.layers.float.get('tnb_rotation_from_v0_v1')

        if mesh_edit_rotation_layer is None:
            mesh_edit_rotation_layer = mesh_edit.faces.layers.float.new('tnb_rotation_from_v0_v1')

        selected_faces = [f for f in mesh_edit.faces if f.select]

        for curr_face in selected_faces:
            # Get the index (in the face) of the vertex with lesser weight
            v_id_min_weight = np.argmin([vertex_group.weight(curr_face.verts[0].index), vertex_group.weight(curr_face.verts[1].index), vertex_group.weight(curr_face.verts[2].index)])

            v_0 = curr_face.verts[v_id_min_weight].co
            w_0 = vertex_group.weight(curr_face.verts[v_id_min_weight].index)
            v_1 = curr_face.verts[(v_id_min_weight+1)%3].co
            w_1 = vertex_group.weight(curr_face.verts[(v_id_min_weight+1)%3].index)
            v_2 = curr_face.verts[(v_id_min_weight+2)%3].co
            w_2 = vertex_group.weight(curr_face.verts[(v_id_min_weight+2)%3].index)
            
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
            geodesic_gradient   = ((curr_face.normal.normalized()).cross(isophote_vector)).normalized()
            local_dir           = (curr_face.verts[1].co - curr_face.verts[0].co).normalized()
            rotation_angle      = geodesic_gradient.angle(local_dir)

            if geodesic_gradient.cross(local_dir).normalized().dot(curr_face.normal.normalized()) > 0:
                mesh_edit.faces[curr_face.index][mesh_edit_rotation_layer] = rotation_angle
            else:
                mesh_edit.faces[curr_face.index][mesh_edit_rotation_layer] = -rotation_angle

        bmesh.update_edit_mesh(bpy.context.object.data)


    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)


    def execute(self, context):
        self.generate_tnb_direction_from_geodesic_gradient(context)

        return {'FINISHED'}
    

def register():
    bpy.utils.register_class(TnbToolsOperatorComputeDirectionsFromGeodesicGradient)


def unregister():
    bpy.utils.unregister_class(TnbToolsOperatorComputeDirectionsFromGeodesicGradient)


if __name__ == "__main__":
    register()