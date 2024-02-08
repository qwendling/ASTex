import bpy
import bmesh
import mathutils
import numpy as np
import random
from bpy.types import (
    Gizmo,
    GizmoGroup,
)

# Triangles of a simple 2D arrow
custom_shape_verts = ( ( 0.0, -1.0,  0.0), (5.0,  0.0,  0.0), (0.0,  0.0,  0.0),
                       ( 0.0,  0.0,  0.0), (5.0,  0.0,  0.0), (0.0,  1.0,  0.0), )


def set_rotation_offset(value):
    if TnBDirectionArrowGizmoGroup.has_single_selected_face():
        querry_face = TnBDirectionArrowGizmoGroup.selected_faces[0]
        TnBDirectionArrowGizmoGroup.mesh_edit.faces[querry_face.index][TnBDirectionArrowGizmoGroup.mesh_face_layer] = value

        TnBDirectionArrowGizmoGroup.save_edit_mesh()
    
    else:
        return None


def get_rotation_offset():
    if TnBDirectionArrowGizmoGroup.has_single_selected_face():
        querry_face = TnBDirectionArrowGizmoGroup.selected_faces[0]
        return TnBDirectionArrowGizmoGroup.mesh_edit.faces[querry_face.index][TnBDirectionArrowGizmoGroup.mesh_face_layer]

    else:
        return None


class TnBDirectionArrowGizmo(Gizmo):
    bl_idname = "VIEW3D_edit_mesh_TnBDirectionArrowGizmo"
    bl_target_properties = (
        {"id": "rotation_from_v0_v1", "type": 'FLOAT', "array_length": 1},
    )

    __slots__ = (
        "custom_shape",
        "init_mouse_y",
        "saved_rotation",
    )


    def draw(self, context):
        self.draw_custom_shape(self.custom_shape)


    def draw_select(self, context, select_id):
        self.draw_custom_shape(self.custom_shape, select_id=select_id)


    def setup(self):
        if not hasattr(self, "custom_shape"):
            self.custom_shape = self.new_custom_shape('TRIS', custom_shape_verts)


    def invoke(self, context, event):
        self.init_mouse_y   = event.mouse_y
        self.saved_rotation = self.target_get_value('rotation_from_v0_v1')[0]
        
        return {'RUNNING_MODAL'}
    

    def exit(self, context, cancel):
        context.area.header_text_set(None)
        TnBDirectionArrowGizmoGroup.save_edit_mesh()


    def modal(self, context, event, tweak):
        delta = (event.mouse_y - self.init_mouse_y) / 200

        if 'SNAP' in tweak:
            delta = round(delta*10) / 100

        if 'PRECISE' in tweak:
            delta /= 10.0

        new_value_rotation_from_v0_v1 = self.saved_rotation + delta

        if 'SNAP' in tweak:
            new_value_rotation_from_v0_v1 = round(new_value_rotation_from_v0_v1 * 100) / 100

        self.target_set_value('rotation_from_v0_v1', value=[new_value_rotation_from_v0_v1])

        return {'RUNNING_MODAL'}


class TnBDirectionArrowGizmoGroup(GizmoGroup):
    bl_idname       = "OBJECT_GGT_TnBDirectionArrowGizmoGroup"
    bl_label        = "TnB texture direction widget"
    bl_space_type   = 'VIEW_3D'
    bl_region_type  = 'WINDOW'
    bl_options      = {'3D', 'PERSISTENT'}

    mesh_object     = None
    mesh_edit       = None
    mesh_face_layer = None
    selected_faces  = None

    @classmethod
    def update_mesh_object_and_mesh_edit_and_layer(cls, ob):
        if (TnBDirectionArrowGizmoGroup.mesh_object is None) or (TnBDirectionArrowGizmoGroup.mesh_object != ob.data):
            TnBDirectionArrowGizmoGroup.mesh_object = ob.data
            TnBDirectionArrowGizmoGroup.mesh_edit   = bmesh.from_edit_mesh(ob.data)

            if TnBDirectionArrowGizmoGroup.mesh_edit.faces.layers.float.get('tnb_rotation_from_v0_v1') is None:
                # Create new layer
                TnBDirectionArrowGizmoGroup.mesh_face_layer = TnBDirectionArrowGizmoGroup.mesh_edit.faces.layers.float.new('tnb_rotation_from_v0_v1')
            else:
                # Get existing layer
                TnBDirectionArrowGizmoGroup.mesh_face_layer = TnBDirectionArrowGizmoGroup.mesh_edit.faces.layers.float.get('tnb_rotation_from_v0_v1')


    @classmethod
    def save_edit_mesh(cls):
        bmesh.update_edit_mesh(TnBDirectionArrowGizmoGroup.mesh_object)


    @classmethod
    def update_selected_faces(cls):
        TnBDirectionArrowGizmoGroup.selected_faces = [f for f in TnBDirectionArrowGizmoGroup.mesh_edit.faces if f.select]


    @classmethod
    def has_single_selected_face(cls):
        if TnBDirectionArrowGizmoGroup.selected_faces is None:
            return False
        
        else:
            return len(TnBDirectionArrowGizmoGroup.selected_faces) == 1


    @classmethod
    def compute_gizmo_matrix(cls):
        if not TnBDirectionArrowGizmoGroup.has_single_selected_face():
            return mathutils.Matrix.Identity(4)
        
        else:
            selected_face = TnBDirectionArrowGizmoGroup.selected_faces[0]
            
            face_center = selected_face.calc_center_median()
            
            v_0     = selected_face.verts[0].co
            v_1     = selected_face.verts[1].co

            local_z = (selected_face.normal).normalized()
            local_x = (v_1 - v_0).normalized()
            local_y = (local_x.cross(local_z)).normalized()

            local_basis_in_world = mathutils.Matrix.Identity(4)
            for i in range(0, 3):
                local_basis_in_world[i][0] = local_x[i]
                local_basis_in_world[i][1] = local_y[i]
                local_basis_in_world[i][2] = local_z[i]

            return mathutils.Matrix.Translation(face_center) @ local_basis_in_world


    @classmethod
    def poll(cls, context):
        if context.mode != 'EDIT_MESH':
            return False
        
        ob = context.edit_object
        
        TnBDirectionArrowGizmoGroup.mesh_object = ob.data
        TnBDirectionArrowGizmoGroup.mesh_edit   = bmesh.from_edit_mesh(ob.data)
        if TnBDirectionArrowGizmoGroup.mesh_edit.faces.layers.float.get('tnb_rotation_from_v0_v1') is None:
            # Create new layer
            TnBDirectionArrowGizmoGroup.mesh_face_layer = TnBDirectionArrowGizmoGroup.mesh_edit.faces.layers.float.new('tnb_rotation_from_v0_v1')
        else:
            # Get existing layer
            TnBDirectionArrowGizmoGroup.mesh_face_layer = TnBDirectionArrowGizmoGroup.mesh_edit.faces.layers.float.get('tnb_rotation_from_v0_v1')
        TnBDirectionArrowGizmoGroup.update_selected_faces()
        
        return TnBDirectionArrowGizmoGroup.has_single_selected_face()


    def setup(self, context):
        ob = context.edit_object
        TnBDirectionArrowGizmoGroup.update_mesh_object_and_mesh_edit_and_layer(ob)
        TnBDirectionArrowGizmoGroup.update_selected_faces()
        TnBDirectionArrowGizmoGroup.save_edit_mesh()
    
        gz = self.gizmos.new(TnBDirectionArrowGizmo.bl_idname)
        gz.target_set_handler('rotation_from_v0_v1', get=get_rotation_offset, set=set_rotation_offset)

        gz.color = 0.0, 1.0, 1.0
        gz.alpha = 0.8

        gz.color_highlight = 0.0, 1.0, 1.0
        gz.alpha_highlight = 0.9

        # units are large, so shrink to something more reasonable.
        gz.scale_basis = 0.12
        gz.use_draw_modal = True

        self.rotation_gizmo = gz


    def refresh(self, context):
        ob = context.edit_object
            
        TnBDirectionArrowGizmoGroup.save_edit_mesh()
        TnBDirectionArrowGizmoGroup.update_mesh_object_and_mesh_edit_and_layer(ob)
        TnBDirectionArrowGizmoGroup.update_selected_faces()

        gz                      = self.rotation_gizmo
        rotation_offset_matrix  = mathutils.Matrix.Rotation( gz.target_get_value('rotation_from_v0_v1')[0], 4, (0.0, 0.0, 1.0) )
        gz.matrix_basis         = ob.matrix_world @ TnBDirectionArrowGizmoGroup.compute_gizmo_matrix() @ rotation_offset_matrix


def register():
    classes = (
        TnBDirectionArrowGizmo,
        TnBDirectionArrowGizmoGroup,
    )

    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    classes = (
        TnBDirectionArrowGizmo,
        TnBDirectionArrowGizmoGroup,
    )

    for cls in classes:
        bpy.utils.unregister_class(cls)