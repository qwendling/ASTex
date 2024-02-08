import bpy
import bmesh


class TnbToolsOperatorTriangulate(bpy.types.Operator):
    bl_idname       = "tnb_tools.operator_triangulate"
    bl_label        = "tnb_tools_triangulate_selected_mesh"
    bl_description  = "Trialgulate the selected mesh"


    @classmethod
    def poll(cls, context):
        single_object_selected = (len(bpy.context.selected_objects) == 1)
        
        if single_object_selected is False:
            return False
        
        selected_object_is_mesh = (bpy.context.selected_objects[0].type == 'MESH')

        return selected_object_is_mesh
    

    def execute(self, context):
        if context.mode == 'EDIT_MESH':
            edit_mesh = bmesh.from_edit_mesh(context.edit_object.data)
            bmesh.ops.triangulate(edit_mesh, faces=edit_mesh.faces[:], quad_method='BEAUTY', ngon_method='BEAUTY')
            bmesh.update_edit_mesh(context.edit_object.data)

        elif context.mode == 'OBJECT':
            selected_mesh       = bpy.context.selected_objects[0]
            all_modifiers_type  = [modif.type for modif in selected_mesh.modifiers]
            
            if not ('TRIANGULATE' in all_modifiers_type):
                bpy.ops.object.modifier_add(type='TRIANGULATE')

            for curr_modifier in selected_mesh.modifiers:
                if curr_modifier.type == 'TRIANGULATE':
                    bpy.ops.object.modifier_apply(modifier=curr_modifier.name)

        return {'FINISHED'}


def register():
    bpy.utils.register_class(TnbToolsOperatorTriangulate)


def unregister():
    bpy.utils.unregister_class(TnbToolsOperatorTriangulate)


if __name__ == "__main__":
    register()