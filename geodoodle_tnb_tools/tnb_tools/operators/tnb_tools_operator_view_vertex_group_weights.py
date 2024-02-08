import bpy
import bmesh


class TnbToolsOperatorViewVertexGroupWeights(bpy.types.Operator):
    bl_idname       = "tnb_tools.operator_view_vert_grp_weight"
    bl_label        = "tnb_tools_view_vert_grp_weight"
    bl_description  = "View vertex group weights"


    @classmethod
    def poll(cls, context):
        return (context.mode == 'EDIT_MESH') and (len(context.selected_objects) == 1) and (context.selected_objects[0].type == 'MESH')


    def execute(self, context):
        context.space_data.overlay.show_weight = True

        return {'FINISHED'}


def register():
    bpy.utils.register_class(TnbToolsOperatorViewVertexGroupWeights)


def unregister():
    bpy.utils.unregister_class(TnbToolsOperatorViewVertexGroupWeights)


if __name__ == "__main__":
    register()