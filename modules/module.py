import bpy
import argparse
import pickle
import os

def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except:
        print(f'we can\'t load {file_path}')

def load_blender(file_path):
    try:
        bpy.ops.wm.open_mainfile(filepath=file_path)
        delete_all_keyframes()
    except:
        print(f'we can\'t load {file_path}')

def save_blender(save, exp_name):
    try:
        save_path = os.path.join(save, exp_name+'.blend')
        bpy.ops.wm.save_mainfile(filepath=save_path)

        print(f'----------save in {save_path}')
    except:
        print('save error')

def delete_all_keyframes():
    # Iterate through all objects in the scene
    for obj in bpy.context.scene.objects:
        # Check if the object has animation data
        if obj.animation_data:
            # Clear all keyframes for this object's action
            obj.animation_data_clear()

        # Check if the object has shape keys with animation data
        if obj.data and hasattr(obj.data, "shape_keys") and obj.data.shape_keys:
            shape_keys = obj.data.shape_keys
            if shape_keys.animation_data:
                shape_keys.animation_data_clear()

        # Check for armature bones with animation data
        if obj.type == 'ARMATURE':
            for bone in obj.pose.bones:
                if bone.bone.select:
                    bone.bone.select = False
            for bone in obj.pose.bones:
                if obj.animation_data:
                    for fcurve in obj.animation_data.action.fcurves:
                        obj.animation_data.action.fcurves.remove(fcurve)
            bpy.ops.object.mode_set(mode='OBJECT')

def make_animation(sampa_data, words):
    sampas = list(words)
    start_frame = 1
    talking_time = 14
    for sampa in sampas:
        mouth_move(sampa_data[sampa]['mouth_motion'], start_frame, talking_time)
        tongue_move(sampa_data[sampa]['tongue_motion'], start_frame, talking_time)
        tooth_move(sampa_data[sampa]['tooth_motion'], start_frame, talking_time)
        start_frame += talking_time

    bpy.context.scene.frame_start = 1
def sec2frame(sec):
    frame = max(int(30 * sec), 1)
    return frame

if __name__ == '__main__':
    sampa_data = load_pickle('./motions.pkl')
    load_blender('./basemodel.blend')
    make_animation(sampa_data, '는동주롭기며')
    save_blender('./', 'test')