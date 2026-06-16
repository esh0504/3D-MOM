import bpy
import argparse
import pickle
import os
import numpy as np
import json

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

def save_blender(save, name):
    try:
        if not os.path.isdir(save):
            os.mkdir(save)
        save_path = os.path.join(save, name + '.blend')
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

"""
values = [value of 'A', value of 'E', value of 'O', value of 'U', 
            value of 'Pb', value of 'FV', value of 'Tooth']
frame: Animation Frame (int)
"""
def mouth_move(values, frame):
    # Setting Mouth Animation
    Lip = ['A', 'O', 'E', 'U', 'P,B', 'F,V']
    # try:
    mouth_obj = bpy.data.shape_keys["Key.002"]
    tooth_obj = bpy.data.shape_keys["Key"]
    if mouth_obj is not None:
        shape_keys = mouth_obj.key_blocks
        frame = frame
        bpy.context.scene.frame_start = frame

        for key, value in zip(Lip, values[:6]):
            shape_key = shape_keys[key]
            shape_key.value = value
            shape_key.keyframe_insert(data_path="value", frame=frame)
        else:
            pass

        shape_keys = tooth_obj.key_blocks
        shape_key = shape_keys['Gums_LV1_OBJ:Mesh']

        shape_key.value = values[6]
        shape_key.keyframe_insert(data_path="value", frame=frame)
    else:
        pass


"""
values = [
            value of 'bone1 rotation', value of 'bone2 rotation', value of 'bone3 rotation', value of 'bone4 rotation', 
            value of 'bone1 pos_x', value of 'bone1 pos_x', value of 'bone1 pos_x', 
            value of 'bone2 pos_x', value of 'bone2 pos_x', value of 'bone2 pos_x', 
            value of 'bone3 pos_x', value of 'bone3 pos_y', value of 'bone3 pos_y', 
            value of 'bone4 pos_x', value of 'bone4 pos_z', value of 'bone4 pos_z'
        ]
frame: Animation Frame (int)
"""
def tongue_move(values, frame):

    Bones = ['Bone001', 'Bone002', 'Bone003', 'Bone004']
    # try:
    # Setting Tongue Animation
    tongue_obj = bpy.data.objects.get('Armature')
    frame = frame
    if tongue_obj and tongue_obj.type == 'ARMATURE':
        bpy.context.view_layer.objects.active = tongue_obj
        tongue_obj.select_set(True)
        bpy.ops.object.mode_set(mode='POSE')

        for idx in range(4):
            bone = tongue_obj.pose.bones.get(Bones[idx])

            bone.rotation_mode = 'XYZ'
            bone.rotation_euler = (0, 0, values[idx] / 57.296)
            bone.keyframe_insert(data_path="rotation_euler", frame=frame)
            # print(idx, values[4+idx*3], values[5+idx*3], values[6+idx*3])
            bone.location = (values[4+idx*3], values[5+idx*3], values[6+idx*3])
            bone.keyframe_insert(data_path="location", frame=frame)

            # Exit pose mode
        bpy.ops.object.mode_set(mode='OBJECT')
    else:
        print('we can\'t move Tongue')
    # except:
    #     print('we can\'t move Tongue')

def sec2frame(sec):
    frame = max(int(30 * sec), 1)
    return frame
"""
sampa_datas = [(sampa,sec)....]

hE"l@U w3:ld
"""
def make_animation(values, frames):

    curr_frame = 1
    for value, frame in zip(values, frames):
        curr_frame += sec2frame(frame)
        mouth_move(value[:7], curr_frame)
        tongue_move(value[7:], curr_frame)

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = curr_frame + 10

import json
def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

if __name__ == '__main__':

    lipmotion = load_json('../charsiu/src/lipmotion.json')  # lipmotion.json
    tonguemotion = load_json('../charsiu/src/tonguemotion.json')  # tonguemotion.json
    vocal2model = load_json('../charsiu/src/vocal2model.json')  # vocal2model.json
    vocab = load_json('../charsiu/src/vocab-ctc.json')

    dummy = np.array([(0.0, 0.78, '[SIL]'), (0.78, 0.85, 'D'), (0.85, 0.93, 'OW'), (0.93, 0.97, 'N'), (0.97, 1.0, 'T'), (1.0, 1.13, 'AE'), (1.13, 1.25, 'S'), (1.25, 1.3, 'K'), (1.3, 1.35, 'M'), (1.35, 1.44, 'IY'), (1.44, 1.48, 'T'), (1.48, 1.53, 'IH'), (1.53, 1.62, 'K'), (1.62, 1.63, 'EH'), (1.63, 1.77, 'AE'), (1.77, 1.78, 'EH'), (1.78, 1.85, 'R'), (1.85, 1.95, 'IY'), (1.95, 2.0, 'AH'), (2.0, 2.06, 'N'), (2.06, 2.26, 'OY'), (2.26, 2.36, 'L'), (2.36, 2.44, 'IY'), (2.44, 2.53, 'R'), (2.53, 2.7, 'EH'), (2.7, 2.78, 'G'), (2.78, 2.84, 'L'), (2.84, 2.94, 'AY'), (2.94, 3.07, 'K'), (3.07, 3.12, 'DH'), (3.12, 3.28, 'AE'), (3.28, 3.31, 'T'), (3.31, 3.61, '[SIL]')])
    frame = 1
    result_array = np.zeros((dummy.shape[0], 23))
    load_blender('../assets/baseWithQuadrangle.blend')


    for i in range(dummy.shape[0]):

        sec = (float(dummy[i][1]) - float(dummy[i][0])) / 2
        frame += sec2frame(sec)*5
        phone = dummy[i][2]
        vocab_idx = str(vocab[phone])
        lip, tongue = vocal2model[vocab_idx]
        result_array[i, :7] = lipmotion[str(lip)]
        result_array[i, 7:] = tonguemotion[str(tongue)]
        mouth_move(result_array[i][:7], frame)
        tongue_move(result_array[i][7:], frame)

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frame + 10
    save_blender('./', 'test')


