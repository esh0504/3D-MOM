# V6 base(06....) demo + comfort_viz integration
import os
import sys
import json
import argparse
import shutil
import time

import bpy
import numpy as np
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
CHARSIU_DIR = os.path.join(PROJECT_DIR, "charsiu_ko")
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")

sys.path.append(PROJECT_DIR)
sys.path.append(CHARSIU_DIR)
from modules.module import *
from modules.GT_Eval import mouth_move, tongue_move
from charsiu_ko.charsiu_predictive_aligner_ko import charsiu_predictive_aligner_ko
from modules.comfort_viz import apply_comfort_viz
from modules import blend_render

BASE_DIR = CHARSIU_DIR


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


parser = argparse.ArgumentParser(description="3D Blender Dataset")
parser.add_argument("--mode", default="prob", choices=["general", "prob", "gt", "all"])
parser.add_argument("--log", type=str2bool, default=True)
parser.add_argument("--dir", type=str2bool, default=True)  # legacy compat
parser.add_argument("--n_prob", type=int, default=5)
parser.add_argument("--input", default="./sample.wav")
parser.add_argument("--setting_dir", default=CHARSIU_DIR)
parser.add_argument("--save_path", type=str, default=None, help="...")
parser.add_argument("--base_modelpath", default=os.path.join(ASSETS_DIR, "mouth.blend"))
parser.add_argument(
    "--model_path",
    default=os.path.join(PROJECT_DIR, "weights", "model_ver_multi_w2v2_fc_10ms_ptk_weighted_v1"),
    help="Path to acoustic model directory (contains processor/ and vocab.json)",
)
parser.add_argument("--gt", type=str, default=None, help="single-file GT .PHN path")

# alpha값 조정 부분
parser.add_argument("--alpha_face", type=float, default=1.0)
parser.add_argument("--alpha_mouth", type=float, default=1.0)
parser.add_argument("--alpha_tongue", type=float, default=1.0)
parser.add_argument("--blend_tag", type=str, default=None, help="Optional tag to add to .blend filenames")

# comfort_viz
parser.add_argument("--comfort_viz", type=str2bool, default=True, help="Apply comfort_viz transparency post-process")
parser.add_argument("--comfort_viz_json", type=str, default=None, help="Path to vertex_groups_v3.json")
parser.add_argument("--comfort_alpha_front", type=float, default=0.7, help="Front(face) alpha for comfort_viz")
parser.add_argument("--comfort_alpha_side", type=float, default=0.4, help="Side(mouth group) alpha for comfort_viz")
parser.add_argument("--comfort_strict", type=str2bool, default=False, help="Raise error if comfort_viz fails")
parser.add_argument("--render", type=str2bool, default=True, help="Render MP4 right after saving .blend")
parser.add_argument("--render_blender_bin", type=str, default=None, help="Blender executable path")
parser.add_argument("--render_engine", type=str, default="auto", help="Render engine: auto/keep/BLENDER_EEVEE/BLENDER_EEVEE_NEXT/CYCLES")
parser.add_argument("--render_video_bitrate", type=int, default=8000, help="Render video bitrate kb/s")
parser.add_argument("--render_audio_bitrate", type=str, default="192k", help="Mux audio bitrate")
parser.add_argument("--render_muxer", type=str, default="auto", help="Audio muxer: auto/ffmpeg/moviepy/none")
parser.add_argument("--render_overwrite", type=str2bool, default=True, help="Overwrite output video if exists")
parser.add_argument("--render_keep_video_only", type=str2bool, default=False, help="Keep temporary silent mp4")
parser.add_argument("--render_verbose", type=str2bool, default=True, help="Verbose render logs")
parser.add_argument("--render_strict", type=str2bool, default=False, help="Raise error if render fails")

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def INIT_SETTING(setting_dir):
    lip_param = load_json(os.path.join(setting_dir, "k_lipmotion.json"))
    tongue_param = load_json(os.path.join(setting_dir, "k_tonguemotion.json"))
    vocal2model = load_json(os.path.join(setting_dir, "k_vocal2model.json"))
    vocab = load_json(os.path.join(setting_dir, "k_vocab-ctc.json"))
    return lip_param, tongue_param, vocal2model, vocab


def seg2ms(segments):
    return np.array([int((float(s) + float(e)) * 50) for s, e in segments], dtype=int)


def read_gt(file_path):
    sampling_rate = 16000
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            start_bit, end_bit, phoneme = line.strip().split()
            start_bit = int(start_bit)
            end_bit = int(end_bit)

            start_time = start_bit / sampling_rate
            end_time = end_bit / sampling_rate
            data.append([float(start_time), float(end_time), phoneme])
    return np.array(data, dtype=object)


def safe_unpack(vocal2model, idx):
    key = str(idx)
    if key in vocal2model:
        val = vocal2model[key][0]
        if isinstance(val, list) and len(val) == 2:
            return val[0], val[1]
        if isinstance(val, list) and len(val) == 1:
            return val[0], 0
    return 0, 0


def copy_texture_files(target_path):
    """
    Blender 모델 텍스처 복사 (mouth.jpg, Mouth_DIFFUSE_02.jpg)
    """
    print(f"[DEBUG] 복사 시작: {target_path}")
    os.makedirs(target_path, exist_ok=True)

    mouth_src = os.path.join(ASSETS_DIR, "mouth.jpg")
    if not os.path.exists(mouth_src):
        raise FileNotFoundError(f"Texture not found: {mouth_src}")
    shutil.copy(mouth_src, os.path.join(target_path, "mouth.jpg"))
    print("[DEBUG] mouth.jpg 복사 완료")

    diffuse_src = None
    for candidate in ["Mouth_DIFFUSE_02.jpg", "Mouth_DIFFUSE_02.png"]:
        candidate_path = os.path.join(ASSETS_DIR, candidate)
        if os.path.exists(candidate_path):
            diffuse_src = candidate_path
            break
    if diffuse_src is None:
        raise FileNotFoundError(
            f"Texture not found: expected one of {os.path.join(ASSETS_DIR, 'Mouth_DIFFUSE_02.jpg')} or "
            f"{os.path.join(ASSETS_DIR, 'Mouth_DIFFUSE_02.png')}"
        )
    shutil.copy(diffuse_src, os.path.join(target_path, os.path.basename(diffuse_src)))
    print(f"[DEBUG] {os.path.basename(diffuse_src)} 복사 완료")


def force_image_relink(image_name_contains, local_filename):
    for img in bpy.data.images:
        if image_name_contains.lower() in img.name.lower():
            new_path = f"//{local_filename}"
            abs_path = bpy.path.abspath(new_path)

            if os.path.exists(abs_path):
                img.filepath = new_path
                img.filepath_raw = new_path
                img.source = "FILE"
                img.use_fake_user = True
                img.reload()
                print(f"[RELINK ✅] {img.name} → {new_path}")
            else:
                print(f"[RELINK ❌] {img.name}: {abs_path} 없음")


def save_blender(path, name):
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"{name}.blend")

    bpy.ops.file.make_paths_relative()

    try:
        engines = bpy.context.scene.render.bl_rna.properties["engine"].enum_items.keys()
        if "BLENDER_EEVEE" in engines:
            bpy.context.scene.render.engine = "BLENDER_EEVEE"
            print("[ENGINE] Set to BLENDER_EEVEE")
        elif "BLENDER_EEVEE_NEXT" in engines:
            bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
            print("[ENGINE] Set to BLENDER_EEVEE_NEXT")
        else:
            bpy.context.scene.render.engine = "CYCLES"
            print("[ENGINE] Fallback to CYCLES")
    except Exception as e:
        print(f"[ERROR] Failed to set render engine: {e}")

    force_image_relink("mouth", "mouth.jpg")
    force_image_relink("diffuse", "Mouth_DIFFUSE_02.jpg")
    force_image_relink("diffuse", "Mouth_DIFFUSE_02.png")

    for img in bpy.data.images:
        try:
            img_path = bpy.path.abspath(img.filepath)
            if not img.packed_file and os.path.exists(img_path):
                img.pack()
                print(f"[PACK ✅] {img.name} 패킹 완료")
        except Exception as e:
            print(f"[PACK ❌] {img.name} 예외: {e}")

    bpy.ops.wm.save_as_mainfile(filepath=filepath)
    print(f"[SAVED] {filepath}")


def make_object_transparent(obj_name, alpha=0.1):
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        print(f"[❌] Object '{obj_name}' not found.")
        return

    for slot in obj.material_slots:
        mat = slot.material
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    node.inputs["Alpha"].default_value = alpha
                    mat.blend_method = "BLEND"
                    mat.shadow_method = "NONE"
                    print(f"[✅] '{obj_name}'의 머티리얼 '{mat.name}' → alpha={alpha}")


BRIGHTNESS_FACE = 1.2
BRIGHTNESS_MOUTH = 1.0
BRIGHTNESS_TONGUE = 1.0


def set_shape_key_value(obj_name, key_name, value):
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        print(f"[❌] Object '{obj_name}' not found.")
        return

    if not obj.data.shape_keys:
        print(f"[❌] Object '{obj_name}' has no shape keys.")
        return

    key_block = obj.data.shape_keys.key_blocks.get(key_name)
    if not key_block:
        print(f"[❌] Shape key '{key_name}' not found in '{obj_name}'.")
        return

    key_block.value = value
    print(f"[✅] '{obj_name}'의 shape key '{key_name}' → value={value}")


def brighten_object_color(obj_name, brightness=0.9):
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        print(f"[❌] Object '{obj_name}' not found.")
        return
    for slot in obj.material_slots:
        mat = slot.material
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    color = node.inputs["Base Color"].default_value
                    new_color = [min(c * brightness + 0.1, 1.0) for c in color[:3]] + [color[3]]
                    node.inputs["Base Color"].default_value = new_color
                    print(f"[🎨] '{obj_name}'의 색 → {new_color}")


def resolve_comfort_viz_json(cli_path):
    if cli_path:
        return cli_path
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    return os.path.join(current_dir, "vertex_groups_v3.json")


def build_final_name(base_name, blend_tag):
    return f"{blend_tag}_{base_name}" if blend_tag else base_name


def format_elapsed(seconds):
    if seconds >= 60:
        return f"{seconds:.2f}s ({seconds / 60:.2f}min)"
    return f"{seconds:.2f}s"


def get_scene_frame_count():
    scene = bpy.context.scene
    return max(scene.frame_end - scene.frame_start + 1, 0)


def format_elapsed_with_fps(seconds, frame_count):
    elapsed = format_elapsed(seconds)
    if seconds > 0 and frame_count > 0:
        fps = frame_count / seconds
        return f"{elapsed} | {fps:.2f} FPS"
    return elapsed


def print_pipeline_timing(label, blend_seconds, render_seconds, total_seconds, frame_count):
    print(f"[⏱️ TIMING] {label} ({frame_count} frames)")
    print(f"  Blend 생성: {format_elapsed_with_fps(blend_seconds, frame_count)}")
    if render_seconds > 0:
        print(f"  Rendering:  {format_elapsed_with_fps(render_seconds, frame_count)}")
    elif render_seconds == 0:
        print("  Rendering:  (skipped)")
    print(f"  전체:       {format_elapsed_with_fps(total_seconds, frame_count)}")


def prepare_scene(base_modelpath):
    load_blender(base_modelpath)
    bpy.context.scene.render.fps = 100
    bpy.ops.file.make_paths_relative()
    print("[DEBUG] Before setting: frame_end =", bpy.context.scene.frame_end)


def apply_visual_postprocess(args):
    make_object_transparent("Body001", alpha=args.alpha_face)
    make_object_transparent("Mouth", alpha=args.alpha_mouth)
    make_object_transparent("Tongue", alpha=args.alpha_tongue)

    brighten_object_color("Body001", brightness=BRIGHTNESS_FACE)
    brighten_object_color("Mouth", brightness=BRIGHTNESS_MOUTH)
    brighten_object_color("Tongue", brightness=BRIGHTNESS_TONGUE)

    set_shape_key_value("Mouth", "Gums_LV1_OBJ:Mesh", 1.0)

    if args.comfort_viz:
        if not args.comfort_viz_json or not os.path.exists(args.comfort_viz_json):
            msg = f"comfort_viz JSON not found: {args.comfort_viz_json}"
            if args.comfort_strict:
                raise FileNotFoundError(msg)
            print(f"[comfort_viz ⚠️] {msg}")
            return

        try:
            result = apply_comfort_viz(
                json_path=args.comfort_viz_json,
                alpha_front=args.comfort_alpha_front,
                alpha_side=args.comfort_alpha_side,
                face_obj="Body001",
                mouth_obj="Mouth",
                verbose=True,
            )
            print(
                f"[comfort_viz ✅] imported={list(result.imported_groups.keys())}, "
                f"separated={result.separated_objects}"
            )
            if result.warnings:
                print(f"[comfort_viz ⚠️] warnings={result.warnings}")
        except Exception as e:
            if args.comfort_strict:
                raise
            print(f"[comfort_viz ❌] apply failed: {e}")


def resolve_blender_bin(cli_bin):
    if cli_bin:
        return cli_bin
    local_candidates = [
        os.path.join(PROJECT_DIR, "tools", "blender", "blender-4.2.9", "blender"),
        os.path.join(PROJECT_DIR, "tools", "blender", "blender-4.2.9", "blender.exe"),
    ]
    for local_bin in local_candidates:
        if os.path.exists(local_bin) and os.access(local_bin, os.X_OK):
            return local_bin
    env_bin = os.environ.get("BLENDER_BIN")
    if env_bin:
        return env_bin
    return None


def run_render_for_blend(blend_path, wav_path, args):
    blender_bin = resolve_blender_bin(args.render_blender_bin)
    if not blender_bin:
        msg = (
            "Blender binary not found. Set --render_blender_bin/BLENDER_BIN "
            "or place it at ./tools/blender/blender-4.2.9/blender "
            "(Linux) or ./tools/blender/blender-4.2.9/blender.exe (Windows)."
        )
        if args.render_strict:
            raise RuntimeError(msg)
        print(f"[render ⚠️] {msg}")
        return

    render_argv = [
        "--blend",
        blend_path,
        "--blender-bin",
        blender_bin,
        "--engine",
        args.render_engine,
        "--video-bitrate",
        str(args.render_video_bitrate),
        "--audio-bitrate",
        args.render_audio_bitrate,
        "--muxer",
        args.render_muxer,
        "--overwrite",
        str(args.render_overwrite),
        "--keep-video-only",
        str(args.render_keep_video_only),
        "--verbose",
        str(args.render_verbose),
    ]
    if wav_path:
        render_argv.extend(["--wav", wav_path])

    try:
        blend_render.main(render_argv)
    except Exception as e:
        if args.render_strict:
            raise
        print(f"[render ❌] failed: {e}")


def finalize_and_save(output_dir, final_name, args, wav_path=None, pipeline_start=None, timing_label=None):
    apply_visual_postprocess(args)
    print(f"[🔒 SAVE] {final_name}.blend will be saved in: {output_dir}")
    save_blender(output_dir, final_name)

    blend_seconds = time.perf_counter() - pipeline_start if pipeline_start is not None else None

    render_seconds = 0.0
    if args.render:
        blend_path = os.path.join(output_dir, f"{final_name}.blend")
        render_start = time.perf_counter()
        run_render_for_blend(blend_path=blend_path, wav_path=wav_path, args=args)
        render_seconds = time.perf_counter() - render_start

    if pipeline_start is not None:
        total_seconds = time.perf_counter() - pipeline_start
        label = timing_label or final_name
        frame_count = get_scene_frame_count()
        print_pipeline_timing(label, blend_seconds, render_seconds, total_seconds, frame_count)


def ensure_result_dirs(result_root, n_prob):
    result_general_path = os.path.join(result_root, "general")
    result_prob_path = os.path.join(result_root, f"prob_{n_prob}")
    result_gt_path = os.path.join(result_root, "gt_real")

    for path in [result_general_path, result_prob_path, result_gt_path]:
        os.makedirs(path, exist_ok=True)
        copy_texture_files(path)

    return result_general_path, result_prob_path, result_gt_path


def run_general(file_path, output_dir, final_name, args, charsiu, lip_param, tongue_param, vocal2model, vocab):
    pipeline_start = time.perf_counter()
    prepare_scene(args.base_modelpath)

    segment = np.array(charsiu.align(file_path), dtype=object)
    frame = 1
    result_array = np.zeros((max(len(segment) * 2, 1), 23))

    for i in range(len(segment)):
        start = float(segment[i][0])
        end = float(segment[i][1])
        phone = segment[i][2]

        frame = int((start + end) * 50)

        if phone not in vocab:
            print(f"[general ⚠️] vocab에 없는 phone: {phone}")
            continue

        params = vocal2model.get(str(vocab[phone]), [[0, 0]])

        if len(params) == 1:
            lip, tongue = params[0]
            result_array[i * 2, :7] = lip_param[str(lip)]
            result_array[i * 2, 7:] = tongue_param[str(tongue)]
            mouth_move(result_array[i * 2][:7], frame)
            tongue_move(result_array[i * 2][7:], frame)
        else:
            for idx, (lip, tongue) in enumerate(params[:2]):
                arr_idx = i * 2 + idx
                result_array[arr_idx, :7] = lip_param[str(lip)]
                result_array[arr_idx, 7:] = tongue_param[str(tongue)]

                if idx == 0:
                    use_frame = int(33 * end + 66 * start)
                else:
                    use_frame = int(33 * start + 66 * end)

                mouth_move(result_array[arr_idx][:7], use_frame)
                tongue_move(result_array[arr_idx][7:], use_frame)

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = max(frame + 10, 11)
    finalize_and_save(
        output_dir,
        final_name,
        args,
        wav_path=file_path,
        pipeline_start=pipeline_start,
        timing_label=f"general/{final_name}",
    )


def run_prob(file_path, output_dir, final_name, args, charsiu, lip_param, tongue_param, vocal2model, logvocab):
    pipeline_start = time.perf_counter()
    prepare_scene(args.base_modelpath)

    log_file = os.path.join(output_dir, final_name + ".txt") if args.log else None

    probability, segment = charsiu.align_probabilistic(audio=file_path)
    segment = np.array(segment, dtype=object)

    if len(segment) == 0:
        print("[prob ⚠️] segment가 비어있어서 빈 씬으로 저장합니다.")
        bpy.context.scene.render.fps = 100
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = probability.shape[1] + 10
        finalize_and_save(
            output_dir,
            final_name,
            args,
            wav_path=file_path,
            pipeline_start=pipeline_start,
            timing_label=f"prob/{final_name}",
        )
        return

    mss = seg2ms(segment)
    mss = np.clip(mss, 0, probability.shape[1] - 1)

    motions = np.zeros((len(segment) * 2, 23), dtype=float)
    sampled_prob = probability[0:, mss, :].squeeze(0)

    log_handle = open(log_file, "w", encoding="utf-8") if log_file else None
    try:
        for motion_idx, (prob_vec, _) in enumerate(zip(sampled_prob, mss)):
            samples, idxs = torch.topk(torch.tensor(prob_vec), args.n_prob)
            start = float(segment[motion_idx][0])
            end = float(segment[motion_idx][1])

            if log_handle:
                log_handle.write(f"{start:.2f}~{end:.2f}\t")
                log_handle.write(
                    ", ".join(
                        [
                            f"{logvocab[idx.item()]:>3} : {samples[j].item():<5.2f}".ljust(15)
                            for j, idx in enumerate(idxs)
                        ]
                    )
                )
                log_handle.write("\n")

            for sample, idx in zip(samples, idxs):
                params = vocal2model.get(str(idx.item()), [])
                if not params:
                    continue

                weight = float(sample.item())

                if len(params) == 1:
                    lip, tongue = params[0]
                    motions[motion_idx * 2, :7] += np.array(lip_param[str(lip)]) * weight
                    motions[motion_idx * 2, 7:] += np.array(tongue_param[str(tongue)]) * weight
                else:
                    for index, (lip, tongue) in enumerate(params[:2]):
                        target_idx = motion_idx * 2 - 1 if index == 0 else motion_idx * 2 + 1
                        if 0 <= target_idx < len(motions):
                            motions[target_idx, :7] += np.array(lip_param[str(lip)]) * weight
                            motions[target_idx, 7:] += np.array(tongue_param[str(tongue)]) * weight

            current_idx = motion_idx * 2
            prev_idx = current_idx - 1
            next_idx = current_idx + 1

            mid_frame = int((start + end) * 50)

            if next_idx >= len(motions) or not np.any(motions[next_idx]):
                mouth_move(motions[current_idx][:7], mid_frame)
                tongue_move(motions[current_idx][7:], mid_frame)
            else:
                left_motion = motions[current_idx].copy()
                if 0 <= prev_idx < len(motions):
                    left_motion += motions[prev_idx]

                right_motion = motions[current_idx].copy() + motions[next_idx]

                mouth_move(left_motion[:7], int(33 * end + 66 * start))
                tongue_move(left_motion[7:], int(33 * end + 66 * start))
                mouth_move(right_motion[:7], int(66 * end + 33 * start))
                tongue_move(right_motion[7:], int(66 * end + 33 * start))

                motions[next_idx] = np.zeros(23)

    finally:
        if log_handle:
            log_handle.close()

    bpy.context.scene.render.fps = 100
    bpy.context.scene.frame_start = 1
    last_time = float(segment[-1][1])
    bpy.context.scene.frame_end = int(last_time * 100) + 10

    print("[DEBUG] Before setting: frame_end =", bpy.context.scene.frame_end)
    finalize_and_save(
        output_dir,
        final_name,
        args,
        wav_path=file_path,
        pipeline_start=pipeline_start,
        timing_label=f"prob/{final_name}",
    )


def run_gt(gt_path, output_dir, final_name, args, lip_param, tongue_param, vocal2model, vocab):
    pipeline_start = time.perf_counter()
    prepare_scene(args.base_modelpath)

    gts = read_gt(gt_path)
    mouth_move(np.array(lip_param[str(0)]), 1)
    tongue_move(np.array(tongue_param[str(0)]), 1)

    last_end_frame = 1

    for start_frame, end_frame, phone in gts:
        try:
            if phone == "h#" or len(phone) > 2:
                continue

            phone_key = phone.upper()
            if phone_key not in vocab:
                continue

            vocab_idx = str(vocab[phone_key])
            params = vocal2model.get(vocab_idx, [[0, 0]])

            if len(params) == 1:
                lip, tongue = params[0]
                frame = int((float(start_frame) + float(end_frame)) * 50)
                mouth_move(np.array(lip_param[str(lip)]), frame)
                tongue_move(np.array(tongue_param[str(tongue)]), frame)
            else:
                for index, attr in enumerate(params[:2]):
                    lip, tongue = attr
                    if index == 0:
                        frame = int(66 * float(start_frame) + 33 * float(end_frame))
                    else:
                        frame = int(33 * float(start_frame) + 66 * float(end_frame))
                    mouth_move(np.array(lip_param[str(lip)]), frame)
                    tongue_move(np.array(tongue_param[str(tongue)]), frame)

            last_end_frame = int(float(end_frame) * 100)
        except Exception as e:
            print(f"[gt ⚠️] skip ({phone}): {e}")

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = last_end_frame + 10
    wav_candidate_upper = os.path.splitext(gt_path)[0] + ".WAV"
    wav_candidate_lower = os.path.splitext(gt_path)[0] + ".wav"
    wav_path = wav_candidate_upper if os.path.exists(wav_candidate_upper) else None
    if wav_path is None and os.path.exists(wav_candidate_lower):
        wav_path = wav_candidate_lower

    finalize_and_save(
        output_dir,
        final_name,
        args,
        wav_path=wav_path,
        pipeline_start=pipeline_start,
        timing_label=f"gt/{final_name}",
    )


def process_directory_input(args, charsiu, lip_param, tongue_param, vocal2model, vocab, logvocab):
    locations = sorted(os.listdir(args.input))
    for location in locations:
        location_path = os.path.join(args.input, location)
        if not os.path.isdir(location_path):
            continue

        dirs = sorted(os.listdir(location_path))
        for dir_name in dirs:
            source_dir = os.path.join(location_path, dir_name)
            if not os.path.isdir(source_dir):
                continue

            result_root = os.path.join(args.save_path, location, dir_name)
            result_general_path, result_prob_path, result_gt_path = ensure_result_dirs(result_root, args.n_prob)

            files_all = os.listdir(source_dir)
            wav_files = [file for file in files_all if file.lower().endswith(".wav")]
            phn_files = [file for file in files_all if file.lower().endswith(".phn")]

            for file in wav_files:
                file_path = os.path.join(source_dir, file)
                wav_filename = os.path.splitext(file)[0]
                final_name = build_final_name(wav_filename, args.blend_tag)

                if args.mode in {"general", "all"}:
                    run_general(
                        file_path=file_path,
                        output_dir=result_general_path,
                        final_name=final_name,
                        args=args,
                        charsiu=charsiu,
                        lip_param=lip_param,
                        tongue_param=tongue_param,
                        vocal2model=vocal2model,
                        vocab=vocab,
                    )

                if args.mode in {"prob", "all"}:
                    run_prob(
                        file_path=file_path,
                        output_dir=result_prob_path,
                        final_name=final_name,
                        args=args,
                        charsiu=charsiu,
                        lip_param=lip_param,
                        tongue_param=tongue_param,
                        vocal2model=vocal2model,
                        logvocab=logvocab,
                    )

            if args.mode in {"gt", "all"}:
                for file in phn_files:
                    gt_path = os.path.join(source_dir, file)
                    gt_name = os.path.splitext(file)[0]
                    final_name = build_final_name(gt_name, args.blend_tag)

                    run_gt(
                        gt_path=gt_path,
                        output_dir=result_gt_path,
                        final_name=final_name,
                        args=args,
                        lip_param=lip_param,
                        tongue_param=tongue_param,
                        vocal2model=vocal2model,
                        vocab=vocab,
                    )


def process_single_file_input(args, charsiu, lip_param, tongue_param, vocal2model, vocab, logvocab):
    wav_filename = os.path.splitext(os.path.basename(args.input))[0]

    result_root = os.path.join(args.save_path, wav_filename)
    result_general_path, result_prob_path, result_gt_path = ensure_result_dirs(result_root, args.n_prob)
    final_name = build_final_name(wav_filename, args.blend_tag)

    if args.mode in {"general", "all"}:
        run_general(
            file_path=args.input,
            output_dir=result_general_path,
            final_name=final_name,
            args=args,
            charsiu=charsiu,
            lip_param=lip_param,
            tongue_param=tongue_param,
            vocal2model=vocal2model,
            vocab=vocab,
        )

    if args.mode in {"prob", "all"}:
        run_prob(
            file_path=args.input,
            output_dir=result_prob_path,
            final_name=final_name,
            args=args,
            charsiu=charsiu,
            lip_param=lip_param,
            tongue_param=tongue_param,
            vocal2model=vocal2model,
            logvocab=logvocab,
        )

    if args.mode in {"gt", "all"}:
        gt_candidates = []
        if args.gt:
            gt_candidates.append(args.gt)

        gt_candidates.append(os.path.splitext(args.input)[0] + ".PHN")
        gt_candidates.append(os.path.splitext(args.input)[0] + ".phn")

        gt_path = next((p for p in gt_candidates if p and os.path.exists(p)), None)

        if gt_path is None:
            print("[gt ⚠️] 단일 파일 GT 경로를 찾지 못했습니다. --gt 로 직접 지정하세요.")
        else:
            run_gt(
                gt_path=gt_path,
                output_dir=result_gt_path,
                final_name=final_name,
                args=args,
                lip_param=lip_param,
                tongue_param=tongue_param,
                vocal2model=vocal2model,
                vocab=vocab,
            )


if __name__ == "__main__":
    args = parser.parse_args()
    args.model_path = os.path.abspath(args.model_path)

    processor_path = os.path.join(args.model_path, "processor")
    vocab_path = os.path.join(args.model_path, "vocab.json")
    model_name = os.path.basename(args.model_path)
    print(f"✅ 사용 모델: {model_name}")
    print(f"✅ 모델 경로: {args.model_path}")
    print(f"✅ Processor 경로: {processor_path}")
    print(f"✅ Vocab 경로: {vocab_path}")

    is_dir = os.path.isdir(args.input)

    if not args.save_path:
        if not is_dir:
            args.save_path = os.path.join(PROJECT_DIR, "results", "results_ko_onlywav_realdict")
        else:
            args.save_path = os.path.join(PROJECT_DIR, "results", "results_ko_realdict")

    args.comfort_viz_json = resolve_comfort_viz_json(args.comfort_viz_json)

    print(f"[📂] 최종 저장 경로: {args.save_path}")
    print(f"[comfort_viz] enabled={args.comfort_viz}, json={args.comfort_viz_json}")

    lip_param, tongue_param, vocal2model, vocab = INIT_SETTING(args.setting_dir)

    logvocab = [None] * len(vocab)
    for phone, idx in vocab.items():
        if idx < len(logvocab):
            logvocab[idx] = phone

    charsiu = charsiu_predictive_aligner_ko(
        model_path=args.model_path,
        processor_path=processor_path,
        vocab_path=vocab_path,
    )

    if is_dir:
        process_directory_input(
            args=args,
            charsiu=charsiu,
            lip_param=lip_param,
            tongue_param=tongue_param,
            vocal2model=vocal2model,
            vocab=vocab,
            logvocab=logvocab,
        )
    else:
        process_single_file_input(
            args=args,
            charsiu=charsiu,
            lip_param=lip_param,
            tongue_param=tongue_param,
            vocal2model=vocal2model,
            vocab=vocab,
            logvocab=logvocab,
        )
