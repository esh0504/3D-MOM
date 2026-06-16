"""
comfort_viz.py

JSON 기반 vertex group 정보를 Blender 씬에 주입하고,
comfort visualization(부분 투명화 / 0-alpha 분리)을 적용하는 모듈입니다.

사용 예시:
    from comfort_viz import apply_comfort_viz

    apply_comfort_viz(
        json_path="/path/to/vertex_groups_v3.json",
        alpha_front=0.7,
        alpha_side=0.4,
    )
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import bpy


DEFAULT_FACE_OBJ = "Body001"
DEFAULT_MOUTH_OBJ = "Mouth"

DEFAULT_ZERO_ALPHA_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("Body001", "VG_NOSEHALL"),
    ("Mouth", "VG_UPPER_TEETH_ROOT"),
    ("Mouth", "VG_THROAT"),
)

DEFAULT_SIDE_ALPHA_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("Mouth", "VG_TEETH"),
    ("Mouth", "VG_GUM_UPPER"),
)


@dataclass
class ComfortVizResult:
    json_path: str
    imported_groups: Dict[str, Dict[str, int]]
    separated_objects: List[str]
    warnings: List[str] = field(default_factory=list)


def _log(message: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[comfort_viz] {message}")


def _warn(message: str, warnings: List[str], verbose: bool = True) -> None:
    warnings.append(message)
    _log(f"WARNING: {message}", verbose=verbose)


def _load_group_json(json_path: str) -> dict:
    if not json_path:
        raise ValueError("json_path is empty.")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"comfort_viz JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_object_mode() -> None:
    active = bpy.context.view_layer.objects.active
    if active is not None:
        try:
            if active.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            # 이미 OBJECT 모드이거나 active context가 없는 경우 무시
            pass


def set_active(obj: bpy.types.Object) -> None:
    if obj is None:
        raise ValueError("set_active() received None object.")

    _ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def enter_edit(obj: bpy.types.Object) -> None:
    if obj is None:
        raise ValueError("enter_edit() received None object.")
    if obj.type != "MESH":
        raise TypeError(f"'{obj.name}' is not a mesh object.")

    set_active(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")


def exit_object() -> None:
    active = bpy.context.view_layer.objects.active
    if active is not None:
        try:
            if active.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass


def select_vg(obj: bpy.types.Object, vg_name: str) -> bool:
    vg = obj.vertex_groups.get(vg_name)
    if vg is None:
        return False

    obj.vertex_groups.active_index = vg.index
    bpy.ops.object.vertex_group_select()
    return True


def _get_base_material(obj: bpy.types.Object) -> Optional[bpy.types.Material]:
    if obj is None or getattr(obj.data, "materials", None) is None:
        return None
    if len(obj.data.materials) == 0:
        return None
    return obj.data.materials[0]


def set_alpha(mat: bpy.types.Material, alpha: float) -> None:
    mat.use_nodes = True

    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED" and "Alpha" in node.inputs:
            node.inputs["Alpha"].default_value = alpha

    if hasattr(mat, "blend_method"):
        mat.blend_method = "BLEND"
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "NONE"


def ensure_alpha_material(
    obj: bpy.types.Object,
    alpha: float,
    name_suffix: str,
) -> int:
    src = _get_base_material(obj)
    if src is None:
        raise ValueError(f"'{obj.name}' has no base material.")

    new_name = f"{src.name}_{name_suffix}"
    mat = bpy.data.materials.get(new_name)
    if mat is None:
        mat = src.copy()
        mat.name = new_name

    set_alpha(mat, alpha)

    slot_index = obj.data.materials.find(mat.name)
    if slot_index == -1:
        obj.data.materials.append(mat)
        slot_index = obj.data.materials.find(mat.name)

    return slot_index


def assign_alpha_to_vg(
    obj: bpy.types.Object,
    vg_name: str,
    slot_index: int,
    warnings: Optional[List[str]] = None,
    verbose: bool = True,
) -> bool:
    warnings = warnings if warnings is not None else []

    if vg_name not in obj.vertex_groups:
        _warn(f"Vertex group '{vg_name}' not found in '{obj.name}'.", warnings, verbose)
        return False

    enter_edit(obj)
    try:
        ok = select_vg(obj, vg_name)
        if ok:
            obj.active_material_index = slot_index
            bpy.ops.object.material_slot_assign()
            _log(f"Alpha assigned: {obj.name}.{vg_name}", verbose)
        return ok
    finally:
        exit_object()


def assign_alpha_entire(
    obj: bpy.types.Object,
    slot_index: int,
    verbose: bool = True,
) -> None:
    enter_edit(obj)
    try:
        bpy.ops.mesh.select_all(action="SELECT")
        obj.active_material_index = slot_index
        bpy.ops.object.material_slot_assign()
        _log(f"Alpha assigned to entire object: {obj.name}", verbose)
    finally:
        exit_object()


def create_zero_alpha_material(src_obj: bpy.types.Object, suffix: str = "ALPHA_ZERO") -> bpy.types.Material:
    src = _get_base_material(src_obj)
    if src is None:
        raise ValueError(f"'{src_obj.name}' has no base material.")

    mat_name = f"{src.name}_{suffix}"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = src.copy()
        mat.name = mat_name

    set_alpha(mat, 0.0)
    return mat


def import_vertex_groups_from_json(
    data: dict,
    warnings: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, int]]:
    warnings = warnings if warnings is not None else []
    result: Dict[str, Dict[str, int]] = {}

    targets = data.get("targets", [])
    if not isinstance(targets, list):
        raise ValueError("Invalid JSON format: 'targets' must be a list.")

    for target in targets:
        obj_name = target.get("object_name")
        groups = target.get("groups", {})
        obj = bpy.data.objects.get(obj_name)

        if obj is None:
            _warn(f"Object '{obj_name}' not found in current scene.", warnings, verbose)
            continue

        existing = 0
        created = 0
        skipped = 0

        for group_name, verts in groups.items():
            vg = obj.vertex_groups.get(group_name)
            if vg is None:
                vg = obj.vertex_groups.new(name=group_name)
                created += 1
            else:
                existing += 1

            for vidx, weight in verts.items():
                idx = int(vidx)
                if idx < len(obj.data.vertices):
                    vg.add([idx], float(weight), "REPLACE")
                else:
                    skipped += 1

        result[obj_name] = {
            "existing_groups": existing,
            "created_groups": created,
            "skipped_vertices": skipped,
        }

        _log(
            f"Imported vertex groups for {obj_name} "
            f"(existing={existing}, created={created}, skipped_vertices={skipped})",
            verbose,
        )

    return result


def separate_vg_to_object_with_zero_alpha(
    obj: bpy.types.Object,
    vg_name: str,
    warnings: Optional[List[str]] = None,
    verbose: bool = True,
) -> Optional[str]:
    warnings = warnings if warnings is not None else []

    if vg_name not in obj.vertex_groups:
        _warn(f"Vertex group '{vg_name}' not found in '{obj.name}'.", warnings, verbose)
        return None

    object_names_before = set(bpy.data.objects.keys())

    enter_edit(obj)
    try:
        if not select_vg(obj, vg_name):
            return None
        bpy.ops.mesh.separate(type="SELECTED")
    finally:
        exit_object()

    new_names = [name for name in bpy.data.objects.keys() if name not in object_names_before]
    new_obj = bpy.data.objects.get(new_names[0]) if new_names else None

    if new_obj is None:
        # fallback: 선택된 오브젝트 중 원본이 아닌 것 찾기
        for candidate in bpy.context.selected_objects:
            if candidate.name != obj.name:
                new_obj = candidate
                break

    if new_obj is None:
        _warn(f"Failed to separate '{vg_name}' from '{obj.name}'.", warnings, verbose)
        return None

    try:
        new_obj.name = f"{obj.name}_{vg_name}_ZERO"
    except Exception:
        # 이름 충돌 시 Blender가 자동 suffix를 붙이게 둠
        pass

    zero_mat = create_zero_alpha_material(obj)
    new_obj.data.materials.clear()
    new_obj.data.materials.append(zero_mat)

    _log(f"Separated '{vg_name}' -> '{new_obj.name}' with 0 alpha", verbose)
    return new_obj.name


def apply_comfort_viz(
    json_path: str,
    alpha_front: float = 0.7,
    alpha_side: float = 0.4,
    face_obj: str = DEFAULT_FACE_OBJ,
    mouth_obj: str = DEFAULT_MOUTH_OBJ,
    zero_alpha_groups: Sequence[Tuple[str, str]] = DEFAULT_ZERO_ALPHA_GROUPS,
    side_alpha_groups: Sequence[Tuple[str, str]] = DEFAULT_SIDE_ALPHA_GROUPS,
    verbose: bool = True,
) -> ComfortVizResult:
    """
    JSON의 vertex group 정보를 현재 Blender 씬에 반영하고,
    comfort visualization용 투명화 세팅을 적용합니다.

    순서:
      1) JSON 기반 vertex group import
      2) 0 alpha 분리 (nose hall / upper teeth root / throat)
      3) face 전체 alpha 적용
      4) mouth 특정 group alpha 적용
    """
    warnings: List[str] = []
    separated_objects: List[str] = []

    data = _load_group_json(json_path)
    imported_groups = import_vertex_groups_from_json(data, warnings=warnings, verbose=verbose)

    # 1) 0-alpha 분리 phase
    for obj_name, vg_name in zero_alpha_groups:
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            _warn(f"Object '{obj_name}' not found for zero-alpha split.", warnings, verbose)
            continue

        separated_name = separate_vg_to_object_with_zero_alpha(
            obj=obj,
            vg_name=vg_name,
            warnings=warnings,
            verbose=verbose,
        )
        if separated_name:
            separated_objects.append(separated_name)

    # 2) face 전체 alpha
    face = bpy.data.objects.get(face_obj)
    if face is None:
        _warn(f"Face object '{face_obj}' not found.", warnings, verbose)
    else:
        face_slot = ensure_alpha_material(face, alpha_front, "ALPHA_FRONT")
        assign_alpha_entire(face, face_slot, verbose=verbose)
        _log(f"Face alpha applied: {face_obj} -> {alpha_front}", verbose)

    # 3) mouth group alpha
    mouth = bpy.data.objects.get(mouth_obj)
    if mouth is None:
        _warn(f"Mouth object '{mouth_obj}' not found.", warnings, verbose)
    else:
        mouth_slot = ensure_alpha_material(mouth, alpha_side, "ALPHA_SIDE")
        for obj_name, vg_name in side_alpha_groups:
            target_obj = bpy.data.objects.get(obj_name)
            if target_obj is None:
                _warn(f"Object '{obj_name}' not found for side alpha.", warnings, verbose)
                continue

            if target_obj == mouth:
                target_slot = mouth_slot
            else:
                target_slot = ensure_alpha_material(target_obj, alpha_side, "ALPHA_SIDE")

            assign_alpha_to_vg(
                obj=target_obj,
                vg_name=vg_name,
                slot_index=target_slot,
                warnings=warnings,
                verbose=verbose,
            )

        _log(f"Side alpha applied: {mouth_obj} -> {alpha_side}", verbose)

    return ComfortVizResult(
        json_path=json_path,
        imported_groups=imported_groups,
        separated_objects=separated_objects,
        warnings=warnings,
    )


def setup_white_render(
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    video_bitrate: int = 8000,
    verbose: bool = True,
) -> None:
    """
    Blender UI에서 쓰던 white background / MP4 렌더 세팅을 함수화한 선택 기능.
    demo 쪽에서는 기본적으로 호출하지 않지만, 필요하면 별도 호출 가능합니다.
    """
    scene = bpy.context.scene

    # 1) Color Management
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    _log("Color Management set to Standard", verbose)

    # 2) World Background
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")

    world = scene.world
    world.use_nodes = True

    bg = None
    for node in world.node_tree.nodes:
        if node.type == "BACKGROUND":
            bg = node
            break

    if bg is not None:
        bg.inputs["Color"].default_value = (1, 1, 1, 1)
        bg.inputs["Strength"].default_value = 1.0

    # 3) Film Transparent OFF
    scene.render.film_transparent = False
    _log("World background set to white", verbose)

    # 4) Optional MP4 output
    if output_dir and output_name:
        os.makedirs(output_dir, exist_ok=True)
        scene.render.filepath = os.path.join(output_dir, output_name)
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "HIGH"
        scene.render.ffmpeg.ffmpeg_preset = "GOOD"
        scene.render.ffmpeg.video_bitrate = video_bitrate
        _log(f"FFMPEG output set: {scene.render.filepath}", verbose)
