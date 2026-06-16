#!/usr/bin/env python3
"""Fast-oriented Blender render wrapper.

Renders a .blend beside itself as white-background MP4 and optionally muxes WAV audio.
Compared with the previous version, this one focuses on reducing avoidable overhead:
- defaults to keeping the engine saved in the .blend
- enables persistent render data by default
- can auto-disable compositor / sequencer when they appear unused
- exposes optional Eevee sample / resolution overrides for faster renders
- defaults Blender FFmpeg preset to REALTIME for faster encoding
- still muxes WAV with ffmpeg stream-copy when available
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional
from imageio_ffmpeg import get_ffmpeg_exe

BLENDER_JOB_SCRIPT = textwrap.dedent(
    r'''
    import argparse
    import os
    import sys
    import bpy


    def safe_set_attr(obj, attr_name, value):
        try:
            setattr(obj, attr_name, value)
            return True
        except Exception as exc:
            print(f"[WARN] Failed to set {obj.__class__.__name__}.{attr_name}={value}: {exc}")
            return False


    def safe_get_attr(obj, attr_name, default=None):
        try:
            return getattr(obj, attr_name)
        except Exception:
            return default


    def safe_set_enum(obj, attr_name, value):
        try:
            setattr(obj, attr_name, value)
            return True
        except Exception as exc:
            print(f"[WARN] Failed to set enum {attr_name}={value}: {exc}")
            return False


    def ensure_white_world(scene):
        if scene.world is None:
            scene.world = bpy.data.worlds.new("World")

        world = scene.world
        world.use_nodes = True

        nodes = world.node_tree.nodes
        links = world.node_tree.links

        bg = None
        out = None
        for node in nodes:
            if node.type == "BACKGROUND":
                bg = node
            elif node.type == "OUTPUT_WORLD":
                out = node

        if bg is None:
            bg = nodes.new("ShaderNodeBackground")
        if out is None:
            out = nodes.new("ShaderNodeOutputWorld")

        try:
            surface_input = out.inputs.get("Surface")
            if surface_input is not None:
                for link in list(surface_input.links):
                    links.remove(link)
                links.new(bg.outputs["Background"], surface_input)
        except Exception as exc:
            print(f"[WARN] Failed to rewire world nodes: {exc}")

        bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs["Strength"].default_value = 1.0


    def is_headless_env():
        # DISPLAY가 없을 때만 headless (Dockerfile Xvfb가 :99를 제공하면 EEVEE 사용 가능)
        return not bool(os.environ.get("DISPLAY"))

    def choose_engine(scene, requested):
        available = list(scene.render.bl_rna.properties["engine"].enum_items.keys())
        current = scene.render.engine
        eevee_engines = {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"}

        def pick_headless_engine():
            for candidate in ("CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
                if candidate in available:
                    print(f"[INFO] Headless/Docker detected, using {candidate}")
                    return candidate
            return available[0]

        if requested == "keep":
            if is_headless_env() and current in eevee_engines and "CYCLES" in available:
                print(f"[WARN] {current} is unreliable without OpenGL; using CYCLES")
                return "CYCLES"
            return current if current in available else available[0]

        if requested == "auto":
            if is_headless_env():
                return pick_headless_engine()
            if current in available:
                return current
            for candidate in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"):
                if candidate in available:
                    return candidate
            return available[0]

        if requested in eevee_engines and is_headless_env() and "CYCLES" in available:
            print(f"[WARN] {requested} requires OpenGL; falling back to CYCLES")
            return "CYCLES"

        if requested in available:
            return requested

        for candidate in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"):
            if candidate in available:
                return candidate
        return available[0]

    def configure_cycles_gpu(scene):
        if scene.render.engine != "CYCLES":
            return

        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
        except Exception as exc:
            print(f"[WARN] Cycles preferences unavailable: {exc}")
            scene.cycles.device = "CPU"
            return

        for device_type in ("CUDA", "OPTIX", "HIP", "ONEAPI"):
            try:
                prefs.compute_device_type = device_type
                prefs.get_devices()
                enabled = False
                for device in prefs.devices:
                    use_device = device.type == device_type
                    device.use = use_device
                    enabled = enabled or use_device
                if enabled:
                    scene.cycles.device = "GPU"
                    print(f"[INFO] Cycles GPU enabled via {device_type}")
                    return
            except Exception as exc:
                print(f"[WARN] Cycles {device_type} setup failed: {exc}")

        scene.cycles.device = "CPU"
        print("[INFO] Cycles falling back to CPU")


    def compositor_seems_unused(scene):
        try:
            if not scene.use_nodes or scene.node_tree is None:
                return True
            meaningful_types = {
                node.type
                for node in scene.node_tree.nodes
                if node.type not in {"R_LAYERS", "COMPOSITE", "VIEWER"}
            }
            return len(meaningful_types) == 0
        except Exception:
            return False


    def sequencer_seems_unused(scene):
        try:
            seq = scene.sequence_editor
            if seq is None:
                return True
            sequences = getattr(seq, "sequences_all", None)
            if sequences is None:
                return True
            return len(sequences) == 0
        except Exception:
            return False


    def set_eevee_render_samples(scene, value):
        if value is None:
            return
        eevee = safe_get_attr(scene, "eevee")
        if eevee is None:
            return
        for attr_name in ("taa_render_samples", "render_samples"):
            if hasattr(eevee, attr_name):
                safe_set_attr(eevee, attr_name, int(value))
                print(f"[INFO] Eevee render samples -> {value} via {attr_name}")
                return
        print("[WARN] Could not find Eevee render-samples property")


    def maybe_set_eevee_bool(scene, attr_name, enabled):
        if enabled is None:
            return
        eevee = safe_get_attr(scene, "eevee")
        if eevee is None:
            return
        if hasattr(eevee, attr_name):
            safe_set_attr(eevee, attr_name, bool(enabled))
            print(f"[INFO] Eevee {attr_name} -> {bool(enabled)}")


    def maybe_set_resolution_percentage(scene, value):
        if value is None:
            return
        value = max(1, min(100, int(value)))
        scene.render.resolution_percentage = value
        print(f"[INFO] resolution_percentage -> {value}")


    def parse_args(argv):
        parser = argparse.ArgumentParser(description="Internal Blender render job")
        parser.add_argument("--output", required=True)
        parser.add_argument("--engine", default="keep")
        parser.add_argument("--video-bitrate", type=int, default=8000)
        parser.add_argument("--audio-codec", default="AAC")
        parser.add_argument("--crf", default="HIGH")
        parser.add_argument("--preset", default="REALTIME")
        parser.add_argument("--fps", type=int, default=None)
        parser.add_argument("--frame-start", type=int, default=None)
        parser.add_argument("--frame-end", type=int, default=None)
        parser.add_argument("--save-blend", action="store_true")
        parser.add_argument("--persistent-data", action="store_true")
        parser.add_argument("--no-persistent-data", action="store_true")
        parser.add_argument("--compositor-mode", choices=["auto", "on", "off"], default="auto")
        parser.add_argument("--sequencer-mode", choices=["auto", "on", "off"], default="auto")
        parser.add_argument("--resolution-percentage", type=int, default=None)
        parser.add_argument("--eevee-render-samples", type=int, default=None)
        parser.add_argument("--eevee-use-shadows", choices=["on", "off"], default=None)
        parser.add_argument("--eevee-use-raytracing", choices=["on", "off"], default=None)
        return parser.parse_args(argv)


    def main():
        argv = sys.argv
        if "--" in argv:
            argv = argv[argv.index("--") + 1 :]
        else:
            argv = []

        args = parse_args(argv)
        scene = bpy.context.scene

        render_engine = choose_engine(scene, args.engine)
        if scene.render.engine != render_engine:
            scene.render.engine = render_engine
        print(f"[INFO] Render engine: {scene.render.engine}")
        configure_cycles_gpu(scene)

        if args.fps is not None:
            scene.render.fps = args.fps
        if args.frame_start is not None:
            scene.frame_start = args.frame_start
        if args.frame_end is not None:
            scene.frame_end = args.frame_end

        safe_set_enum(scene.view_settings, "view_transform", "Standard")
        safe_set_enum(scene.view_settings, "look", "None")
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0

        ensure_white_world(scene)
        scene.render.film_transparent = False

        if args.no_persistent_data:
            safe_set_attr(scene.render, "use_persistent_data", False)
        elif args.persistent_data:
            safe_set_attr(scene.render, "use_persistent_data", True)

        if args.compositor_mode == "off":
            safe_set_attr(scene.render, "use_compositing", False)
            print("[INFO] Compositor disabled by request")
        elif args.compositor_mode == "on":
            safe_set_attr(scene.render, "use_compositing", True)
        elif compositor_seems_unused(scene):
            safe_set_attr(scene.render, "use_compositing", False)
            print("[INFO] Compositor auto-disabled (appears unused)")

        if args.sequencer_mode == "off":
            safe_set_attr(scene.render, "use_sequencer", False)
            print("[INFO] Sequencer disabled by request")
        elif args.sequencer_mode == "on":
            safe_set_attr(scene.render, "use_sequencer", True)
        elif sequencer_seems_unused(scene):
            safe_set_attr(scene.render, "use_sequencer", False)
            print("[INFO] Sequencer auto-disabled (appears unused)")

        maybe_set_resolution_percentage(scene, args.resolution_percentage)
        set_eevee_render_samples(scene, args.eevee_render_samples)
        maybe_set_eevee_bool(scene, "use_shadows", None if args.eevee_use_shadows is None else (args.eevee_use_shadows == "on"))
        maybe_set_eevee_bool(scene, "use_raytracing", None if args.eevee_use_raytracing is None else (args.eevee_use_raytracing == "on"))

        scene.render.use_file_extension = True
        scene.render.use_overwrite = True
        scene.render.use_placeholder = False
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.audio_codec = args.audio_codec
        scene.render.ffmpeg.constant_rate_factor = args.crf
        scene.render.ffmpeg.ffmpeg_preset = args.preset
        scene.render.ffmpeg.video_bitrate = args.video_bitrate
        scene.render.filepath = args.output

        print(f"[INFO] frame range: {scene.frame_start}..{scene.frame_end}")
        print(f"[INFO] fps: {scene.render.fps}")
        print(f"[INFO] resolution: {scene.render.resolution_x}x{scene.render.resolution_y} @ {scene.render.resolution_percentage}%")
        print(f"[INFO] output: {scene.render.filepath}")
        print(f"[INFO] persistent_data: {safe_get_attr(scene.render, 'use_persistent_data', None)}")
        print(f"[INFO] use_compositing: {safe_get_attr(scene.render, 'use_compositing', None)}")
        print(f"[INFO] use_sequencer: {safe_get_attr(scene.render, 'use_sequencer', None)}")

        if args.save_blend:
            try:
                bpy.ops.wm.save_mainfile()
                print("[INFO] Saved .blend with updated render settings")
            except Exception as exc:
                print(f"[WARN] Failed to save .blend before render: {exc}")

        bpy.ops.render.render(animation=True)
        print("[DONE] Render finished")


    if __name__ == "__main__":
        main()
    '''
)


class BlendRenderError(RuntimeError):
    pass


def str2bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast-oriented: render a .blend beside itself as .mp4, then optionally mux WAV audio."
    )
    parser.add_argument("--blend", required=True, help="Path to input .blend")
    parser.add_argument("--wav", default=None, help="Optional WAV to mux into the MP4")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output MP4 path. Default: same directory/name as .blend",
    )
    parser.add_argument(
        "--blender-bin",
        default=os.environ.get("BLENDER_BIN") or shutil.which("blender"),
        help="Blender executable path. Can also be supplied via BLENDER_BIN env var.",
    )
    parser.add_argument(
        "--engine",
        default="keep",
        choices=["auto", "keep", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"],
        help="Render engine selection inside Blender. 'keep' is fastest/safest to match the saved .blend.",
    )
    parser.add_argument("--fps", type=int, default=None, help="Optional FPS override")
    parser.add_argument("--frame-start", type=int, default=None, help="Optional frame start override")
    parser.add_argument("--frame-end", type=int, default=None, help="Optional frame end override")
    parser.add_argument("--video-bitrate", type=int, default=8000, help="FFmpeg video bitrate (kb/s)")
    parser.add_argument(
        "--audio-bitrate",
        default="192k",
        help="Audio bitrate used when muxing WAV with ffmpeg/moviepy",
    )
    parser.add_argument(
        "--muxer",
        default="auto",
        choices=["auto", "ffmpeg", "moviepy", "none"],
        help="Audio muxing backend selection. 'none' disables audio muxing.",
    )
    parser.add_argument(
        "--crf",
        default="HIGH",
        choices=["NONE", "LOSSLESS", "PERC_LOSSLESS", "HIGH", "MEDIUM", "LOW", "VERYLOW"],
        help="Blender FFmpeg constant rate factor",
    )
    parser.add_argument(
        "--preset",
        default="REALTIME",
        choices=["BEST", "GOOD", "REALTIME"],
        help="Blender FFmpeg preset. REALTIME is usually faster than GOOD.",
    )

    parser.add_argument(
        "--keep-video-only",
        type=str2bool,
        default=False,
        help="Keep the temporary silent MP4 used before audio muxing.",
    )
    parser.add_argument(
        "--save-blend",
        type=str2bool,
        default=False,
        help="Save the .blend after applying white-background/render settings.",
    )
    parser.add_argument(
        "--overwrite",
        type=str2bool,
        default=True,
        help="Overwrite existing output MP4 if it exists.",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="Print subprocess commands and progress.",
    )

    # Speed-oriented options.
    parser.add_argument(
        "--persistent-data",
        type=str2bool,
        default=True,
        help="Enable persistent render data cache inside Blender.",
    )
    parser.add_argument(
        "--compositor-mode",
        default="auto",
        choices=["auto", "on", "off"],
        help="auto: disable compositor only when it appears unused.",
    )
    parser.add_argument(
        "--sequencer-mode",
        default="auto",
        choices=["auto", "on", "off"],
        help="auto: disable sequencer only when it appears unused.",
    )
    parser.add_argument(
        "--resolution-percentage",
        type=int,
        default=None,
        help="Optional render scale override (1-100). Lower is faster.",
    )
    parser.add_argument(
        "--eevee-render-samples",
        type=int,
        default=None,
        help="Optional Eevee render-samples override. Lower is faster.",
    )
    parser.add_argument(
        "--eevee-use-shadows",
        default=None,
        choices=["on", "off"],
        help="Optional Eevee shadow toggle override.",
    )
    parser.add_argument(
        "--eevee-use-raytracing",
        default=None,
        choices=["on", "off"],
        help="Optional Eevee raytracing toggle override.",
    )
    return parser.parse_args(argv)


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Optional[Path], Path, Path]:
    blend_path = Path(args.blend).expanduser().resolve()
    if not blend_path.is_file():
        raise FileNotFoundError(f".blend file not found: {blend_path}")

    wav_path = None
    if args.wav:
        wav_path = Path(args.wav).expanduser().resolve()
        if not wav_path.is_file():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

    output_path = Path(args.output).expanduser().resolve() if args.output else blend_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_only_path = output_path.with_name(output_path.stem + ".__video_only__.mp4")
    return blend_path, wav_path, output_path, video_only_path


def write_temp_blender_script() -> str:
    fd, script_path = tempfile.mkstemp(prefix="blend_render_fast_", suffix=".py")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(BLENDER_JOB_SCRIPT)
    return script_path


# def run_command(cmd: list[str], verbose: bool = True) -> None:
#     log("[CMD] " + " ".join(cmd), verbose)
#     subprocess.run(cmd, check=True)

def build_blender_render_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("NVIDIA_VISIBLE_DEVICES", "all")
    env.setdefault("NVIDIA_DRIVER_CAPABILITIES", "compute,utility,graphics")

    nvidia_egl = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if os.path.isfile(nvidia_egl):
        env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
        if not env.get("DISPLAY"):
            # Xvfb 없는 순수 headless: EGL surfaceless
            env.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", nvidia_egl)
            env.setdefault("EGL_PLATFORM", "surfaceless")

    return env


def run_command(cmd: list[str], verbose: bool = True) -> None:
    env = build_blender_render_env()
    if verbose:
        log("[CMD] " + " ".join(cmd), verbose)
    subprocess.run(cmd, check=True, env=env)

def run_blender_render(
    blender_bin: str,
    blend_path: Path,
    video_only_path: Path,
    args: argparse.Namespace,
) -> None:
    if not blender_bin:
        raise BlendRenderError("Blender executable not found. Pass --blender-bin or set BLENDER_BIN.")

    script_path = write_temp_blender_script()
    try:
        cmd = [
            blender_bin,
            "-b",
            str(blend_path),
            "-P",
            script_path,
            "--",
            "--output",
            str(video_only_path),
            "--engine",
            args.engine,
            "--video-bitrate",
            str(args.video_bitrate),
            "--crf",
            args.crf,
            "--preset",
            args.preset,
            "--compositor-mode",
            args.compositor_mode,
            "--sequencer-mode",
            args.sequencer_mode,
        ]

        if args.fps is not None:
            cmd.extend(["--fps", str(args.fps)])
        if args.frame_start is not None:
            cmd.extend(["--frame-start", str(args.frame_start)])
        if args.frame_end is not None:
            cmd.extend(["--frame-end", str(args.frame_end)])
        if args.save_blend:
            cmd.append("--save-blend")
        if args.persistent_data:
            cmd.append("--persistent-data")
        else:
            cmd.append("--no-persistent-data")
        if args.resolution_percentage is not None:
            cmd.extend(["--resolution-percentage", str(args.resolution_percentage)])
        if args.eevee_render_samples is not None:
            cmd.extend(["--eevee-render-samples", str(args.eevee_render_samples)])
        if args.eevee_use_shadows is not None:
            cmd.extend(["--eevee-use-shadows", args.eevee_use_shadows])
        if args.eevee_use_raytracing is not None:
            cmd.extend(["--eevee-use-raytracing", args.eevee_use_raytracing])

        run_command(cmd, verbose=args.verbose)
    finally:
        try:
            os.remove(script_path)
        except OSError:
            pass

    if not video_only_path.is_file():
        raise BlendRenderError(f"Render finished but output not found: {video_only_path}")



def mux_with_ffmpeg(
    video_only_path: Path,
    wav_path: Path,
    output_path: Path,
    overwrite: bool,
    verbose: bool,
):

    ffmpeg_path = get_ffmpeg_exe()

    cmd = [
        ffmpeg_path,
        "-y" if overwrite else "-n",
        "-i", str(video_only_path),
        "-i", str(wav_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]

    if verbose:
        print("[FFMPEG]", " ".join(cmd))

    subprocess.run(cmd, check=True)

def attach_audio(
    video_only_path: Path,
    wav_path: Optional[Path],
    output_path: Path,
    args: argparse.Namespace,
) -> None:

    if wav_path is None or args.muxer == "none":
        if output_path.exists() and args.overwrite:
            output_path.unlink()
        shutil.move(str(video_only_path), str(output_path))
        return

    mux_with_ffmpeg(
        video_only_path=video_only_path,
        wav_path=wav_path,
        output_path=output_path,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    blend_path, wav_path, output_path, video_only_path = resolve_paths(args)

    log(f"[INFO] .blend : {blend_path}", args.verbose)
    log(f"[INFO] .wav   : {wav_path if wav_path else '(none)'}", args.verbose)
    log(f"[INFO] output : {output_path}", args.verbose)
    log(f"[INFO] blender: {args.blender_bin}", args.verbose)

    if output_path.exists() and not args.overwrite:
        raise BlendRenderError(f"Output already exists and overwrite is disabled: {output_path}")

    if video_only_path.exists():
        video_only_path.unlink()

    run_blender_render(
        blender_bin=args.blender_bin,
        blend_path=blend_path,
        video_only_path=video_only_path,
        args=args,
    )

    attach_audio(
        video_only_path=video_only_path,
        wav_path=wav_path,
        output_path=output_path,
        args=args,
    )

    if video_only_path.exists() and not args.keep_video_only:
        try:
            video_only_path.unlink()
        except OSError:
            pass

    log(f"[DONE] Final MP4: {output_path}", args.verbose)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\n[ABORT] Interrupted by user", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
