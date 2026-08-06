
# Script by pifroggi https://github.com/pifroggi/vs_degrade
# or tepete and pifroggi on Discord

import numpy as np
import random, re
import subprocess, threading, os, sys
import shlex, shutil, ctypes, ctypes.util
import vapoursynth as vs
from pathlib import Path

core = vs.core


def find_ffmpeg():
    found = shutil.which("ffmpeg")
    if found:
        return str(Path(found).resolve())

    here  = Path(__file__).resolve().parent
    bases = (here, here.parent, Path(sys.executable).resolve().parent)
    subs  = ("", "bin", "tools", "ffmpeg", "plugins", "vs-plugins", "scripts", "vs-scripts", "vapoursynth", "vapoursynth/plugins")
    names = ("ffmpeg.exe", "ffmpeg") if os.name == "nt" else ("ffmpeg",)

    for base in bases:
        for subdirectory in subs:
            directory = base / subdirectory

            if not directory.is_dir():
                continue

            for name in names:
                candidate = directory / name

                if candidate.is_file() and (os.name == "nt" or os.access(candidate, os.X_OK)):
                    return str(candidate.resolve())

    raise FileNotFoundError("vs_degrade: FFmpeg not found on system PATH or near the vs_degrade package. Add path manually.")


def find_turbojpeg():
    name = "turbojpeg"
    if os.name == "nt":
        tried_names      = [name + ".dll"]
        system_locations = [Path("C:/libjpeg-turbo64/bin"),
                            Path("C:/libjpeg-turbo/bin")]
    elif sys.platform == "darwin":
        tried_names      = ["lib" + name + ".0.dylib", "lib" + name + ".dylib"]
        system_locations = [Path("/usr/local/lib"),
                            Path("/opt/homebrew/lib"),
                            Path("/opt/homebrew/opt/jpeg-turbo/lib"),
                            Path("/usr/local/opt/jpeg-turbo/lib"),
                            Path("/opt/libjpeg-turbo/lib64"),
                            Path("/opt/libjpeg-turbo/lib")]
    else:
        tried_names      = ["lib" + name + ".so.0", "lib" + name + ".so"]
        multiarch        = getattr(sys.implementation, "_multiarch", "")
        system_locations = [Path("/usr/local/lib"),
                            Path("/usr/local/lib64"),
                            Path("/usr/lib") / multiarch if multiarch else Path("/usr/lib"),
                            Path("/lib") / multiarch if multiarch else Path("/lib"),
                            Path("/usr/lib"),
                            Path("/usr/lib64"),
                            Path("/lib"),
                            Path("/lib64"),
                            Path("/opt/libjpeg-turbo/lib64"),
                            Path("/opt/libjpeg-turbo/lib")]

    # try dynamic linker
    hit = ctypes.util.find_library(name)
    if hit and (os.name != "nt" or Path(hit).name.lower() == name.lower() + ".dll"):
        return hit

    # try system PATH
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue

        for libname in tried_names:
            candidate = Path(directory) / libname
            if candidate.is_file():
                return str(candidate.resolve())

    # try common system library locations
    for directory in system_locations:
        if not directory.is_dir():
            continue

        for libname in tried_names:
            candidate = directory / libname
            if candidate.is_file():
                return str(candidate.resolve())

        if os.name != "nt":
            patterns = ["lib" + name + ".so.*"] if sys.platform != "darwin" else ["lib" + name + ".*.dylib"]
            for pattern in patterns:
                for candidate in sorted(directory.glob(pattern)):
                    if candidate.is_file():
                        return str(candidate.resolve())

    # try nearby locations
    here   = Path(__file__).resolve().parent
    bases  = (here, here.parent, Path(sys.executable).resolve().parent)
    subs   = ("", "bin", "lib", "lib64", "tools", "plugins", "vs-plugins", "scripts", "vs-scripts", "ffmpeg", "turbojpeg", "libjpeg-turbo", "vapoursynth", "vapoursynth/plugins")

    for base in bases:
        for sub in subs:
            search_path = base / sub
            if not search_path.is_dir():
                continue

            for libname in tried_names:
                candidate = search_path / libname
                if candidate.is_file():
                    return str(candidate.resolve())

            if os.name != "nt":
                patterns = ["lib" + name + ".so.*"] if sys.platform != "darwin" else ["lib" + name + ".*.dylib"]
                for pattern in patterns:
                    for candidate in sorted(search_path.glob(pattern)):
                        if candidate.is_file():
                            return str(candidate.resolve())

    raise FileNotFoundError(f"vs_degrade: Turbojpeg not found by dynamic linker, on system PATH, or near the vs_degrade package. Add path manually.")


def jpeg(clip, quality=50, fields=False, seed=0, planes=[0, 1, 2], path=None):
    """Degrades a YUV clip directly as is, without upsampling chroma or doing any format/color conversions, since Jpeg also works in YUV. Adds purely spatial compression artifacts very fast.
    Args:
        clip: Clip to degrade. Jpeg supports YUV420P8, YUV422P8, and YUV444P8 formats.
        quality: Image quality in the range 1-100 with 1 being the worst.
            Can be a constant value or randomized each frame by providing a range: `quality=[30, 80]`
        fields: Will separate the clip into fields, degrade each field, then put them back together.
            This creates interlacing artifacts like combing and more mosquito noise.
        seed: Seed used for quality randomization.
        planes: Which planes to degrade. Any unmentioned planes will simply be copied. If `None`, all planes will be degraded.
        path: Path to libjpeg-turbo (`turbojpeg.dll` on Windows, `libturbojpeg.so` on Linux), if not auto-detected.
    """
    
    from turbojpeg import TurboJPEG, TJSAMP_420, TJSAMP_422, TJSAMP_444
    
    # checks and settings
    turbojpeg_binary = os.path.abspath(path) if path else find_turbojpeg()
    turbojpeg_func   = TurboJPEG(lib_path=turbojpeg_binary)
    subsample_map    = {(1, 1): TJSAMP_420,
                        (1, 0): TJSAMP_422,
                        (0, 0): TJSAMP_444}
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_degrade.jpeg: Clip must be a vapoursynth clip.")
    if clip.format.id == vs.PresetVideoFormat.NONE or clip.width == 0 or clip.height == 0:
        raise TypeError("vs_degrade.jpeg: Clip must have constant format and dimensions.")
    if clip.format.color_family != vs.YUV:
        raise ValueError("vs_degrade.jpeg: Jpeg works in YUV. This expects the input clip to be YUV already.")
    if not isinstance(seed, int):
        raise TypeError("vs_degrade.jpeg: Seed must be an integer.")
    
    orig_clip      = clip
    clip_format    = clip.format
    num_planes     = clip_format.num_planes
    subsampling    = (clip_format.subsampling_w, clip_format.subsampling_h)
    subsampling_w  = 1 << clip_format.subsampling_w # 1 for 444, 2 for 422/420
    subsampling_h  = 1 << clip_format.subsampling_h # 1 for 444/422, 2 for 420
    
    if clip.format.bits_per_sample != 8:
        raise ValueError("vs_degrade.jpeg: Jpeg only supports 8-bit.")
    if subsampling not in subsample_map:
        raise ValueError("vs_degrade.jpeg: Jpeg only supports 420, 422, and 444 subsampling.")
    if planes is None:
        planes = list(range(num_planes))
    if isinstance(planes, int):
        planes = [planes]
    if num_planes == 1:
        planes = [0]
    if not set(planes) <= set(range(num_planes)):
        raise ValueError("vs_degrade.jpeg: Invalid plane index specified.")
    
    jpeg_subsample = subsample_map[subsampling]     # turbojpeg enum
    base_seed      = seed & 0xFFFFFFFFFFFFFFFF

    # constant quality, or range of quality
    if isinstance(quality, (list, tuple)):
        if len(quality) != 2:
            raise ValueError("vs_degrade.jpeg: Quality must be an array of two value representing a range, or a single value.")
        qmin, qmax = map(int, quality)
        if not (1 <= qmin <= qmax <= 100):
            raise ValueError("vs_degrade.jpeg: Quality values must be in the range 1-100 and the first value must be smaller than the second.")
        quality_range = (qmin, qmax)
        constant_q = False
    else:
        if not (1 <= quality <= 100):
            raise ValueError("vs_degrade.jpeg: Quality must be in the range 1-100.")
        q_fixed = int(quality)
        constant_q = True

    def _frame_random(n):
        # makes every frame deterministic
        value = (base_seed + n + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        value = value ^ (value >> 31)
        return random.Random(value)

    def _degrade_jpeg(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        # optionally randomize quality
        quality = q_fixed if constant_q else _frame_random(n).randint(*quality_range)
    
        # convert planes to arrays
        Y = np.asarray(f[0])
        U = np.asarray(f[1])
        V = np.asarray(f[2])
        h, w = Y.shape

        # pack YUV into one contiguous buffer
        yuv_buf = np.concatenate((Y.ravel(), U.ravel(), V.ravel()))

        # encode to jpeg
        encoded = turbojpeg_func.encode_from_yuv(yuv_buf, height=h, width=w, quality=quality, jpeg_subsample=jpeg_subsample, align=1)

        # decode back to raw planes
        Yd, Ud, Vd = turbojpeg_func.decode_to_yuv_planes(encoded)

        # remove padding
        Yd = Yd[:, :w]
        Ud = Ud[:, :w // subsampling_w]
        Vd = Vd[:, :w // subsampling_w]

        # output
        out = f.copy()
        if 0 in planes:
            np.copyto(np.asarray(out[0]), Yd)
        if 1 in planes:
            np.copyto(np.asarray(out[1]), Ud[:h // subsampling_h])
        if 2 in planes:
            np.copyto(np.asarray(out[2]), Vd[:h // subsampling_h])
        return out
    
    # optionally degrade each field seperately
    if fields:
        clip = core.std.SetFieldBased(clip, 2)
        clip = core.std.SeparateFields(clip, tff=True)
        clip = core.std.ModifyFrame(clip, clip, selector=_degrade_jpeg)
        clip = core.std.DoubleWeave(clip, tff=True)
        clip = core.std.SelectEvery(clip, 2, 0)
        return core.std.CopyFrameProps(clip, orig_clip)
    else:
        return core.std.ModifyFrame(clip, clip, selector=_degrade_jpeg)


def ffmpeg(clip, temp_window=10, args="-c:v mpeg2video -q:v 10", fields=False, seed=0, planes=[0, 1, 2], path=None):
    """Runs randomizable FFmpeg commands in chunks directly on a YUV clip as is, without upsampling chroma or doing any format/color conversions. Adds spatial and temporal compression artifacts.
    Args:
        clip: Clip to degrade. Supports YUV420P8/P10, YUV422P8/P10, YUV444P8/P10 formats.
        temp_window: Temporal window length. The amount of frames to encode at once.
        args: The video encoding arguments of an FFmpeg command. Arguments can optionally be randomized per temporal window:
                * `{rand(5,50)}` sets randomizer range for int values.
                * `{randf(-0.5,0.9)}` sets randomizer range for float values.
                * `{choice(veryfast,medium,veryslow)}` chooses randomly from a list.
            Multiple full commands can be randomized per temporal window by providing a list. FFmpeg filters can also be applied.
        fields: Will seperate the clip into fields, degrade with FFmpeg, then put them back together.
            This creates interlacing artifacts like combing and more mosquito noise.
        seed: Seed used for ffmpeg args randomization.
        planes: Which planes to degrade. Any unmentioned planes will simply be copied. If `None`, all planes will be degraded.
        path: Path to FFmpeg (`ffmpeg.exe` on Windows, just `ffmpeg` on Linux), if not auto-detected.
    """

    # fix error handling
    import signal
    try:                                 # ignore SIGPIPE so python gets BrokenPipeError instead if encoding fails
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    except (AttributeError, ValueError): # not needed on windows
        pass

    # checks and settings
    ffmpeg_binary = os.path.abspath(path) if path else find_ffmpeg()
    subsample_map = {vs.YUV420P8:  "yuv420p",
                     vs.YUV422P8:  "yuv422p",
                     vs.YUV444P8:  "yuv444p",
                     vs.YUV420P10: "yuv420p10le",
                     vs.YUV422P10: "yuv422p10le",
                     vs.YUV444P10: "yuv444p10le"}

    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_degrade.ffmpeg: Clip must be a vapoursynth clip.")
    if clip.format.id == vs.PresetVideoFormat.NONE or clip.width == 0 or clip.height == 0:
        raise TypeError("vs_degrade.ffmpeg: Clip must have constant format and dimensions.")
    if clip.format.bits_per_sample not in [8, 10]:
        raise ValueError("vs_degrade.ffmpeg: Only 8-bit and 10-Bit encoding is supported for now.")
    if clip.format.id not in [vs.YUV420P8, vs.YUV422P8, vs.YUV444P8, vs.YUV420P10, vs.YUV422P10, vs.YUV444P10]:
        raise ValueError("vs_degrade.ffmpeg: Only YUV420P8/P10, YUV422P8/P10, and YUV444P8/P10 encoding is supported for now.")
    if not isinstance(temp_window, int):
        raise TypeError("vs_degrade.ffmpeg: Temporal window length must be an integer.")
    if temp_window < 1:
        raise ValueError("vs_degrade.ffmpeg: Temporal window length must be at least 1.")
    if not isinstance(seed, int):
        raise TypeError("vs_degrade.ffmpeg: Seed must be an integer.")

    orig_clip   = clip
    clip_format = clip.format
    num_planes  = clip_format.num_planes
    bps         = clip_format.bytes_per_sample # 1 for 8bit, 2 for 10bit padded
    pixfmt      = subsample_map[clip_format.id]
    base_seed   = seed & 0xFFFFFFFFFFFFFFFF

    if planes is None:
        planes = list(range(num_planes))
    if isinstance(planes, int):
        planes = [planes]
    if num_planes == 1:
        planes = [0]
    if not set(planes) <= set(range(num_planes)):
        raise ValueError("vs_degrade.ffmpeg: Invalid plane index specified.")

    if fields:
        temp_window = temp_window * 2
        clip        = core.std.SetFieldBased(clip, 2)
        clip        = core.std.SeparateFields(clip, tff=True)

    num_frames = clip.num_frames
    w, h       = clip.width, clip.height
    cw, ch     = w >> clip_format.subsampling_w, h >> clip_format.subsampling_h
    frame_size = (w*h + 2*cw*ch) * bps

    if bps == 1:
        dtype = np.uint8
    else:
        dtype = np.dtype("<u2") # uint16 but little-endian just to make sure

    # build tokenized templates
    if isinstance(args, str):
        templates = [shlex.split(args)]
    elif isinstance(args, (list, tuple)) and all(isinstance(t, str) for t in args):
        templates = [shlex.split(t) for t in args]
    else:
        raise TypeError("vs_degrade.ffmpeg: Args must be a string or a list of strings.")

    # patterns for {rand(a,b)} and {randf(a,b)} and {choice(a,b)}
    _rand_int = re.compile(r"{rand\((-?\d+),\s*(-?\d+)\)}")
    _rand_flt = re.compile(r"{randf\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)}")
    _rand_cho = re.compile(r"{choice\(([^{}()]*)\)}")

    def _get_window(window_clip, window_size):
        if window_size == 1:
            return [window_clip]
        
        # pad to a multiple of window_size so all offset clips have the same length
        pad = (-window_clip.num_frames) % window_size
        if pad:
            pad_clip    = core.std.BlankClip(clip=window_clip, length=pad)
            window_clip = core.std.Splice([window_clip, pad_clip])

        # offset_clips[i] contains frames i, i+window_size, i+2*window_size, ...
        return [core.std.SelectEvery(window_clip[i:], cycle=window_size, offsets=[0]) for i in range(window_size)]

    def _window_random(window_start):
        # randomizes deterministicly
        value = (base_seed + window_start + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        value = value ^ (value >> 31)
        return random.Random(value)

    def _randomize_values(token, window_random):
        # replaces {rand(a,b)} and {randf(a,b)} and {choice(a,b)}
        def _r_int(m):
            return str(window_random.randint(int(m[1]), int(m[2])))
        def _r_flt(m):
            return "{:.6g}".format(window_random.uniform(float(m[1]), float(m[2])))
        def _r_cho(m):
            opts = [o.strip() for o in m[1].split(',')]
            return window_random.choice(opts)

        token = _rand_cho.sub(_r_cho, token)  # first so text is not intepreted as numbers
        token = _rand_int.sub(_r_int, token)
        token = _rand_flt.sub(_r_flt, token)
        return token

    def _replace_placeholders(token, window_start):
        # replaces the placeholders
        return token.replace("{w}", str(w)).replace("{h}", str(h)).replace("{pixfmt}", pixfmt).replace("{n}", str(window_start))

    def _read_stream(pipe, sink):
        # background reader that drains pipe into a bytearray
        with pipe:
            for chunk in iter(lambda: pipe.read(65536), b''):
                sink.extend(chunk)

    def _encode_window(window_start, source_frames):
        cur_window    = len(source_frames)
        window_random = _window_random(window_start)

        # pick random args template
        template_tokens = window_random.choice(templates)
        tokens = [_replace_placeholders(_randomize_values(token, window_random), window_start) for token in template_tokens]

        # encode/decode commands
        enc_cmd = ([ffmpeg_binary, "-loglevel", "error", "-xerror", "-f", "rawvideo", "-pix_fmt", pixfmt, "-s", f"{w}x{h}", "-i", "-", "-pix_fmt", f"+{pixfmt}"] + tokens + ["-f", "nut", "-"])
        dec_cmd = ([ffmpeg_binary, "-loglevel", "error", "-xerror", "-i", "-", "-f", "rawvideo", "-pix_fmt", pixfmt, "-"])

        # create encoder and decoder subprocess
        enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        try:
            dec = subprocess.Popen(dec_cmd, stdin=enc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        except Exception:
            enc.stdin.close()
            enc.stdout.close()
            enc.kill()
            enc.wait()
            raise
        enc.stdout.close()

        # prepare buffers on separate threads that drain all subprocess output pipes
        outbuf  = bytearray()
        enc_err = bytearray()
        dec_err = bytearray()
        readers = [threading.Thread(target=_read_stream, args=(dec.stdout, outbuf)),
                   threading.Thread(target=_read_stream, args=(enc.stderr, enc_err)),
                   threading.Thread(target=_read_stream, args=(dec.stderr, dec_err))]
        for reader in readers:
            reader.start()

        # feed a window of frames into the encoder
        feed_error = None
        try:
            for rf in source_frames:
                enc.stdin.write(np.asarray(rf[0])[:, :w].tobytes())
                enc.stdin.write(np.asarray(rf[1])[:, :cw].tobytes())
                enc.stdin.write(np.asarray(rf[2])[:, :cw].tobytes())
        
        # error if ffmpeg fails
        except (BrokenPipeError, OSError) as e:
            feed_error = e
        finally:
            try:
                enc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            enc_return = enc.wait()
            dec_return = dec.wait()
            for reader in readers:
                reader.join()

        enc_err_txt = enc_err.decode(errors="replace")
        dec_err_txt = dec_err.decode(errors="replace")
        full_err    = "\n".join(part for part in [f"Encoder:\n{enc_err_txt}" if enc_err_txt else "",
                                                  f"Decoder:\n{dec_err_txt}" if dec_err_txt else ""] if part)
        expected    = cur_window * frame_size

        # error if ffmpeg fails or returns an unexpected amount of video
        if feed_error is not None or enc_return != 0 or dec_return != 0 or len(outbuf) != expected:
            hint = ""
            if ("do not have a common format" in full_err) or ("Specified pixel format" in full_err and "is invalid or not supported" in full_err):
                hint = "\nvs_degrade.ffmpeg: Format of the input clip is not supported by the chosen ffmpeg codec."
            size_error = "" if len(outbuf) == expected else f"\nDecoded output size: {len(outbuf)} bytes; expected {expected} bytes."
            raise RuntimeError(
                f"{hint}"
                f"\nvs_degrade.ffmpeg: Encoding failed with args '{' '.join(tokens)}\n\n'"
                f"Full FFmpeg Error:\n{full_err or '<empty>'}"
                f"\nEncoder return code: {enc_return}"
                f"\nDecoder return code: {dec_return}"
                f"{size_error}"
                ) from feed_error

        return outbuf

    def _degrade_ffmpeg(n, f):
        window_start  = n * temp_window
        cur_window    = min(temp_window, num_frames - window_start)
        source_frames = f[1:cur_window + 1]
        outbuf        = _encode_window(window_start, source_frames)
        out           = f[0].copy()

        # copy decoded frames next to each other into one wide output frame
        for i, rf in enumerate(source_frames):
            frame_offset = i * frame_size
            Y_offset     = frame_offset
            U_offset     = Y_offset + w*h*bps
            V_offset     = U_offset + cw*ch*bps
            Y = np.frombuffer(outbuf, dtype=dtype, count=w*h,   offset=Y_offset).reshape((h,  w))
            U = np.frombuffer(outbuf, dtype=dtype, count=cw*ch, offset=U_offset).reshape((ch, cw))
            V = np.frombuffer(outbuf, dtype=dtype, count=cw*ch, offset=V_offset).reshape((ch, cw))

            np.copyto(np.asarray(out[0])[:, i*w:(i+1)*w],   Y if 0 in planes else np.asarray(rf[0])[:, :w])
            np.copyto(np.asarray(out[1])[:, i*cw:(i+1)*cw], U if 1 in planes else np.asarray(rf[1])[:, :cw])
            np.copyto(np.asarray(out[2])[:, i*cw:(i+1)*cw], V if 2 in planes else np.asarray(rf[2])[:, :cw])

        return out

    # turn each temporal window into one wide frame so vapoursynth can handle caching
    offset_clips = _get_window(clip, temp_window)
    out_shape    = core.std.BlankClip(clip=offset_clips[0], width=w * temp_window, height=h, keep=True)
    stacked_clip = core.std.ModifyFrame(out_shape, clips=[out_shape, *offset_clips], selector=_degrade_ffmpeg)

    # slice wide frames back into individual frames and restore chronological order
    if temp_window == 1:
        clip_degraided = stacked_clip
    else:
        offset_clips   = [core.std.Crop(stacked_clip, left=i * w, right=(temp_window - 1 - i) * w) for i in range(temp_window)]
        clip_degraided = core.std.Interleave(offset_clips)
    
    # trim, weave, return
    if clip_degraided.num_frames != num_frames:
        clip_degraided = core.std.Trim(clip_degraided, last=num_frames - 1)
    if fields:
        clip_degraided = core.std.DoubleWeave(clip_degraided, tff=True)
        clip_degraided = core.std.SelectEvery(clip_degraided, 2, 0)
    return core.std.CopyFrameProps(clip_degraided, orig_clip)
