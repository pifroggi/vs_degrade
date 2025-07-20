
# Script by pifroggi https://github.com/pifroggi/vs_degrade
# or tepete and pifroggi on Discord

import numpy as np
import random, re
import subprocess, threading, io, os, sys
import shlex, shutil, ctypes.util
import vapoursynth as vs
from pathlib import Path

core = vs.core

def find_binary(name):
    tried_names = [name]
    if not name.startswith("lib"):
        tried_names.append("lib" + name)

    # try dynamic linker
    for libname in tried_names:
        hit = ctypes.util.find_library(libname)
        if hit:
            return hit

    # try system PATH
    for libname in tried_names:
        hit = shutil.which(libname)
        if hit:
            return hit

    # try nearby locations
    here   = Path(__file__).resolve().parent
    bases  = {here, here.parent, Path(sys.executable).resolve().parent}
    subs   = ("", "bin", "tools", "plugins", "vs-plugins", "scripts", "vs-scripts", "ffmpeg", "turbojpeg")

    for base in bases:
        for sub in subs:
            search_path = base / sub
            if not search_path.is_dir():
                continue

            for file in search_path.iterdir():
                if file.name in tried_names or file.stem in tried_names:
                    return str(file.resolve())
    
    raise FileNotFoundError(f"vs_degrade: '{name}' not found by dynamic linker, on system PATH, or near the vs_degrade script. Add path manually.")

def jpeg(clip, quality=50, fields=False, planes=[0, 1, 2], path=None):
    # adds jpeg compression directly to YUV clips without any format/color conversions while keeping chroma sampling.
    from turbojpeg import TurboJPEG, TJSAMP_420, TJSAMP_422, TJSAMP_444
    
    # checks and settings
    turbojpeg_binary = path or find_binary("turbojpeg")  # find turbojpeg
    turbojpeg_func   = TurboJPEG(lib_path=turbojpeg_binary)
    pad              = 4  # default jpeg padding for turbojpeg, needed for crop later
    subsample_map    = {(1, 1): TJSAMP_420,
                        (1, 0): TJSAMP_422,
                        (0, 0): TJSAMP_444}
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_degrade.jpeg: This is not a Vapoursynth clip.")
    
    orig_clip      = clip
    clip_format    = clip.format
    num_planes     = clip.format.num_planes
    subsampling    = (clip_format.subsampling_w, clip_format.subsampling_h)
    subsampling_w  = 1 << clip_format.subsampling_w # 1 for 444, 2 for 422/420
    subsampling_h  = 1 << clip_format.subsampling_h # 1 for 444/422, 2 for 420
    jpeg_subsample = subsample_map[subsampling]     # turbojpeg enum
    
    if clip_format.color_family != vs.YUV:
        raise ValueError("vs_degrade.jpeg: Jpeg works in YUV. This expects the input clip to be YUV already.")
    if clip_format.bits_per_sample != 8:
        raise ValueError("vs_degrade.jpeg: Jpeg only support 8-bit.")
    if subsampling not in subsample_map:
        raise ValueError("vs_degrade.jpeg: Jpeg only supports 444, 422, and 420 subsampling.")
    if planes is None:
        planes = list(range(num_planes))
    if isinstance(planes, int):
        planes = [planes]
    if num_planes == 1:
        planes = [0]

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

    def _degradejpeg(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        # optionally randomize quality
        quality = q_fixed if constant_q else random.randint(*quality_range)
    
        # convert planes to arrays
        Y = np.asarray(f[0])
        U = np.asarray(f[1])
        V = np.asarray(f[2])
        h, w = Y.shape

        # pack YUV into one contiguous buffer
        yuv_buf = np.concatenate((Y.ravel(), U.ravel(), V.ravel()))

        # encode to jpeg
        encoded = turbojpeg_func.encode_from_yuv(yuv_buf, height=h, width=w, quality=quality, jpeg_subsample=jpeg_subsample)

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
        clip = core.std.SeparateFields(clip, tff=True)
        clip = core.std.ModifyFrame(clip, clip, selector=_degradejpeg)
        clip = core.std.DoubleWeave(clip, tff=True)
        clip = core.std.SelectEvery(clip, 2, 0)
        return core.std.CopyFrameProps(clip, orig_clip)
    else:
        return core.std.ModifyFrame(clip, clip, selector=_degradejpeg)

def ffmpeg(clip, chunk=10, args="-c:v mpeg2video -q:v 10", fields=False, planes=[0, 1, 2], path=None):
    # runs randomizable ffmpeg commands in chunks on a clip.
    
    # fix error handling
    import signal
    try:                   # ignore SIGPIPE so python gets BrokenPipeError instead if encoding fails
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    except AttributeError: # not needed on windows
        pass
    
    # checks and settings
    ffmpeg_binary = path or find_binary("ffmpeg") # find ffmpeg
    frame_cache   = {}                            # stores processed frames
    frame_size    = []                            # stores size values after first frame
    subsample_map = {vs.YUV420P8: "yuv420p",
                     vs.YUV422P8: "yuv422p",
                     vs.YUV444P8: "yuv444p"}
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_degrade.ffmpeg: This is not a Vapoursynth clip.")

    orig_clip   = clip
    clip_format = clip.format
    num_planes  = clip.format.num_planes
    num_frames  = clip.num_frames
    pixfmt      = subsample_map[clip_format.id]

    if clip_format.id not in [vs.YUV420P8, vs.YUV422P8, vs.YUV444P8]:
        raise ValueError("vs_degrade.ffmpeg: Only YUV444P8, YUV422P8, and YUV420P8 encoding is supported for now.")
    if chunk < 1:
        raise ValueError("vs_degrade.ffmpeg: Number of frames in a chunk must be at least 1.")
    if planes is None:
        planes = list(range(num_planes))
    if isinstance(planes, int):
        planes = [planes]
    if num_planes == 1:
        planes = [0]
    if fields:
        chunk      = chunk * 2
        num_frames = num_frames * 2

    # build tokenized templates
    if isinstance(args, str):
        templates = [shlex.split(args)]
    elif isinstance(args, (list, tuple)) and all(isinstance(t, str) for t in args):
        templates = [shlex.split(t) for t in args]
    else:
        raise TypeError("vs_degrade.ffmpeg: Args must be a string or list of strings.")

    # patterns for {rand(a,b)} and {randf(a,b)} and {choice(a,b)}
    _rand_int = re.compile(r"{rand\((-?\d+),\s*(-?\d+)\)}")
    _rand_flt = re.compile(r"{randf\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)}")
    _rand_cho = re.compile(r"{choice\(([^{}()]*)\)}")

    def _randomize_values(token):
        # replaces {rand(a,b)} and {randf(a,b)} and {choice(a,b)}
        def _r_int(m):
            return str(random.randint(int(m[1]), int(m[2])))
        def _r_flt(m):
            return "{:.6g}".format(random.uniform(float(m[1]), float(m[2])))
        def _r_cho(m):
            opts = [o.strip() for o in m[1].split(',')]
            return random.choice(opts)

        token = _rand_cho.sub(_r_cho, token)  # first so text is not interpetet as numbers
        token = _rand_int.sub(_r_int, token)
        token = _rand_flt.sub(_r_flt, token)
        return token

    def _read_stream(pipe, sink):
        # background reader that drains pipe into a bytearray
        with pipe:
            for chunk in iter(lambda: pipe.read(65536), b''):
                sink.extend(chunk)

    def _encode_chunk(chunk_start, current_frame_n, current_frame):
        nonlocal frame_size
        cur_chunk        = min(chunk, num_frames - chunk_start) # handle last chunk length
        if not frame_size:
            w, h         = current_frame.width, current_frame.height
            cw, ch       = w >> clip_format.subsampling_w, h >> clip_format.subsampling_h
            frame_size.extend([w*h + 2*cw*ch, w, h, cw, ch])
        fs, w, h, cw, ch = frame_size

        # pick random args template
        template_tokens = random.choice(templates)
        tokens = [
            _randomize_values(t).format(w=w, h=h, pixfmt=pixfmt, n=chunk_start)
            for t in template_tokens
        ]

        # encode/decode commands
        enc_cmd = ([ffmpeg_binary, "-loglevel", "error", "-xerror", "-f", "rawvideo", "-pix_fmt", pixfmt, "-s", f"{w}x{h}", "-i", "-"] + tokens + ["-f", "nut", "-"])
        dec_cmd = ([ffmpeg_binary, "-loglevel", "error", "-xerror", "-i", "-", "-f", "rawvideo", "-pix_fmt", pixfmt, "-"])

        # create encoder and decoder subprocess
        enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        dec = subprocess.Popen(dec_cmd, stdin=enc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        enc.stdout.close()

        # prepare buffer on separate thread that will read dec output
        outbuf = bytearray()
        rdr = threading.Thread(target=_read_stream, args=(dec.stdout, outbuf))
        rdr.start()

        # feed a chunk of frames into the encoder
        try:
            for i in range(cur_chunk):
                fn = chunk_start + i
                rf = current_frame if fn == current_frame_n else clip.get_frame(fn)
                enc.stdin.write(np.asarray(rf[0])[:, :w].tobytes())
                enc.stdin.write(np.asarray(rf[1])[:, :cw].tobytes())
                enc.stdin.write(np.asarray(rf[2])[:, :cw].tobytes())

        # error if ffmpeg fails
        except (BrokenPipeError, OSError) as e:
            enc.stdin.close()
            enc.wait()
            dec.wait()
            err_txt = enc.stderr.read().decode(errors="replace")
            raise RuntimeError(
                f"vs_degrade.ffmpeg: Encoding failed with args '{' '.join(tokens)}'\n"
                f"{err_txt or '<empty>'}"
                ) from e

        # close encoder input and wait for both processes to finish
        enc.stdin.close()
        enc.wait()
        dec.wait()
        rdr.join()

        # slice decoded video into individual frames and cache them
        for i in range(cur_chunk):
            frame_cache[chunk_start + i] = memoryview(outbuf)[i*fs:(i+1)*fs]
            
    def _degradeffmpeg(n, f):
        if n not in frame_cache:
            _encode_chunk((n // chunk) * chunk, n, f)

        fs, w, h, cw, ch = frame_size
        blob = frame_cache.pop(n)

        # extract planes from raw frame data
        Y = np.frombuffer(blob, dtype=np.uint8, count=w*h).reshape((h, w))
        U = np.frombuffer(blob, dtype=np.uint8, count=cw*ch, offset=w*h).reshape((ch, cw))
        V = np.frombuffer(blob, dtype=np.uint8, count=cw*ch, offset=w*h+cw*ch).reshape((ch, cw))

        out = f.copy()
        if 0 in planes:
            np.copyto(np.asarray(out[0]), Y)
        if 1 in planes:
            np.copyto(np.asarray(out[1]), U)
        if 2 in planes:
            np.copyto(np.asarray(out[2]), V)
        return out

    if fields:
        clip = core.std.SeparateFields(clip, tff=True)  # clip needs reference for getframe here
        clip_degraided = core.std.ModifyFrame(clip, clip, selector=_degradeffmpeg)
        clip_degraided = core.std.DoubleWeave(clip_degraided, tff=True)
        clip_degraided = core.std.SelectEvery(clip_degraided, 2, 0)
        return core.std.CopyFrameProps(clip_degraided, orig_clip)
    else:
        return core.std.ModifyFrame(clip, clip, selector=_degradeffmpeg)
