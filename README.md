# Degradation functions for VapourSynth
Mainly to generate datasets without the need for intermediates. Maybe useful to compare encoding settings.

### Requirements
* [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo/releases) *(optional)*
   * __Windows:__ Download & install `libjpeg-turbo-X.X.X-vc-x64.exe`  
     __Linux:__ Via package manager e.g. `apt install libturbojpeg`
   * Then install python package via `pip install PyTurboJPEG`
* [ffmpeg](https://ffmpeg.org/download.html) *(optional)*
   * Download and add to PATH, or put into your vapoursynth folder.  
     You have probably already done that.


### Setup
Put the `vs_degrade.py` file into your vapoursynth scripts folder.  
Or install via pip: `pip install -U git+https://github.com/pifroggi/vs_degrade.git`

<br />

## Jpeg Degradation
Degrades a YUV clip directly as is, without upsampling chroma or doing any format/color conversions, since Jpeg also works in YUV. Adds purely spatial compression artifacts.

```python
import vs_degrade
clip = vs_degrade.jpeg(clip, quality=50, fields=False, planes=[0, 1, 2], path=None)
```

__*`clip`*__  
Clip to degrade. Jpeg supports YUV444P8, YUV422P8, and YUV420P8 formats.

__*`quality`*__  
Image quality in the range 1-100 with 1 being the worst.  
Can be a constant value or randomized each frame by providing a range: `quality=[30, 80]`

__*`fields`* (optional)__  
Will separate the clip into fields, degrade each field seperately, then put them back together.  
This creates interlacing artifacts like combing and more mosquito noise.

__*`planes`* (optional)__  
Which planes to degrade. Any unmentioned planes will simply be copied.  
If nothing is set, all planes will be degraded.

__*`path`* (optional)__  
Path to libjpeg-turbo (`turbojpeg.dll` on Windows, `libturbojpeg.so` on Linux), if not auto-detected.

<br />

## FFmpeg Degradation
Runs randomizable FFmpeg commands in chunks directly on a YUV clip as is, without upsampling chroma or doing any format/color conversions. Adds spatial and temporal compression artifacts within each chunk.

```python
import vs_degrade
clip = vs_degrade.ffmpeg(clip, chunk=10, args="-c:v mpeg2video -q:v 10", fields=False, planes=[0, 1, 2], path=None)
```

__*`clip`*__  
Clip to degrade. Currently supports YUV444P8, YUV422P8, and YUV420P8 formats.

__*`chunk`*__  
Amount of frames to encode at once.

__*`args`*__  
The video encoding arguments of an FFmpeg command.
* Simplest example using the MPEG-2 codec with quality 10:
  ```python
  args = "-c:v mpeg2video -q:v 10"
  ```
* Arguments can optionally be randomized per chunk:  
  `{rand(5,50)}` sets randomizer range for int values  
  `{randf(-0.5,0.9)}` sets randomizer range for float values  
  `{choice(veryfast,medium,veryslow)}` chooses randomly from a list  
  Example using the H.264 codec with random crf and preset:
  ```python
  args = "-c:v libx264 -crf {rand(5,50)} -preset {choice(veryfast,medium,veryslow)}"
  ```
* Full commands can be randomized per chunk by providing a list:
  ```python
  args = ["-c:v mpeg2video -q:v {rand(5,30)}",
          "-c:v libx264 -crf {rand(5,50)} -x264-params bframes={rand(0,16)}",
          "-c:v libx265 -crf {rand(5,50)} -preset {choice(veryfast,medium,veryslow)}",
          "-c:v libvpx-vp9 -crf {rand(5,60)} -b:v 0",
          "-c:v prores_ks -p:v {rand(0,4)} -q:v {rand(1,9)}"]
  clip = vs_degrade.ffmpeg(clip, chunk=10, args=args)
  ```
* FFmpeg filters can also be applied. This one for example randomly sharpens before compressing:  
  ```python
  args = "-vf unsharp=5:5:{randf(0.0,1.0)} -c:v mpeg2video -q:v 10"
  ```
  This adds gibbs ringing and skips compression. `{w}` `{h}` gets dimensions, output needs to be equal:
  ```python
  args = "-vf scale={w}*0.85:{h}*0.85:sws_flags=sinc,scale={w}:{h}:sws_flags=sinc' -c:v rawvideo"
  ```
* You may want to add additional interlacing flags if `fields=True`, but it is not strictly necessary:  
  ```python
  args = "-c:v mpeg2video -q:v 10 -flags +ildct+ilme -top 1"
  ```  
__*`fields`* (optional)__  
Will seperate the clip into fields, degrade with FFmpeg, then put them back together.  
This creates interlacing artifacts like combing and more mosquito noise.

__*`planes`* (optional)__  
Which planes to degrade. Any unmentioned planes will simply be copied.  
If nothing is set, all planes will be degraded.

__*`path`* (optional)__  
Path to FFmpeg (`ffmpeg.exe` on Windows, `ffmpeg` on Linux), if not auto-detected.
