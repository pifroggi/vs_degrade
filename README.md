# Degradation functions for Vapoursynth
Mainly to create datasets without the need for intermediates. Maybe useful to compare encoding settings.

### Requirements
* [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo/releases)   *(optional)*
   * __Windows:__ Download & install `libjpeg-turbo-X.X.X-vc-x64.exe`  
     __Linux:__ Via package manager e.g. `apt install libturbojpeg`
   * Then install python package via `pip install PyTurboJPEG`
* [ffmpeg](https://ffmpeg.org/download.html) *(optional)*  
  Download and add to PATH or put into your Vapoursynth folder.  
  Path can also be manually set to a different location.


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
Can be a single value or randomized each frame by providing a range: `quality=[30, 80]`

__*`fields`* (optional)__  
Will separate the clip into fields, degrade each field seperately, then put them back together. This creates interlacing artifacts like combing and more mosquito noise.

__*`planes`* (optional)__  
Which planes to degrade. Any unmentioned planes will simply be copied.  
If nothing is set, all planes will be degraded.

__*`path`* (optional)__  
Path to libjpeg-turbo, in case it is not auto-detected.

<br />

## FFmpeg Degradation
Runs randomizable FFmpeg commands in chunks directly on a YUV clip as is, without upsampling chroma or doing any format/color conversions. Can add spatial and temporal compression artifacts within each chunk.

```python
import vs_degrade
clip = vs_degrade.ffmpeg(clip, chunk=10, args="-c:v mpeg2video -q:v 10", fields=False, planes=[0, 1, 2], path=None)
```

__*`clip`*__  
Clip to degrade. Currently supports YUV444P8, YUV422P8, and YUV420P8 formats.

__*`chunk`*__  
Amount of frames to encode at once.

__*`args`* (optional)__  
The encoding arguments of a FFmpeg command: `args="-c:v mpeg2video -q:v 10"`  
* Int values can be randomized per chunk by providing a range: `args="-c:v mpeg2video -q:v {rand(5,30)}`  
  For a float value range: `{randf(-0.5,0.9)}`  
  For choosing from a list: `{choice(veryfast,medium,veryslow)}`  
* Full commands can be randomized per chunk by providing a list:
  ```python
  args = ["-c:v mpeg2video -q:v {rand(5,30)}",
          "-c:v libx264 -crf {rand(5,50)} -x264-params bframes={rand(0,16)}",
          "-c:v libx265 -crf {rand(5,50)} -preset {choice(veryfast,medium,veryslow)}",
          "-c:v libvpx-vp9 -crf {rand(5,60)} -b:v 0",
          "-c:v prores_ks -p:v {rand(0,4)} -q:v {rand(1,9)}"]
  clip = vs_degrade.ffmpeg(clip, chunk=10, args=args)
  ```
* FFmpeg filters can also be applied, but input and output dimensions need to be equal:  
  `args="-vf eq=contrast={randf(0.5,1.5)}:brightness={randf(-0.2,0.2)} -c:v mpeg2video -q:v 10"`  
* You may want to add additional interlacing flags if `fields=True`, but it is not strictly necessary:  
  `args="-c:v mpeg2video -q:v 10 -flags +ildct+ilme -top 1"`  
* Make sure your randomized values are actually in the range supported by FFmpeg.

__*`fields`* (optional)__  
Will seperate the clip into fields, degrade with FFmpeg, then put them back together. This creates interlacing artifacts like combing and more mosquito noise.

__*`planes`* (optional)__  
Which planes to degrade. Any unmentioned planes will simply be copied.  
If nothing is set, all planes will be degraded.

__*`path`* (optional)__  
Path to FFmpeg, in case it is not auto-detected.
