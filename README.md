# DashcamView

VIOFO A119 V3 Dash Cam Front Camera

![alt text](img/a119.png)

140° Wide Angle

![alt text](img/image.png)

## Autospeed

https://github.com/autowarefoundation/vision_pilot/tree/e45165837e847f2ca5e5df5247cb4167379ecfc7/Models/model_library/AutoSpeed

https://github.com/autowarefoundation/auto_speed

## Autosteer

## EgoLanes

ref visualisation
https://github.com/autowarefoundation/vision_pilot/tree/e45165837e847f2ca5e5df5247cb4167379ecfc7/Models/visualizations/EgoLanes

## Monocular Depth

visualisation

https://github.com/autowarefoundation/vision_pilot/tree/e45165837e847f2ca5e5df5247cb4167379ecfc7/Models/visualizations/Scene3D

## play.py options

Run:

```bash
python3 play.py [--merge {sahi,nmw,nms}]
```

CLI arguments:

- `--merge sahi` (default): SAHI sliced detection for small/distant objects.
- `--merge nmw`: dual-pass YOLO merged with weighted boxes fusion.
- `--merge nms`: dual-pass YOLO merged with standard NMS.

Keyboard controls during playback:

- `q`: quit
- `n`: next video
- `d`: toggle camera/top-down display
- `y`: toggle YOLO vehicle detection
- `e`: toggle EgoLanes overlay
- `r`: toggle raw lane debug points
- `f`: toggle lane render mode (class-mask overlay / polynomial-fit)
- `<` / `>` (also `,` / `.`): decrease/increase top-down max depth by 5 m

Default top-down max depth at startup is `30 m`.

Outputs written by `play.py`:

- `output.mp4`: processed camera view
- `output2.m4v`: top-down assumed-position view
