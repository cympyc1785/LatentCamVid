import bpy, json, sys
s = bpy.context.scene
json.dump({'start': s.frame_start, 'end': s.frame_end,
           'fps': s.render.fps / max(1, s.render.fps_base)},
          open(sys.argv[sys.argv.index('--') + 1], 'w'))
