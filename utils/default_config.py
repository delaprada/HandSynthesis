import math

obj_trans = {
    'location': (0, 0, 0),
    'rotation_mode': 'XYZ',
    'rotation_euler': (math.radians(-90), math.radians(0), math.radians(0)),
    'scale': (0.001, 0.001, 0.001),
}

# Camera settings for 4 views
camera_settings = [
    {
        'name': '0',
        'location': (0, 0, 0.5),
        'rotation': (math.radians(0), math.radians(0), math.radians(0)),
    },
    {
        'name': '1',
        'location': (0, 0, -0.5),
        'rotation': (math.radians(0), math.radians(180), math.radians(0)),
    },
    {
        'name': '2',
        'location': (0, -0.5, 0),
        'rotation': (math.radians(90), math.radians(180), math.radians(0)),
    },
    {
        'name': '3',
        'location': (0, 0.5, 0),
        'rotation': (math.radians(-90), math.radians(180), math.radians(0)),
    }
]

tex_wall_settings = [
  {
      'name': 'front',
      'location': (0.0, 8.0, 0.0),
      'rotation': (math.pi * 90.0 / 180.0, 0.0, 0.0),
      'size': 20,
  },
  {
      'name': 'left',
      'location': (8.0, 0.0, 0.0),
      'rotation': (0, math.pi * 90.0 / 180.0, 0.0),
      'size': 20,
  },
  {
      'name': 'back',
      'location': (0.0, -8.0, 0.0),
      'rotation': (math.pi * 90.0 / 180.0, 0.0, 0.0),
      'size': 20,
  },
  {
      'name': 'right',
      'location': (-8.0, 0.0, 0.0),
      'rotation': (0.0, math.pi * 90.0 / 180.0, 0.0),
      'size': 20,
  },
]
