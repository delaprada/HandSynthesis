import bpy
from typing import Optional, Tuple

    # # Area.001
    # utils.create_area_light(
    #     location=(6.49636, -3.51545, 2.63636),
    #     rotation=(math.radians(-12.2869), math.radians(71.0853), math.radians(-40.6498)),
    #     scale=(1.75815, 1.75815, 1.75815),
    #     energy=300,
    # )

    # # Area.002
    # utils.create_area_light(
    #     location=(-4.57593, -1.80111, 3.52997),
    #     rotation=(math.radians(-65.303), math.radians(-57.7442), math.radians(-236.04)),
    #     scale=(1.75815, 1.75815, 1.75815),
    #     energy=250,
    # )

    # # Area.003
    # utils.create_area_light(
    #     location=(5.69629, 1.71852, 2.28518),
    #     rotation=(math.radians(-82.26), math.radians(-3.41832), math.radians(-106.563)),
    #     scale=(1.75815, 1.75815, 1.75815),
    #     energy=50,
    # )
    
    # Sun light
    # utils.create_sun_light(
    #     location=(5.69629, 1.71852, 2.28518),
    #     rotation=(math.radians(-82.26), math.radians(-3.41832), math.radians(-106.563)),
    #     # scale=(1.75815, 1.75815, 1.75815),
    #     energy=50,
    # )

    # utils.create_point_light(
    #     location=(-4.57593, -1.80111, 3.52997),
    #     rotation=(math.radians(-65.303), math.radians(-57.7442), math.radians(-236.04)),
    #     scale=(1.75815, 1.75815, 1.75815),
    #     energy=2000,
    # )

def create_light(light_type: str, location: Tuple[float, float, float] = (0.0, 0.0, 5.0),
                     rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                     scale: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                     name: Optional[str] = None,
                     energy: float = 100.0) -> bpy.types.Object:
    bpy.ops.object.light_add(type=light_type, location=location, rotation=rotation, scale=scale)

    if name is not None:
        bpy.context.object.name = name
    
    light_ob = bpy.context.object
    light = light_ob.data
    light.energy = energy

    return bpy.context.object
