from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfWaveTerrainCfg,
)

TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=1.0),
    }
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=0.3),
        "pyramid_stairs_down": BoxPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.05),
            step_width=0.3,
            platform_width=1.5,
            border_width=0.25,
        ),
        "pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.0, 0.1),
            platform_width=2.0,
            border_width=0.25,
            horizontal_scale=0.15,
            vertical_scale=0.005,
        ),
        "pyramid_slope_inv": HfPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.0, 0.1),
            inverted=True,
            platform_width=2.0,
            border_width=0.25,
            horizontal_scale=0.15,
            vertical_scale=0.005,
        ),
        "wave": HfWaveTerrainCfg(
            proportion=0.2,
            amplitude_range=(0.0, 0.03),
            num_waves=2,
            horizontal_scale=0.15,
            vertical_scale=0.005,
        ),
    }
)

TERRAINS_ENTITY_CFG = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TERRAINS_CFG,
    max_init_terrain_level=5,
    env_spacing=2.5,
)

ROUGH_TERRAINS_ENTITY_CFG = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=5,
    env_spacing=2.5,
)

PLANE_ENTITY_CFG = TerrainEntityCfg(
    terrain_type="plane"
)
