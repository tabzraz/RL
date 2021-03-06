from gym.envs.registration import register

# Atari Games with a deterministic frameskip
for game in ['air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']:
    for obs_type in ['image']:
        # space_invaders should yield SpaceInvaders-v0 and SpaceInvaders-ram-v0
        name = ''.join([g.capitalize() for g in game.split('_')])
        if obs_type == 'ram':
            name = '{}-ram'.format(name)

        nondeterministic = False
        if game == 'elevator_action' and obs_type == 'ram':
            # ElevatorAction-ram-v0 seems to yield slightly
            # non-deterministic observations about 10% of the time. We
            # should track this down eventually, but for now we just
            # mark it as nondeterministic.
            nondeterministic = True

        register(
            id='Wrapped_{}-v0'.format(name),
            entry_point='Envs.AtariWrapper:AtariEnv',
            kwargs={'game_name': name, 'colours': False},
            tags={'wrapper_config.TimeLimit.max_episode_steps': 100000},
            nondeterministic=nondeterministic,
        )

# Grid world non slippery
register(
        id="FrozenLake4x4_NoSlip-v0",
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False},
        tags={'wrapper_config.TimeLimit.max_episode_steps': 100},
        reward_threshold=0.78, # optimum = .8196
)

for s in range(15):
    register(
            id="Maze-{}-v1".format(s),
            entry_point="Envs.MazeEnv_Corners:MazeEnv",
            kwargs={'size': (s, s)},
            tags={'wrapper_config.TimeLimit.max_episode_steps': s*s*7*7*3},
    )
    register(
            id="Maze-{}-v0".format(s),
            entry_point="Envs.MazeEnv:MazeEnv",
            kwargs={'size': (s, s)},
            tags={'wrapper_config.TimeLimit.max_episode_steps': s*s*7*7*3},
    )

for n in range(100):
    register(
            id="Room-{}-v0".format(n),
            entry_point="Envs.BigRoom:BigRoom",
            kwargs={'size': n},
            tags={'wrapper_config.TimeLimit.max_episode_steps': n * n * 5 + 1},
    )

for n in range(50):
    register(
            id="Wide-Maze-{}-v0".format(n),
            entry_point="Envs.SnakingMaze:SnakingMaze",
            kwargs={'size': n, 'corridor_width': 10},
            tags={'wrapper_config.TimeLimit.max_episode_steps': n * n * 10 * 10 * 5 + 1},
    )

for n in range(50):
    register(
            id="Med-Maze-{}-v0".format(n),
            entry_point="Envs.SnakingMaze:SnakingMaze",
            kwargs={'size': n, 'corridor_width': 5},
            tags={'wrapper_config.TimeLimit.max_episode_steps': n * n * 5 * 5 * 5 + 1},
    )

for n in range(50):
    register(
            id="Thin-Maze-{}-v0".format(n),
            entry_point="Envs.SnakingMaze:SnakingMaze",
            kwargs={'size': n, 'corridor_width': 3},
            tags={'wrapper_config.TimeLimit.max_episode_steps': n * n * 3 * 3 * 5 + 1},
    )

    register(
            id="Thin-Maze-{}-Deadly-v0".format(n),
            entry_point="Envs.DeadlySnakingMaze:DeadlySnakingMaze",
            kwargs={'size': n, 'corridor_width': 3},
            tags={'wrapper_config.TimeLimit.max_episode_steps': n * n * 3 * 3 * 5 + 1},
    )

    register(
            id="Thin-Maze-{}-Neg-v0".format(n),
            entry_point="Envs.SnakingMaze:SnakingMaze",
            kwargs={'size': n, 'corridor_width': 3, 'neg_reward': True},
            tags={'wrapper_config.TimeLimit.max_episode_steps': n * n * 3 * 3 * 5 + 1},
    )

for n in range(50):
    register(
            id="Empty-Room-{}-v0".format(n),
            entry_point="Envs.EmptyRoom:EmptyRoom",
            kwargs={'size': n},
            tags={'wrapper_config.TimeLimit.max_episode_steps': n * n * 3 * 3 * 5 + 1},
    )

# VizDoom
register(
        id="DoomMaze-v0",
        entry_point='Envs.DoomMaze:DoomMaze',
        kwargs={'level': 9},
        tags={'wrapper_config.TimeLimit.max_episode_steps': 10000}
)

register(
        id="DoomMazeHard-v0",
        entry_point='Envs.DoomMaze:DoomMaze',
        kwargs={'level': 10},
        tags={'wrapper_config.TimeLimit.max_episode_steps': 10000}
)

for i in range(5):
    register(
            id="Doom_Maze_{}-v0".format(i + 1),
            entry_point='Envs.DoomMaze:DoomMaze',
            kwargs={'level': (11 + i)},
            tags={'wrapper_config.TimeLimit.max_episode_steps': 20000}
    )

# Mario
register(
        id="Mario-1-1-v0",
        entry_point='Envs.Mario:Mario',
        kwargs={},
        tags={'wrapper_config.TimeLimit.max_episode_steps': 10000}
)