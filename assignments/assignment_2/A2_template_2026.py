"""EC A2 template code - neuroevolution for targeted locomotion with ARIEL.

WHAT THIS FILE IS
-----------------
A *demo*, not a solution. It spawns a robot, drives it with a neural network
whose weights are RANDOM, runs the simulation, and reports how close the robot
ended up to a target.

There is deliberately NO evolution in here. Building the EA (representation,
initialisation, parent selection, variation, survivor selection) is the assignment.
See "YOUR JOB" at the bottom of this file.

THE ASSIGNMENT IN A NUTSHELL
------------------------------
Evolve the weights of a neural network controller so that a robot moves from
SPAWN_POS to TARGET_POSITION within the simulation time.

    fitness = distance between the robot's final position and TARGET_POSITION

HOW TO RUN
----------

Change MODE below to switch between an interactive viewer, a headless run,
a rendered video, or a single frame.
"""

# Standard library
from pathlib import Path
import time
from typing import Literal

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
import sys

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import (
    gecko,
    snake,
    turtle,
    spider_8,
    centipede_3,
)
from ariel.ec import EA, EAOperation, FloatMutator, Individual, Population, set_seed
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

# Type aliases
type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
RNG = np.random.default_rng(SEED)

# ariel.ec's own generators/mutators/crossover draw from a separate,
# package-level RNG. Reseed it too if you build your EA on ariel.ec,
# or every one of your "multiple seeds" runs the same variation operators.
set_seed(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]  # where the robot starts
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]  # where it should end up
SIM_DURATION: float = 15.0  # seconds of simulated time per evaluation
MODE: ViewerTypes = "simple"  # see run_experiment() for the options


# ============================================================================ #
#  1. THE BODY AND THE WORLD
# ============================================================================ #
def build_world() -> SimpleFlatWorld:
    """Create the environment the robot lives in.

    YOU MAY CHANGE THIS. Options include: SimpleFlatWorld, RuggedTerrainWorld,
    CraterTerrainWorld, AmphitheatreTerrainWorld, OlympicArena, ...
    (SimpleTiltedWorld is not supported for this task.)

    Whatever you pick, keep it FIXED for all runs you compare against each
    other, and say in your report which one you used. A controller evolved on
    flat ground and one evolved on rugged terrain are not comparable numbers.
    """
    return SimpleFlatWorld()


def build_robot() -> CoreModule:
    """Create the robot body.

    YOU MAY CHANGE THIS. Options include the prebuilt bodies in
    `ariel.body_phenotypes.robogen_lite.prebuilt_robots` (gecko, spider, ...).

    Two consequences of this choice, and they matter:
      * The body determines `model.nu` (the number of hinges you must send
        commands to) - that is the OUTPUT size of your controller.
      * The body determines the size of `data.qpos` - if you feed qpos to your
        network, that is (part of) your INPUT size.
    Change the body and your genotype length changes with it. Keep the body
    FIXED within an experiment.
    """
    return gecko()


# ============================================================================ #
#  2. THE CONTROLLER CONTRACT
# ============================================================================ #
#
# MuJoCo calls the controller every physics step with (model, data); its job
# is to write into data.ctrl.
#
#   INPUTS   : whatever you read from `data` (qpos, qvel, time, ...), plus any
#              task info you already know, e.g. the vector to TARGET_POSITION.
#              INPUT SIZE is your choice, but must stay CONSTANT.
#   OUTPUTS  : exactly `model.nu` values, one per actuated hinge.
#   RANGE    : hinges accept [-pi/2, +pi/2] radians. A tanh output gives
#              [-1, 1] - rescale: actions * (np.pi / 2).
#   WRITING  : DIRECT (data.ctrl[:] = actions) commands the angle straight -
#              fast, but can destabilise the sim on large jumps. DELTA
#              (data.ctrl[:] += actions * alpha, alpha ~ 0.05, then clip) is
#              smoother but accumulates, so clipping is required. Pick one,
#              justify it, use it everywhere.
#   NaN      : blown-up weights silently write NaN into data.ctrl. Assert
#              against it while developing.
#
# ============================================================================ #

# Controller architecture - decide before writing your EA.
HIDDEN_SIZE: int = 6
BEAT_FREQ: float = 1.0    # beats per second
USE_DIRECTION: bool = True
N_EXTRA: int = 4 if USE_DIRECTION else 2
POP_SIZE: int = 20
GENERATIONS: int = 500    # 3 for the test, 500 for the final runs
ALGO: str = sys.argv[2] if len(sys.argv) > 2 else "0.5"
RANDOM_SEARCH: bool = ALGO == "random"
SIGMA: float = 0.0 if RANDOM_SEARCH else float(ALGO)
CONFIG: str = "random" if RANDOM_SEARCH else f"sigma{SIGMA}"
HISTORY: list = []


def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
    weights: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Map robot state to hinge commands: in -> hidden -> actions.

    In this demo `weights` is drawn at RANDOM. In your assignment, `weights`
    is what the evolutionary algorithm produces: an individual's genotype,
    reshaped into these matrices. You are free to change the architecture
    itself (layers, activations, ...) - just keep input/output sizes correct.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model. Use `model.nu` for the number of hinges.
    data : mj.MjData
        The MuJoCo data. This is where you read the robot's state from.
    weights : list of ndarray
        [w1, w2] - the layer weight matrices.

    Returns
    -------
    npt.NDArray[np.float64]
        `model.nu` action values, already scaled to [-pi/2, pi/2].
    """
    w1, w2 = weights

    # --- INPUTS ---------------------------------------------------------- #
    # Bare qpos - the simplest choice, not necessarily a good one. See
    # YOUR JOB below.
    t = data.time                                      # the simulation's clock, in seconds
    beat = [np.sin(2 * np.pi * BEAT_FREQ * t),
            np.cos(2 * np.pi * BEAT_FREQ * t)]
    to_target = np.array(TARGET_POSITION[:2]) - data.qpos[0:2]   # GPS: (dx, dy) to the target
    parts = [data.qpos, beat]
    if USE_DIRECTION:
        parts.append(to_target)
    inputs = np.concatenate(parts)

    # --- FORWARD PASS ----------------------------------------------------- #
    layer1 = np.tanh(inputs @ w1)
    outputs = np.tanh(layer1 @ w2)  # in [-1, 1]

    # --- RESCALE TO THE HINGE RANGE --------------------------------------- #
    return outputs * (np.pi / 2)  # in [-pi/2, pi/2]


def make_random_weights(
    input_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:
    """Draw a random parameter set for `nn_controller`.

    THIS IS THE FUNCTION YOUR EA REPLACES. Instead of sampling weights from a
    normal distribution, your EA will search for them.

    Note the total parameter count printed by main(): that is the length of the
    flat vector an individual's genotype has to encode. Reshaping a flat
    genotype back into these matrices is on you.
    """
    return [
        RNG.normal(scale=0.5, size=(input_size, HIDDEN_SIZE)),
        RNG.normal(scale=0.5, size=(HIDDEN_SIZE, output_size)),
    ]

def unflatten(genome, input_size, output_size):
    """Pack a flat genome into the two weight matrices [w1, w2]."""
    n1 = input_size * HIDDEN_SIZE                                     # how many eggs fit in carton 1?
    w1 = genome[:n1].reshape(input_size, HIDDEN_SIZE)           # carton 1: inputs -> mixers
    w2 = genome[n1:].reshape(HIDDEN_SIZE, output_size)           # carton 2: mixers -> motors
    return [w1, w2]


# ============================================================================ #
#  3. POSITION AND FITNESS
# ============================================================================ #
#
# The robot is spawned with a free joint, so data.qpos[0:3] IS the core's
# (x, y, z) world position. Read it before and after stepping - no tracker or
# bookkeeping needed. (`data.geom("robot1_core").xpos` works too.)
#
# ============================================================================ #


def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    """Return the robot core's current (x, y, z) world position."""
    return np.asarray(data.qpos[0:3]).copy()


def fitness_function(
    initial_position: npt.NDArray[np.float64],
    final_position: npt.NDArray[np.float64],
) -> float:
    """Score one evaluation. LOWER IS BETTER.

    The plain version: how far is the robot from the target when time runs out?

    `initial_position` is unused here on purpose - it is passed in because the
    moment you want a less naive fitness you will need it. Some things worth
    thinking about (and, ideally, comparing in your report):
      * Distance *reduced* rather than distance remaining, so a robot that
        starts closer is not rewarded for standing still.
      * Penalising a robot that falls over or leaves the arena.
      * Whether the z-axis should count at all - a robot that jumps is not
        closer to the target in any way you care about.
    See `ariel.simulation.tasks.targeted_locomotion` for some worked variants.
    """
    target = np.asarray(TARGET_POSITION)
    return float(np.linalg.norm(final_position[:2] - target[:2]))


# ============================================================================ #
#  4. RUNNING ONE EVALUATION
# ============================================================================ #


def run_experiment(genome, mode: ViewerTypes = MODE) -> float:
    """Set up the world, run one simulation, and return the fitness.

    This is the function your EA calls once per individual, with `mode` set
    to "simple" (headless).

    Returns
    -------
    float
        The fitness of this run. Lower is better.
    """
    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    # --- World and robot --------------------------------------------------- #
    world = build_world()
    robot = build_robot()

    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    # Compile the world into a model. USE AS IS.
    model = world.spec.compile()
    data = mj.MjData(model)

    # Put the simulation in a clean, known state before reading anything.
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    # --- Wire up the controller -------------------------------------------- #
    # Sizes are read from the compiled model, never hardcoded - they depend on
    # the body you chose in build_robot().
    input_size = len(data.qpos) + N_EXTRA
    output_size = model.nu

    weights = unflatten(genome, input_size, output_size)

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        """Compute and apply actions; MuJoCo calls this every physics step."""
        actions = nn_controller(m, d, weights)

        # DIRECT application (see the controller contract above).
        d.ctrl[:] = actions

        # DELTA application - comment out the line above and use these instead:
        # delta = 0.05
        # d.ctrl[:] += actions * delta
        # d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

    # --- Record the starting point ----------------------------------------- #
    initial_position = get_core_position(data)

    # --- Run ---------------------------------------------------------------- #
    if mode != "no_control":
        mj.set_mjcb_control(control_callback)

    match mode:
        case "launcher":
            # Interactive window. Great for seeing what your robot does,
            # useless inside an evolutionary loop.
            viewer.launch(model=model, data=data)
        case "simple":
            # Headless. THIS is the one your EA uses.
            simple_runner(model, data, duration=SIM_DURATION)
        case "video":
            # Render to an mp4 - for the figures in your report.
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(
                model,
                data,
                duration=SIM_DURATION,
                video_recorder=recorder,
            )
        case "frame":
            # A single image of the scene. Useful to check your spawn position
            # and that the robot is not clipping through the floor.
            single_frame_renderer(model, data, steps=1, show=True)
        case "no_control":
            # No controller attached: drag the hinges around by hand.
            viewer.launch(model=model, data=data)

    # Detach the callback again so the next run starts clean.
    mj.set_mjcb_control(None)

    # --- Score -------------------------------------------------------------- #
    final_position = get_core_position(data)
    fitness = fitness_function(initial_position, final_position)

    # console.log(f"start  : {np.round(initial_position, 3)}")
    # console.log(f"end    : {np.round(final_position, 3)}")
    # console.log(f"target : {np.round(TARGET_POSITION, 3)}")
    # console.log(f"fitness: {fitness:.4f}   (lower is better)")

    return fitness


def make_individual(num_weights: int) -> Individual:
    ind = Individual()
    ind.genotype = RNG.normal(scale=0.5, size=num_weights).tolist()
    return ind


def evaluate(population: Population) -> Population:
    # Only simulate individuals that don't have a fitness yet.
    for ind in population.unevaluated:
        ind.fitness = run_experiment(np.array(ind.genotype), MODE)
    return population


def parent_selection(population: Population) -> Population:
    # Best half of the living population become parents.
    ranked = population.alive.sort(sort="min")
    for i, ind in enumerate(ranked):
        ind.tags = {"parent": i < len(ranked) // 2}
    return population


def reproduce(population: Population) -> Population:
    # POP_SIZE children: copy a random parent, nudge every weight.
    parents = population.where(lambda ind: bool(ind.tags.get("parent", False)))
    for _ in range(POP_SIZE):
        p = parents[int(RNG.integers(len(parents)))]
        child = Individual()
        child.genotype = FloatMutator.gaussian(p.genotype, std=SIGMA, mutation_probability=1.0)
        population.append(child)
    return population


def survivor_selection(population: Population) -> Population:
    # Parents + children compete; only the best POP_SIZE stay alive.
    ranked = population.alive.sort(sort="min")
    for ind in ranked[POP_SIZE:]:
        ind.alive = False
    return population


def record(population: Population) -> Population:
    fits = [ind.fitness for ind in population.alive]
    gen = len(HISTORY)
    HISTORY.append((gen, min(fits), float(np.mean(fits))))
    print(f"gen {gen:3d}   best {min(fits):.4f}   mean {float(np.mean(fits)):.4f}")
    return population

def random_search(num_weights: int) -> None:
    """Same budget as the EA: GENERATIONS + 1 blocks of POP_SIZE random brains."""
    best_so_far = float("inf")
    for gen in range(GENERATIONS + 1):
        fits = [
            run_experiment(RNG.normal(scale=0.5, size=num_weights), MODE)
            for _ in range(POP_SIZE)
        ]
        best_so_far = min(best_so_far, min(fits))
        HISTORY.append((gen, best_so_far, float(np.mean(fits))))
        print(f"block {gen:3d}   best so far {best_so_far:.4f}   block mean {float(np.mean(fits)):.4f}")

def main() -> None:
    """Run a single demo evaluation with a randomly-weighted controller."""
    # A quick look at the size of the problem you are about to search.
    mj.set_mjcb_control(None)
    world = build_world()
    robot = build_robot()
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )
    model = world.spec.compile()
    data = mj.MjData(model)

    input_size = len(data.qpos) + N_EXTRA
    output_size = model.nu
    num_weights = (
        input_size * HIDDEN_SIZE
        + HIDDEN_SIZE * output_size
    )
    console.log(f"controller inputs (len(data.qpos)) : {input_size}")
    console.log(f"controller outputs (model.nu)      : {output_size}")
    console.log(f"genotype length (total weights)    : {num_weights}")

    if RANDOM_SEARCH:
        random_search(num_weights)
    else:
        initial = Population([make_individual(num_weights) for _ in range(POP_SIZE)])
        initial = evaluate(initial)
        record(initial)

        ops = [
            EAOperation(parent_selection),
            EAOperation(reproduce),
            EAOperation(evaluate),
            EAOperation(survivor_selection),
            EAOperation(record),
        ]
        ea = EA(
            initial,
            ops,
            num_steps=GENERATIONS,
            is_maximisation=False,
            db_file_path=DATA / f"{CONFIG}_seed{SEED}.db",
            quiet=False,
        )
        ea.run()

    out = DATA / f"{CONFIG}_seed{SEED}.csv"
    with open(out, "w") as fh:
        fh.write("generation,best,mean\n")
        for gen, b, m in HISTORY:
            fh.write(f"{gen},{b},{m}\n")
    print("saved", out)


if __name__ == "__main__":
    main()


# ============================================================================ #
#  YOUR JOB
# ============================================================================ #
#
# Everything above runs one robot with random weights. It will score badly, and
# it will score badly in a slightly different way every time you change SEED.
# Your task is to replace "random" with "evolved".
#
# Build a proper EA on top of `ariel.ec`. You are expected to use that module -
# it gives you the population/individual data model, the operators, and free
# persistence of every generation to a SQLite database, which you will want
# when it is time to plot convergence curves for the report.
#
#     from ariel.ec import EA, EAOperation, Individual, Population
#
# For a complete, runnable example of how those pieces fit together (a one-max
# EA with parent selection, crossover, mutation and survivor selection written
# as separate steps), read:
#
#     examples/new_EC_engine_example.py
#
# and the API documentation at:
#
#     https://ci-group.github.io/ariel/
#
# ---- EXPERIMENTAL RIGOUR ---------------------------------------------------
#
#   One run proves nothing. Repeat every configuration over several
#     independent seeds and report mean and spread.
#   Log best/mean/worst fitness per generation. The database `ariel.ec`
#     writes makes this straightforward.
#   Compare against a baseline. Random search with the same evaluation
#     budget is a simple, but reasonable choice; and it is nearly free to run.
#   Keep body, world, SIM_DURATION and fitness function identical across
#     everything you compare. Change one thing at a time.
#
# ============================================================================ #
