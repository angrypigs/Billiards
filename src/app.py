from src.utils import *
from src.window import Window
from src.train import SupervisedTrainer
from src.smart_teacher import SmartTeacher
from src.simulation import AISimulation

def main() -> None:
    sep = '=' * 40
    n = input(f"""
{sep}

Welcome to AI billiards menu

{sep}
              
Enter:
- "train" to train the model,
- "teacher" to generate "ideal" shoots data for model by "ghost ball" shooting method,
- "showcase" to watch the AI playing,
- "simulate" to get model stats for given amount of matches (no graphics)
- anything else to just play
""")

    if n == "train":
        trainer = SupervisedTrainer()
        trainer.train(epochs=50, batch_size=256)
    elif n == "showcase":
        balls = input("Enter balls quantity or skip to leave it normal mode: ")
        balls = 0 if not balls.isdigit() else max(min(int(balls), 15), 1)
        window = Window(True, special_mode=balls)
    elif n == "teacher":
        teacher = SmartTeacher()
        samples = input("Enter samples quantity or skip to leave it 1k: ")
        samples = 1000 if not samples.isdigit() else max(int(samples), 100)
        teacher.run(samples)
    elif n == "simulate":
        simulation = AISimulation()
        rounds = input("Enter rounds quantity or skip to leave it 1: ")
        rounds = 1 if not rounds.isdigit() else max(int(rounds), 1)
        simulation.simulate_matches(rounds)
    else:
        window = Window()