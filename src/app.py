from src.utils import *
from src.window import Window
from src.train import SupervisedTrainer
from src.smart_teacher import SmartTeacher

def main() -> None:
    n = input("""
Enter:
- "train" to train the model,
- "simulate" to load RL agent (no updating graphics),
- "showcase" to watch the AI playing,
- "dummy" to generate dummy data by shooting in "best of random" direction
- anything else to just play
""")

    if n == "train":
        trainer = SupervisedTrainer()
        trainer.train(epochs=50, batch_size=256)
        pass
    elif n == "showcase":
        balls = input("Enter balls quantity or skip to leave it normal mode: ")
        balls = 0 if not balls.isdigit() else max(min(int(balls), 15), 1)
        window = Window(True, special_mode=balls)
    elif n == "teacher":
        teacher = SmartTeacher()
        samples = input("Enter samples quantity or skip to leave it 1k: ")
        samples = 1000 if not samples.isdigit() else max(int(samples), 100)
        teacher.run(samples)
    else:
        window = Window()