from src.utils import *
from src.window import Window
from src.train import SupervisedTrainer
from src.dummy_player import dummyPlayer
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
        window = Window(True)
    elif n == "dummy":
        dummy = dummyPlayer()
        rounds = input("Enter number of rounds or skip to go just one: ")
        batch = input("Enter batch size or skip to let it default (20): ")
        rounds = 1 if not rounds.isdigit() else max(int(rounds), 1)
        batch = 20 if not batch.isdigit() else max(int(batch), 1)
        dummy.play_rounds(rounds=rounds, batch_size=batch)
    elif n == "teacher":
        teacher = SmartTeacher()
        samples = input("Enter samples quantity or skip to leave it 1k: ")
        samples = 1000 if not samples.isdigit() else max(int(samples), 100)
        teacher.run(samples)
    else:
        window = Window()