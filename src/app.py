from src.utils import *
from src.window import Window
from src.agent_rl import RLAgent
from src.train import SupervisedTrainer
from src.dummy_player import dummyPlayer

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
        trainer.train_from_db()
        pass
    elif n == "simulate":
        rounds = input("Enter number of rounds or skip to go just one: ")
        special = input("Enter number of balls in special mode or skip to leave it normal: ")
        agent = RLAgent()
        rounds = 1 if not rounds.isdigit() else max(int(rounds), 1)
        special = 0 if not special.isdigit() else max(min(int(special), 15), 1)
        for _ in range(rounds):
            agent.play_episode(special_mode=special)
    elif n == "showcase":
        window = Window(True)
    elif n == "dummy":
        dummy = dummyPlayer()
        rounds = input("Enter number of rounds or skip to go just one: ")
        batch = input("Enter batch size or skip to let it default (20): ")
        rounds = 1 if not rounds.isdigit() else max(int(rounds), 1)
        batch = 20 if not batch.isdigit() else max(int(batch), 1)
        dummy.play_rounds(rounds=rounds, batch_size=batch)
    else:
        window = Window()