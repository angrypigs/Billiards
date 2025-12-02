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
        agent = RLAgent()
        for _ in range(int(rounds) if rounds.isdigit() else 1):
            agent.play_episode()
    elif n == "showcase":
        window = Window(True)
    elif n == "dummy":
        dummy = dummyPlayer()
        rounds = input("Enter number of rounds or skip to go just one: ")
        batch = input("Enter batch size or skip to let it default (20): ")
        dummy.play_rounds(rounds=int(rounds) if rounds.isdigit() else 1, 
                          batch_size=int(batch) if batch.isdigit() else 20)
    else:
        window = Window()