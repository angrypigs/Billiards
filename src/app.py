from src.utils import *
from src.window import Window
from src.agent_rl import RLAgent
from src.train import SupervisedTrainer
from src.dummy_player import dummyPlayer

def main() -> None:
    n = input("""Enter:
                - "train" to train the model,
                - "simulate" to load RL agent (no updating graphics),
                - "showcase" to watch the AI playing,
                - "dummy" to generate dummy data by shooting in "best of random" direction
                - anything else to just play
                """)

    if n == "train":
        trainer = SupervisedTrainer()
        trainer.train_from_db()
    elif n == "simulate":
        agent = RLAgent()
        agent.play_episode()
    elif n == "dummy":
        dummy = dummyPlayer()
        dummy.play_rounds(rounds=10)
    else:
        window = Window()