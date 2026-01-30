from src.utils import *
from src.window import Window
from src.train import SupervisedTrainer
from src.smart_teacher import SmartTeacher
from src.simulation import AISimulation

def main() -> None:
    sep = '=' * 50
    choice = input(f"""
{sep}

Welcome to the AI Billiards Menu

{sep}
              
Available commands:
- "train"    : Train the model
- "teacher"  : Generate training data (ideal shots using 'ghost ball' method)
- "showcase" : Watch the AI play
- "simulate" : Run headless simulation to gather statistics (no graphics)
- [Enter]    : Play manually

Your choice: """).strip().lower()

    if choice == "train":
        trainer = SupervisedTrainer()
        trainer.train(epochs=50, batch_size=256)
        
    elif choice == "showcase":
        balls = input("Enter number of balls (or press Enter for default): ")
        balls_count = int(balls) if balls.isdigit() else 0
        if balls_count > 0:
            balls_count = max(min(balls_count, 15), 1)
            
        Window(True, special_mode=balls_count)
        
    elif choice == "teacher":
        samples = input("Enter number of samples (or press Enter for default 1000): ")
        samples_count = int(samples) if samples.isdigit() else 1000
        teacher = SmartTeacher()
        teacher.run(max(samples_count, 100))
        
    elif choice == "simulate":
        rounds = input("Enter number of rounds (or press Enter for default 1): ")
        rounds_count = int(rounds) if rounds.isdigit() else 1
        
        simulation = AISimulation()
        simulation.simulate_matches(max(rounds_count, 1))
        
    else:
        Window()

if __name__ == "__main__":
    main()