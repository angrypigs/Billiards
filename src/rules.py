from src.physics import ShotEvents

class Rules:
    
    def __init__(self, mode: str = "8ball") -> None:
        self.reset()

    def reset(self) -> None:
        self.turn = 0
        
    def handle_turn(self, shot_events: ShotEvents) -> int:
        if not shot_events.pocketed_balls:
            self.turn = abs(self.turn - 1)
        return self.turn
