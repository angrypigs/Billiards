from random import uniform

class Computer:
    
    def __init__(self):
        pass
    
    def shoot(self) -> tuple[float, float]:
        return (uniform(-180, 180), uniform(10, 15))