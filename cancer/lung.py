
from pydantic import BaseModel

class Lung(BaseModel):
    Age: int
    Gender: int
    AirPollution: int
    Alcoholuse: int
    DustAllergy: int
    OccuPationalHazards: int
    GeneticRisk: int
    chronicLungDisease: int
    BalancedDiet: int
    Obesity: int
    Smoking: int
    PassiveSmoker: int
    ChestPain: int
    CoughingofBlood: int
    Fatigue: int
    WeightLoss: int
    ShortnessofBreath: int
    Wheezing: int
    SwallowingDifficulty: int
    ClubbingofFingerNail: int
    FrequentCold: int
    DryCough: int
    Snoring: int