import time
from dataclasses import dataclass

@dataclass
class Thresholds:
    absence_warn: float = 5.0
    absence_violate: float = 20.0
    away_warn: float = 10.0
    away_violate: float = 30.0

class ProctorState:
    """Rule-based timing state machine."""
    def __init__(self, thresholds: Thresholds):
        self.t = thresholds
        self.absence_start = None
        self.away_start = None

    def reset_absence(self):
        self.absence_start = None

    def reset_away(self):
        self.away_start = None

    def update(self, person_count: int, is_focused: bool):
        now = time.time()

        # Multiple people: immediate violation
        if person_count > 1:
            self.reset_absence()
            self.reset_away()
            return "VIOLATION", "Multiple people detected", 0.0

        # Absence timers
        if person_count == 0:
            if self.absence_start is None:
                self.absence_start = now
            dt = now - self.absence_start
            if dt > self.t.absence_violate:
                return "VIOLATION", "Candidate absent", dt
            if dt > self.t.absence_warn:
                return "WARNING", "Candidate absent", dt
            return "OK", "Candidate absent (grace)", dt
        else:
            self.reset_absence()

        # Away timers (only if exactly one person)
        if not is_focused:
            if self.away_start is None:
                self.away_start = now
            dt = now - self.away_start
            if dt > self.t.away_violate:
                return "VIOLATION", "Looking away", dt
            if dt > self.t.away_warn:
                return "WARNING", "Looking away", dt
            return "OK", "Looking away (grace)", dt
        else:
            self.reset_away()

        return "OK", "Normal", 0.0
