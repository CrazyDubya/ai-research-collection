from tinytroupe.tools import TinyTool
import json
import logging
import random
from datetime import datetime, timedelta
from tinytroupe.agent import TinyPerson, TinyToolUse
logger = logging.getLogger("tinytroupe")


class RPGCharacterStats:
    """Manages character statistics and leveling"""

    def __init__(self):
        self.level = 1
        self.experience = 0
        self.health = 100
        self.energy = 100
        self.hunger = 0
        self.skills = {
            "strength": 1,
            "intelligence": 1,
            "charisma": 1
        }

    def gain_experience(self, amount):
        """Add experience points and check for level up"""
        self.experience += amount
        level_threshold = self.level * 100  # Simple leveling formula

        if self.experience >= level_threshold:
            self.level_up()
            return True
        return False

    def level_up(self):
        """Increase level and improve stats"""
        self.level += 1
        self.health += 10
        self.energy += 5

        # Randomly improve one skill
        skill = random.choice(list(self.skills.keys()))
        self.skills[skill] += 1


class RPGBasicNeeds(TinyTool):
    """Tool for managing basic character needs like eating and sleeping"""

    def __init__(self, owner=None):
        super().__init__(
            "basic_needs",
            "Handles basic character needs like eating and sleeping",
            owner=owner,
            real_world_side_effects=False
        )
        self.last_meal_time = datetime.now()
        self.last_sleep_time = datetime.now()

    def eat(self, food_quality=1):
        """Handle eating and hunger reduction"""
        if self.owner and hasattr(self.owner, 'stats'):
            self.owner.stats.hunger = max(0, self.owner.stats.hunger - (20 * food_quality))
            self.owner.stats.energy += 5 * food_quality
            self.last_meal_time = datetime.now()
            return True
        return False

    def sleep(self, hours=8):
        """Handle sleeping and energy restoration"""
        if self.owner and hasattr(self.owner, 'stats'):
            self.owner.stats.energy = min(100, self.owner.stats.energy + (10 * hours))
            self.owner.stats.hunger += 5 * hours  # Get hungry while sleeping
            self.last_sleep_time = datetime.now()
            return True
        return False

    def _process_action(self, agent, action):
        if action['type'] == "EAT" and action['content'] is not None:
            food_spec = json.loads(action['content'])
            return self.eat(food_quality=food_spec.get('quality', 1))

        elif action['type'] == "SLEEP" and action['content'] is not None:
            sleep_spec = json.loads(action['content'])
            return self.sleep(hours=sleep_spec.get('hours', 8))

        return False

    def actions_definitions_prompt(self):
        return """
            - EAT: You can eat to reduce hunger and gain some energy. Specify food quality (1-3) in JSON format:
                {"quality": 2}
            - SLEEP: You can sleep to restore energy. Specify hours in JSON format:
                {"hours": 8}
        """

    def actions_constraints_prompt(self):
        return """
            - You should eat when hunger is above 50
            - You should sleep when energy is below 30
            - You cannot sleep for more than 10 hours at once
        """


class RPGWork(TinyTool):
    """Tool for handling work and earning experience"""

    def __init__(self, owner=None):
        super().__init__(
            "work",
            "Handles working actions and rewards",
            owner=owner,
            real_world_side_effects=False
        )
        self.work_history = []

    def perform_work(self, work_type, hours):
        """Handle working and gaining experience"""
        if self.owner and hasattr(self.owner, 'stats'):
            # Energy cost depends on work type
            energy_cost = {
                              "physical": 15,
                              "mental": 10,
                              "social": 8
                          }.get(work_type, 10) * hours

            # Check if character has enough energy
            if self.owner.stats.energy < energy_cost:
                return False

            # Deduct energy and increase hunger
            self.owner.stats.energy -= energy_cost
            self.owner.stats.hunger += 8 * hours

            # Calculate experience gained
            exp_gain = {
                           "physical": 20,
                           "mental": 25,
                           "social": 15
                       }.get(work_type, 15) * hours

            # Apply skill bonuses
            if work_type == "physical":
                exp_gain *= (1 + (self.owner.stats.skills["strength"] * 0.1))
            elif work_type == "mental":
                exp_gain *= (1 + (self.owner.stats.skills["intelligence"] * 0.1))
            elif work_type == "social":
                exp_gain *= (1 + (self.owner.stats.skills["charisma"] * 0.1))

            # Add experience and record work
            leveled_up = self.owner.stats.gain_experience(exp_gain)
            self.work_history.append({
                "type": work_type,
                "hours": hours,
                "exp_gained": exp_gain,
                "time": datetime.now()
            })

            return leveled_up  # Return whether character leveled up

        return False

    def _process_action(self, agent, action):
        if action['type'] == "WORK" and action['content'] is not None:
            work_spec = json.loads(action['content'])
            return self.perform_work(
                work_type=work_spec.get('type', 'physical'),
                hours=work_spec.get('hours', 1)
            )
        return False

    def actions_definitions_prompt(self):
        return """
            - WORK: You can work to gain experience and potentially level up. Specify work type and hours in JSON format:
                {"type": "physical|mental|social", "hours": 4}
        """

    def actions_constraints_prompt(self):
        return """
            - You cannot work for more than 12 hours at once
            - You should not work when energy is below 20
            - Different work types use different skills and give different rewards
            - Working makes you hungry and tired
        """


class RPGCharacter(TinyPerson):
    """Extended TinyPerson class with RPG mechanics"""

    def __init__(self, name=None):
        super().__init__(name=name)

        # Initialize character stats
        self.stats = RPGCharacterStats()

        # Add RPG-specific tools
        self.basic_needs = RPGBasicNeeds(owner=self)
        self.work = RPGWork(owner=self)

        # Add tools to mental faculties
        self.add_mental_faculty(TinyToolUse([self.basic_needs, self.work]))

    def get_status(self):
        """Return current character status"""
        return {
            "name": self.name,
            "level": self.stats.level,
            "experience": self.stats.experience,
            "health": self.stats.health,
            "energy": self.stats.energy,
            "hunger": self.stats.hunger,
            "skills": self.stats.skills
        }