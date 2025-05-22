from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld
import random

# Define Jedi Council Members
def create_yoda():
    yoda = TinyPerson("Yoda")
    yoda.define("age", 900)
    yoda.define("nationality", "Unknown")
    yoda.define("occupation", "Jedi Grand Master")
    yoda.define("personality_traits", [
        "Wise and patient",
        "Deeply connected to the Force",
        "Speaks in unconventional manner"
    ])
    yoda.define("skills", ["Force mastery", "Lightsaber combat", "Teaching"])
    yoda.define("current_goals", ["Maintain peace", "Train new Jedi"])
    yoda.define("current_emotions", "Concerned about the growing darkness")
    return yoda

yoda=create_yoda()

def create_mace_windu():
    mace = TinyPerson("Mace Windu")
    mace.define("age", 53)
    mace.define("nationality", "Human")
    mace.define("occupation", "Jedi Master")
    mace.define("personality_traits", [
        "Stern and powerful",
        "Unwavering in convictions",
        "Strategic thinker"
    ])
    mace.define("skills", ["Vaapad lightsaber form", "Force combat", "Leadership"])
    mace.define("current_goals", ["Protect the Republic", "Counter the Sith threat"])
    mace.define("current_emotions", "Alert and vigilant")
    return mace

mace_windu = create_mace_windu()

def create_ki_adi_mundi():
    # Initialize Ki-Adi-Mundi as a TinyPerson
    ki_adi_mundi = TinyPerson("Ki-Adi-Mundi")

    # Define Ki-Adi-Mundi's attributes using the define method
    ki_adi_mundi.define("age", 65)
    ki_adi_mundi.define("occupation", "Jedi Master")
    ki_adi_mundi.define("personality_traits", [
        "Calm",
        "Analytical",
        "Diplomatic",
        "Reserved"
    ])
    ki_adi_mundi.define("goals", [
        "Represent the Cerean people",
        "Promote unity among species"
    ])
    ki_adi_mundi.define("interests", [
        "Galactic politics",
        "Cultural studies",
        "The Force"
    ])

    return ki_adi_mundi


# Create Ki-Adi-Mundi
ki_adi_mundi = create_ki_adi_mundi()


def create_ahsoka_tano():
    # Initialize Ahsoka Tano as a TinyPerson
    ahsoka_tano = TinyPerson("Ahsoka Tano")

    # Define Ahsoka Tano's attributes using the define method
    ahsoka_tano.define("age", 22)
    ahsoka_tano.define("occupation", "Jedi Knight")
    ahsoka_tano.define("personality_traits", [
        "Curious",
        "Determined",
        "Loyal",
        "Optimistic"
    ])
    ahsoka_tano.define("goals", [
        "Prove herself as a Jedi",
        "Protect her friends"
    ])
    ahsoka_tano.define("interests", [
        "Lightsaber combat",
        "Droids",
        "Exploring new planets"
    ])

    return ahsoka_tano


# Create Ahsoka Tano
ahsoka_tano = create_ahsoka_tano()
# Define a Timeline class to manage events
class Timeline:
    def __init__(self):
        self.events = []
        self.current_time = 0

    def add_event(self, event):
        self.events.append((self.current_time, event))
        self.current_time += 1

    def get_recent_events(self, n=3):
        # Return the most recent n events
        return self.events[-n:] if len(self.events) >= n else self.events

# Define a Memory class for each Jedi
class Memory:
    def __init__(self):
        self.memories = []

    def add_memory(self, memory):
        self.memories.append(memory)

    def recall_memories(self, n=3):
        # Return the most recent n memories
        return self.memories[-n:] if len(self.memories) >= n else self.memories

# Add Memory to each Jedi
yoda.memory = Memory()
mace_windu.memory = Memory()
ki_adi_mundi.memory = Memory()
ahsoka_tano.memory = Memory()
# Create the Jedi Council Environment
jedi_council_chamber = TinyWorld("Jedi Council Chamber")
jedi_council_chamber.broadcast("The Jedi Council convenes to discuss the growing Sith threat.")

# Add the Jedi Council Members to the environment
jedi_council_chamber.add_agents([yoda, mace_windu, ki_adi_mundi, ahsoka_tano])

# Initialize the timeline
timeline = Timeline()

# Define a list of possible events
possible_events = [
    "A Sith Lord has been spotted in the Outer Rim.",
    "A Republic trade ship has been attacked by unknown forces.",
    "A new planet has requested protection from the Jedi.",
    "A Jedi Knight has gone missing in the Unknown Regions.",
    "A Separatist fleet is amassing near the Corellian system."
]

# Function to inject random events
def inject_random_event(timeline, possible_events):
    event = random.choice(possible_events)
    timeline.add_event(event)
    return event

# Define the initial scenario
scenario = "The Jedi Council must decide how to respond to a growing Sith threat in the Outer Rim."

# Start the simulation
for step in range(10):  # Number of interaction rounds
    # Inject a random event at random intervals
    if random.random() < 0.3:  # 30% chance of injecting an event
        event = inject_random_event(timeline, possible_events)
        print(f"** Event at Time {timeline.current_time}: {event} **")

    # Get recent events from the timeline
    recent_events = timeline.get_recent_events(n=2)

    # Allow each Jedi to recall recent memories and react to recent events
    for jedi in [yoda, mace_windu, ki_adi_mundi, ahsoka_tano]:
        # Recall recent memories
        recent_memories = jedi.memory.recall_memories(n=2)
        memory_context = "\n".join(recent_memories)

        # React to recent events
        event_context = "\n".join([f"Time {t}: {e}" for t, e in recent_events])
        response = jedi.listen_and_act(
            f"Recent events:\n{event_context}\n\nYour memories:\n{memory_context}\n\nScenario: {scenario}"
        )
        print(f"{jedi.name}: {response}")

        # Add the response to the Jedi's memory
        jedi.memory.add_memory(f"{jedi.name}: {response}")