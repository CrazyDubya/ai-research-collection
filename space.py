import json
import random
import time
from datetime import datetime, timedelta
import requests
from tinytroupe.agent import TinyPerson, CustomMentalFaculty, RecallFaculty, FilesAndWebGroundingFaculty
from tinytroupe.environment import TinyWorld
from tinytroupe.story import TinyStory
from tinytroupe.extraction import ArtifactExporter
from tinytroupe.enrichment import TinyEnricher
from tinytroupe.tools import TinyWordProcessor

class SpaceStationMentalFaculty(CustomMentalFaculty):
    def __init__(self):
        super().__init__("Space Station Mental Faculty")
        self.add_action("PERFORM_TASK", "Perform a task related to the resident's role", self._process_perform_task)
        self.add_action("INTERACT_WITH_ENVIRONMENT", "Interact with the space station environment", self._process_interact_with_environment)
        self.add_action("SOCIALIZE", "Socialize with other residents", self._process_socialize)

    def _process_perform_task(self, agent, action):
        task_description = self._generate_task_description(agent)
        agent.think(f"I will now {task_description}")
        return True

    def _process_interact_with_environment(self, agent, action):
        interaction_description = self._generate_interaction_description(agent)
        agent.think(f"I will now {interaction_description}")
        return True

    def _process_socialize(self, agent, action):
        target_agent = random.choice(agent.environment.agents)
        if target_agent != agent:
            agent.think(f"I will now socialize with {target_agent.name}")
            agent.listen_and_act(f"Hi {target_agent.name}, how are you doing today?", max_content_length=100)
        return True

    def _generate_task_description(self, agent):
        role = agent.get("occupation")
        task_templates = {
            "Engineer": ["repair the {system}", "perform maintenance on the {system}", "upgrade the {system}"],
            "Doctor": ["treat a patient with {condition}", "perform a {procedure}", "research a new {treatment}"],
            "Scientist": ["conduct an experiment on {subject}", "analyze data from {study}", "write a research paper on {topic}"],
            "Security Officer": ["patrol {location}", "investigate a {incident}", "conduct a security drill on {scenario}"],
            "Chef": ["prepare a {dish}", "manage {inventory}", "design a new {menu}"],
            "Janitor": ["clean {area}", "perform {maintenance}", "restock {supplies}"]
        }
        template = random.choice(task_templates[role])
        return template.format(system=random.choice(["life support", "power grid", "communication array"]),
                               condition=random.choice(["flu", "broken bone", "allergic reaction"]),
                               procedure=random.choice(["surgery", "physical exam", "lab test"]),
                               treatment=random.choice(["medication", "therapy", "vaccine"]),
                               subject=random.choice(["plant growth", "material properties", "animal behavior"]),
                               study=random.choice(["radiation exposure", "microgravity effects", "isolation psychology"]),
                               topic=random.choice(["space agriculture", "asteroid mining", "interplanetary travel"]),
                               location=random.choice(["docking bay", "residential quarters", "science lab"]),
                               incident=random.choice(["theft", "altercation", "unauthorized access"]),
                               scenario=random.choice(["outbreak", "hull breach", "power failure"]),
                               dish=random.choice(["steak", "sushi", "pasta"]),
                               inventory=random.choice(["ingredients", "kitchen equipment", "food supplies"]),
                               menu=random.choice(["breakfast", "lunch", "dinner"]),
                               area=random.choice(["hallway", "bathroom", "dining area"]),
                               maintenance=random.choice(["floor polishing", "filter replacement", "waste disposal"]),
                               supplies=random.choice(["cleaning agents", "paper products", "sanitizers"])
                               )

    def _generate_interaction_description(self, agent):
        interaction_templates = [
            "observe the {object} in the {location}",
            "use the {equipment} in the {location}",
            "read about {subject} in the {location}",
            "enjoy the {activity} in the {location}"
        ]
        template = random.choice(interaction_templates)
        return template.format(object=random.choice(["stars", "Earth", "Moon"]),
                               location=random.choice(["observation deck", "recreation room", "personal quarters"]),
                               equipment=random.choice(["treadmill", "telescope", "virtual reality system"]),
                               subject=random.choice(["space history", "astrophysics", "science fiction"]),
                               activity=random.choice(["view", "music", "game"]))

class SpaceStationWorld(TinyWorld):
    super().__init__(name)
    self.current_datetime = datetime(2100, 1, 1)  # Start date for the simulation
    self.locations = ["Docking Bay", "Residential Quarters", "Science Lab", "Medical Bay", "Engineering", "Observation Deck", "Recreation Room", "Dining Area"]
    self.shared_exporter = ArtifactExporter(base_output_folder="./simulation_artifacts")
    self.shared_enricher = TinyEnricher()

    def _step(self, timedelta_per_step=timedelta(days=1)):
    self._advance_datetime(timedelta_per_step)
    for agent in self.agents:
        agent.act()
        self._handle_actions(agent, agent.pop_latest_actions())
        self._generate_random_event()

    def _generate_random_event(self):
    if random.random() < 0.1:  # 10% chance of a random event occurring
        event_templates = [
            "A {anomaly} has been detected in the {location}.",
            "The {system} is experiencing a {problem}.",
            "An {emergency} has been reported in the {location}."
        ]
        template = random.choice(event_templates)
        event_description = template.format(anomaly=random.choice(["radiation spike", "gravitational anomaly", "unusual energy reading"]),
                                            location=random.choice(self.locations),
                                            system=random.choice(["life support", "power grid", "communication array"]),
                                            problem=random.choice(["malfunction", "overload", "failure"]),
                                            emergency=random.choice(["medical", "security", "environmental"]))
        self.broadcast(f"Attention all residents: {event_description}")

def ollama_generate(prompt, max_retries=3, retry_delay=1):
    data = {
        "prompt": prompt,
        "model": "phi3:14b-medium-128k-instruct-q8_0",
        "stream": False,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_API_URL, json=data)
            response_text = response.text

            try:
                # Try to parse the response as JSON
                response_data = json.loads(response_text)
                return response_data["response"]
            except json.JSONDecodeError:
                # If parsing fails, try to extract the JSON substring from the response
                try:
                    json_start = response_text.index("{")
                    json_end = response_text.rindex("}")
                    json_substring = response_text[json_start : json_end + 1]
                    response_data = json.loads(json_substring)
                    return response_data["response"]
                except (ValueError, KeyError):
                    # If extracting the JSON substring fails, move on to the next attempt
                    pass

        except requests.exceptions.RequestException:
            # If there's an error with the request, move on to the next attempt
            pass

        # Delay before the next attempt
        time.sleep(retry_delay)

    # If all attempts fail, raise an exception
    raise Exception("Failed to get a valid response from the Ollama API after multiple attempts.")
def generate_resident(role, world):
    prompt = f"Generate a realistic space station resident with the role of {role}. Provide their name, age, a brief bio, and a short description of their typical day on the space station."
    resident_data = ollama_generate(prompt)

    if not resident_data:
        raise ValueError("Received empty response from Ollama API")

    try:
        resident_data = json.loads(resident_data)  # Assuming Ollama returns a JSON string
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        print(f"Response text: {resident_data}")
        raise

    resident = TinyPerson(resident_data["name"])
    resident.define("age", resident_data["age"])
    resident.define("occupation", role)
    resident.define("bio", resident_data["bio"])
    resident.add_mental_faculty(SpaceStationMentalFaculty())
    resident.add_mental_faculty(RecallFaculty())
    resident.add_mental_faculty(FilesAndWebGroundingFaculty())

    diary_entry = f"Day 1 on {world.name}:\n\n{resident_data['typical_day']}"
    diary_writer = TinyWordProcessor(owner=resident, exporter=world.shared_exporter, enricher=world.shared_enricher)
    diary_writer.write_document(title=f"{resident.name}'s Diary", content=diary_entry, author=resident.name)

    return resident

def generate_story(world):
    prompt = f"Generate a brief story outline for the space station {world.name} on {world.current_datetime.strftime('%Y-%m-%d')}. Focus on the interactions and events involving the residents."
    story_data = ollama_generate(prompt)
    story_data = json.loads(story_data)  # Assuming Ollama returns a JSON string
    story = TinyStory(world, story_data["story"])

    # Generate a news article based on the story
    article_prompt = f"Write a news article about the following story on the space station {world.name}:\n\n{story_data['story']}"
    article_data = ollama_generate(article_prompt)
    article_writer = TinyWordProcessor(owner=None, exporter=world.shared_exporter, enricher=world.shared_enricher)
    article_writer.write_document(title=f"{world.name} News - {world.current_datetime.strftime('%Y-%m-%d')}", content=article_data, author="Space Station News")

    return story

def run_simulation(num_days):
    world = SpaceStationWorld("Orbital Station Alpha")
    roles = ["Engineer", "Doctor", "Scientist", "Security Officer", "Chef", "Janitor"]
    residents = [generate_resident(role, world) for role in roles]
    world.add_agents(residents)

    for resident in residents:
        resident.read_documents_from_web(["https://en.wikipedia.org/wiki/International_Space_Station",
                                          "https://www.nasa.gov/audience/forstudents/k-4/stories/nasa-knows/what-is-a-space-station-k4.html"])

    for _ in range(num_days):
        world.run(steps=1, timedelta_per_step=timedelta(days=1))
        story = generate_story(world)
        print(f"\nDate: {world.current_datetime.strftime('%Y-%m-%d')}")
        print(f"Story: {story.start_story()}")

        for agent in world.agents:
            print(f"\n{agent.name}'s thoughts and actions:")
            agent.pp_current_interactions()

            diary_entry = f"Dear Diary,\n\nToday on {world.name} was quite eventful. Here are some of my thoughts and experiences:\n\n{agent.pop_actions_and_get_contents_for('TALK', only_last_action=False)}\n\n{agent.name}"
            diary_writer = TinyWordProcessor(owner=agent, exporter=world.shared_exporter, enricher=world.shared_enricher)
            diary_writer.write_document(title=f"{agent.name}'s Diary - {world.current_datetime.strftime('%Y-%m-%d')}", content=diary_entry, author=agent.name)

# Run the simulation for 10 days
run_simulation(num_days=10)