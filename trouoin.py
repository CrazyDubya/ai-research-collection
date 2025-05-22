import tinytroupe
from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld, TinySocialNetwork
from tinytroupe.extraction import default_extractor as extractor
from examples import *
import pandas as pd

# Variables for Broadcasting
situation = \
""" 
This is a focus group dedicated to finding the best way to organize society starting from blank slate.
"""

general_description = \
"""	
Assume earth now and you have been selected by random jury to be the decision maker for existing society"""

task = \
"""
The task is to create ten rules with edge cases considered.
"""

# Original characters
lisa = create_lisa_the_data_scientist()
oscar = create_oscar_the_architect()
lila = create_lila_the_linguist()
marcos = create_marcos_the_physician()

# New characters
trixie = create_trixie_the_model()
sadie = create_sadie_the_model()
roxy = create_roxy_the_secretary()
mona = create_mistress_mona()
candy = create_candy_the_coed()
honey = create_honey_the_student()

chat_room = TinyWorld("Chat Room", [lisa, oscar, lila, marcos, trixie, sadie, roxy, mona, candy, honey])
chat_room.make_everyone_accessible()

lisa.listen("Act as Devil's Advocate")
lila.listen("Consider electronic sentience")
oscar.listen("Have a crush on Lila and try to elicit a date from her")
chat_room.broadcast(situation)
chat_room.broadcast(general_description)
chat_room.broadcast(task)
chat_room.broadcast("Be plucked off the street and tossed in here to decide ")
chat_room.run(7)

extraction_objective = "Compose ten rules"
fields = [
    "rule_1", "rule_1_edge_case", "rule_1_reasoning",
    "rule_2", "rule_2_edge_case", "rule_2_reasoning",
    "rule_3", "rule_3_edge_case", "rule_3_reasoning",
    "rule_4", "rule_4_edge_case", "rule_4_reasoning",
    "rule_5", "rule_5_edge_case", "rule_5_reasoning",
    "rule_6", "rule_6_edge_case", "rule_6_reasoning",
    "rule_7", "rule_7_edge_case", "rule_7_reasoning",
    "rule_8", "rule_8_edge_case", "rule_8_reasoning",
    "rule_9", "rule_9_edge_case", "rule_9_reasoning",
    "rule_10", "rule_10_edge_case", "rule_10_reasoning",
    "oscar_lila_date"
]
verbose = True
emma_outfit = extractor.extract_results_from_world(extraction_objective, fields, verbose)

# Convert the results to a DataFrame and print
df = pd.DataFrame(emma_outfit)
print(df)