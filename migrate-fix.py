import re

def reformat_agent_definitions(input_file, output_file):
    """
    Reads the original file, parses agents, and reformats them to match the examples.py structure.
    """
    with open(input_file, 'r') as infile:
        content = infile.read()

    # Regular expression to find and parse character creation functions
    agent_pattern = re.compile(
        r"def\s+(create_\w+)\(.*?:\s*(.*?)\s*return\s+\w+",
        re.DOTALL
    )
    matches = agent_pattern.findall(content)

    reformatted_agents = []

    for func_name, func_body in matches:
        # Reformat the body to clean up unnecessary lines and maintain consistent spacing
        func_body = re.sub(r"\s*\n\s*\n+", "\n", func_body)  # Remove extra newlines
        func_body = re.sub(r"person\.define", "agent.define", func_body)  # Fix misnamed object calls

        reformatted_agent = f"""
def {func_name}():
    agent = TinyPerson("{func_name.replace('create_', '').replace('_', ' ').title()}")
{func_body}
    return agent
"""
        reformatted_agents.append(reformatted_agent.strip())

    # Write the reformatted agents to the output file
    with open(output_file, 'w') as outfile:
        for agent in reformatted_agents:
            outfile.write(agent + "\n\n")

    print(f"Reformatted agents have been saved to '{output_file}'")


if __name__ == "__main__":
    input_file = "chars/political_peeps.py"  # Replace with the actual filename
    output_file = "fixed_definitions.py"    # Name of the fixed file

    reformat_agent_definitions(input_file, output_file)
