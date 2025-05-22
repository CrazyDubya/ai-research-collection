import os
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
from openai import OpenAI

# Set up the folder to save the XML responses
os.makedirs('char_xml', exist_ok=True)

# Define API configuration for the Hermes model
model_name = 'deepseek-llama3.3-70b'
HERMES_API_KEY = os.getenv('HERMES_API_KEY')
# OpenAI client configuration for Hermes model
client = OpenAI(
    api_key=HERMES_API_KEY,
    base_url="https://api.lambdalabs.com/v1"
)


def send_request(client, prompt):
    print(f"Sending request to the Hermes model...")
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_name
    )
    return response.choices[0].message.content


def parse_xml(xml_string):
    try:
        root = ET.fromstring(xml_string)
        return root
    except ET.ParseError:
        print("Error parsing XML. Attempting to fix...")
        fixed_xml = f"<root>{xml_string}</root>"
        try:
            root = ET.fromstring(fixed_xml)
            return root
        except ET.ParseError:
            print("Unable to fix XML. Returning None.")
            return None


def generate_characters(client, num_requests):
    chunk_size = 20
    all_characters = []

    for start in range(0, num_requests, chunk_size):
        end = min(start + chunk_size, num_requests)
        current_chunk_size = end - start
        print(f"Generating characters {start + 1} to {end}")

        prompt = f"""Generate {current_chunk_size} diverse female characters with first name, last name, and a brief background.
        Provide the response in XML format as follows:
        <characters>
          <character>
            <first>FirstName</first>
            <last>LastName</last>
            <background>Brief background</background>
          </character>
          ...
        </characters>"""

        response = send_request(client, prompt)
        root = parse_xml(response)

        if root is not None:
            for character in root.findall('.//character'):
                all_characters.append({
                    'first': character.find('first').text,
                    'last': character.find('last').text,
                    'background': character.find('background').text
                })

    return all_characters


def expand_character(client, character):
    prompt = f"""Based on the name {character['first']} {character['last']} and background: {character['background']},
    expand on the following details. Provide the response in XML format as follows:
    <details>
      <detailed_outfit_incl_shoes_undergarments>{{Detailed expansion}}</detailed_outfit_incl_shoes_undergarments>
      <detailed_fancy_outfit_incl_shoes_undergarments>{{Detailed expansion}}</detailed_outfit_incl_shoes_undergarments>
      <detailed_pool_outfit_incl_sandals_coverup>{{Detailed expansion}}</detailed_pool_outfit_incl_sandals_coverup>
      <detailed_gym_outfit_incl_sneakers_underwear>{{Detailed expansion}}</detailed_gym_outfit_incl_sneakers_underwear>
      <detailed_casual_outfit_incl_shoes_underwear>{{Detailed expansion}}</detailed_casual_outfit_incl_shoes_underwear> 
        <detailed_sleepwear>{{Detailed expansion}}</detailed_sleepwear>
        <detailed_sexy_sleepwear>{{Detailed expansion}}</detailed_sexy_sleepwear> 
      <submissive_willing_tasks>{{Detailed expansion}}</submissive_willing_tasks>
      <submissive_personality>{{Detailed expansion}}</submissive_personality>
      <submissive_scene>{{Detailed expansion}}</submissive_scene>
      <submissive_punish_scene>{{Detailed expansion}}</submissive_punish_scene>
      <sunbissive_sexy_scene>{{Detailed expansion}}</submissive_sexy_scene>
      <submissive_tasks>{{Detailed expansion}}</submissive_tasks>
    </details>
    Please be detailed and verbose in your expansions."""

    response = send_request(client, prompt)
    root = parse_xml(response)

    if root is not None:
        character['details'] = {
            'detailed_outfit_incl_shoes_under': root.find('.//detailed_outfit_incl_shoes_under').text,
            'submissive_willing_tasks': root.find('.//submissive_willing_tasks').text,
            'submissive_personality': root.find('.//submissive_personality').text,
            'submissive_scene': root.find('.//submissive_scene').text,
            'submissive_tasks': root.find('.//submissive_tasks').text
        }

    return character


def save_character_xml(character):
    root = ET.Element("character")
    ET.SubElement(root, "first").text = character['first']
    ET.SubElement(root, "last").text = character['last']
    ET.SubElement(root, "background").text = character['background']
    details = ET.SubElement(root, "details")
    for key, value in character['details'].items():
        ET.SubElement(details, key).text = value

    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    filename = f"char_xml/{character['first']}_{character['last']}_{int(time.time())}.xml"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(xml_str)

    print(f"Character saved to {filename}")


def main():
    num_requests = int(input("How many characters do you want to generate? "))

    characters = generate_characters(client, num_requests)

    for character in characters:
        print(f"Expanding details for {character['first']} {character['last']}...")
        expanded_character = expand_character(client, character)
        save_character_xml(expanded_character)


if __name__ == "__main__":
    main()
