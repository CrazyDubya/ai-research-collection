import os
import time
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from openai import OpenAI

# Set up folders to save the responses
os.makedirs('char_xml', exist_ok=True)
os.makedirs('char_json', exist_ok=True)

# Define API configuration for the Hermes model
model_name = 'hermes-3-llama-3.1-405b-fp8-128k'
HERMES_API_KEY = os.getenv('HERMES_API_KEY')
# OpenAI client configuration for Hermes model
client = OpenAI(
    api_key=HERMES_API_KEY,
    base_url="https://api.lambdalabs.com/v1"
)


def print_colored(text, color="white"):
    """Print colored text to the console for better user experience."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")


def send_request(client, prompt, temperature=0.8):
    """Send a request to the Hermes model with the given prompt."""
    print_colored("Sending request to the Hermes model...", "cyan")
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print_colored(f"Error sending request: {e}", "red")
        return None


def parse_xml(xml_string):
    """Parse XML string and handle potential errors."""
    if not xml_string:
        return None

    try:
        # Try to parse the XML directly
        root = ET.fromstring(xml_string)
        return root
    except ET.ParseError as e:
        print_colored(f"Error parsing XML: {e}", "yellow")

        # Try to extract XML content if it's embedded in non-XML text
        try:
            start_idx = xml_string.find('<')
            end_idx = xml_string.rfind('>')
            if start_idx >= 0 and end_idx > start_idx:
                xml_content = xml_string[start_idx:end_idx + 1]
                root = ET.fromstring(xml_content)
                return root
        except:
            pass

        # Final attempt: wrap in a root element
        try:
            fixed_xml = f"<root>{xml_string}</root>"
            root = ET.fromstring(fixed_xml)
            return root
        except ET.ParseError:
            print_colored("Unable to fix XML. Returning None.", "red")
            return None


def get_user_preferences():
    """Get user preferences for character generation."""
    print_colored("\n===== CHARACTER CREATOR SETUP =====", "magenta")
    print_colored("This tool will help you create detailed character profiles for artistic rendering.", "cyan")

    preferences = {}

    # Get gender preference
    print_colored("\nWhat gender would you like the characters to be?", "green")
    print("1. Female")
    print("2. Male")
    print("3. Non-binary")
    print("4. Mixed (variety of genders)")

    gender_choice = input("Enter your choice (1-4): ").strip()
    gender_map = {
        "1": "female",
        "2": "male",
        "3": "non-binary",
        "4": "mixed"
    }
    preferences["gender"] = gender_map.get(gender_choice, "mixed")

    # Get setting/genre preference
    print_colored("\nWhat setting or genre would you like for these characters?", "green")
    print("Examples: modern day, fantasy, sci-fi, historical (1800s), post-apocalyptic, etc.")
    preferences["setting"] = input("Enter setting/genre: ").strip()

    # Get age range preference
    print_colored("\nWhat age range would you like for these characters?", "green")
    print("Examples: young adult (18-25), adult (25-40), middle-aged (40-60), elderly (60+), mixed ages")
    preferences["age_range"] = input("Enter age range: ").strip()

    # Get style preference
    print_colored("\nWhat artistic style are you aiming for?", "green")
    print("Examples: realistic, anime/manga, cartoon, stylized, painterly, etc.")
    preferences["art_style"] = input("Enter artistic style: ").strip()

    # Get additional notes
    print_colored("\nAny additional notes or specific elements you'd like included?", "green")
    print("Examples: focus on diverse body types, include cultural diversity, specific professions, etc.")
    preferences["additional_notes"] = input("Enter additional notes (press Enter to skip): ").strip()

    # Get number of characters
    while True:
        try:
            preferences["num_characters"] = int(input("\nHow many characters would you like to generate? "))
            if preferences["num_characters"] > 0:
                break
            print_colored("Please enter a positive number.", "yellow")
        except ValueError:
            print_colored("Please enter a valid number.", "yellow")

    # Get output format preference
    print_colored("\nChoose output format:", "green")
    print("1. XML only")
    print("2. JSON only")
    print("3. Both XML and JSON")

    format_choice = input("Enter your choice (1-3): ").strip()
    format_map = {
        "1": "xml",
        "2": "json",
        "3": "both"
    }
    preferences["output_format"] = format_map.get(format_choice, "both")

    print_colored("\n===== PREFERENCES SUMMARY =====", "magenta")
    for key, value in preferences.items():
        if key != "num_characters" and key != "output_format":
            print_colored(f"{key.replace('_', ' ').title()}: {value}", "white")

    print_colored(f"Number of characters: {preferences['num_characters']}", "white")
    print_colored(f"Output format: {preferences['output_format']}", "white")
    print_colored("==============================\n", "magenta")

    return preferences


def generate_characters(client, preferences):
    """Generate initial character profiles based on user preferences."""
    chunk_size = 5  # Smaller chunk size for more detailed characters
    all_characters = []
    num_requests = preferences["num_characters"]

    gender = preferences["gender"]
    setting = preferences["setting"]
    age_range = preferences["age_range"]
    additional_notes = preferences["additional_notes"]

    for start in range(0, num_requests, chunk_size):
        end = min(start + chunk_size, num_requests)
        current_chunk_size = end - start
        print_colored(f"Generating characters {start + 1} to {end}...", "cyan")

        # Create a more detailed prompt for initial character generation
        prompt = f"""Generate {current_chunk_size} diverse, visually distinct {gender} characters for a {setting} setting, 
        focusing on characters in the {age_range} age range.

        {additional_notes if additional_notes else ""}

        Each character should have a unique appearance, personality, and background that would make them 
        visually interesting and distinctive for an artist to draw. Focus on creating characters with 
        memorable visual traits and personalities.

        Provide the response in XML format as follows:
        <characters>
          <character>
            <first>FirstName</first>
            <last>LastName</last>
            <age>Age (numerical)</age>
            <gender>Gender</gender>
            <ethnicity>Ethnicity/Cultural Background</ethnicity>
            <occupation>Occupation/Role</occupation>
            <background>Detailed background story (at least 3-4 sentences)</background>
            <key_visual_traits>3-5 distinctive visual characteristics that define this character's appearance</key_visual_traits>
          </character>
          ...
        </characters>"""

        response = send_request(client, prompt)
        root = parse_xml(response)

        if root is not None:
            for character in root.findall('.//character'):
                try:
                    char_data = {
                        'first': character.find('first').text.strip() if character.find(
                            'first') is not None and character.find('first').text else "Unknown",
                        'last': character.find('last').text.strip() if character.find(
                            'last') is not None and character.find('last').text else "Unknown",
                        'age': character.find('age').text.strip() if character.find(
                            'age') is not None and character.find('age').text else "Unknown",
                        'gender': character.find('gender').text.strip() if character.find(
                            'gender') is not None and character.find('gender').text else gender,
                        'ethnicity': character.find('ethnicity').text.strip() if character.find(
                            'ethnicity') is not None and character.find('ethnicity').text else "Unknown",
                        'occupation': character.find('occupation').text.strip() if character.find(
                            'occupation') is not None and character.find('occupation').text else "Unknown",
                        'background': character.find('background').text.strip() if character.find(
                            'background') is not None and character.find('background').text else "Unknown",
                        'key_visual_traits': character.find('key_visual_traits').text.strip() if character.find(
                            'key_visual_traits') is not None and character.find('key_visual_traits').text else "Unknown"
                    }
                    all_characters.append(char_data)
                except Exception as e:
                    print_colored(f"Error processing character data: {e}", "red")
                    continue

    return all_characters


def expand_character_details(client, character, preferences):
    """Expand character details with rich visual and personality information."""
    art_style = preferences["art_style"]
    setting = preferences["setting"]

    prompt = f"""Based on the character {character['first']} {character['last']}, age {character['age']}, 
    {character['gender']}, {character['ethnicity']}, who works as a {character['occupation']} 
    with the following background: {character['background']}

    And these key visual traits: {character['key_visual_traits']}

    Create an extremely detailed character profile suitable for an artist to draw in a {art_style} style 
    for a {setting} setting. Focus on visual details that would help an artist create a complete 
    and consistent character design.

    Provide the response in XML format as follows:
    <character_details>
      <physical_appearance>
        <face>Detailed description of facial features, expressions, and distinguishing marks</face>
        <body>Body type, height, build, posture, and distinctive physical characteristics</body>
        <hair>Hair style, color, texture, length, and how it's typically worn</hair>
        <eyes>Eye color, shape, and any distinctive eye features</eyes>
        <skin>Skin tone, texture, and any markings like scars, tattoos, or birthmarks</skin>
      </physical_appearance>

      <wardrobe>
        <everyday_outfits>3-5 complete everyday outfit descriptions with colors, materials, and style</everyday_outfits>
        <formal_attire>Description of what they wear for formal occasions</formal_attire>
        <casual_wear>Description of their casual, relaxed clothing</casual_wear>
        <accessories>Jewelry, bags, hats, glasses, and other accessories they commonly wear</accessories>
        <footwear>Different types of shoes/footwear they own and when they wear them</footwear>
        <signature_item>One signature clothing item or accessory that defines their look</signature_item>
      </wardrobe>

      <personality>
        <traits>5-7 key personality traits with brief explanations of how they manifest</traits>
        <mannerisms>Specific gestures, expressions, or habits that are characteristic</mannerisms>
        <speech_pattern>How they talk, vocabulary choices, accent, or verbal tics</speech_pattern>
        <values>Core beliefs and what matters most to them</values>
        <fears>What frightens or worries them</fears>
        <motivations>What drives them and their goals</motivations>
      </personality>

      <background_details>
        <formative_experiences>Key events that shaped who they are</formative_experiences>
        <relationships>Important relationships and how they influence the character</relationships>
        <skills>Special abilities, training, or knowledge they possess</skills>
        <possessions>Important items they own beyond clothing</possessions>
      </background_details>

      <artistic_notes>
        <color_palette>Suggested color palette for this character</color_palette>
        <poses>Characteristic poses or stances that reflect their personality</poses>
        <expressions>Typical facial expressions that convey their character</expressions>
        <lighting>Lighting suggestions that would complement this character</lighting>
      </artistic_notes>
    </character_details>

    Be extremely detailed and specific in your descriptions, especially for visual elements. 
    Focus on concrete details rather than abstract concepts. For example, instead of saying 
    "she dresses fashionably," describe the specific style, pieces, colors, and how they're worn.
    """

    print_colored(f"Generating detailed profile for {character['first']} {character['last']}...", "cyan")
    response = send_request(client, prompt, temperature=0.7)
    root = parse_xml(response)

    if root is None:
        print_colored(f"Failed to generate details for {character['first']} {character['last']}", "red")
        return character

    # Extract all the detailed information
    character['details'] = {}

    # Process physical appearance
    physical = root.find('.//physical_appearance')
    if physical is not None:
        character['details']['physical_appearance'] = {}
        for elem in physical:
            if elem.text:
                character['details']['physical_appearance'][elem.tag] = elem.text.strip()

    # Process wardrobe
    wardrobe = root.find('.//wardrobe')
    if wardrobe is not None:
        character['details']['wardrobe'] = {}
        for elem in wardrobe:
            if elem.text:
                character['details']['wardrobe'][elem.tag] = elem.text.strip()

    # Process personality
    personality = root.find('.//personality')
    if personality is not None:
        character['details']['personality'] = {}
        for elem in personality:
            if elem.text:
                character['details']['personality'][elem.tag] = elem.text.strip()

    # Process background details
    background = root.find('.//background_details')
    if background is not None:
        character['details']['background_details'] = {}
        for elem in background:
            if elem.text:
                character['details']['background_details'][elem.tag] = elem.text.strip()

    # Process artistic notes
    artistic = root.find('.//artistic_notes')
    if artistic is not None:
        character['details']['artistic_notes'] = {}
        for elem in artistic:
            if elem.text:
                character['details']['artistic_notes'][elem.tag] = elem.text.strip()

    return character


def save_character_xml(character):
    """Save character data as XML file."""
    root = ET.Element("character")

    # Add basic information
    ET.SubElement(root, "first").text = character['first']
    ET.SubElement(root, "last").text = character['last']
    ET.SubElement(root, "age").text = character['age']
    ET.SubElement(root, "gender").text = character['gender']
    ET.SubElement(root, "ethnicity").text = character['ethnicity']
    ET.SubElement(root, "occupation").text = character['occupation']
    ET.SubElement(root, "background").text = character['background']
    ET.SubElement(root, "key_visual_traits").text = character['key_visual_traits']

    # Add detailed information
    if 'details' in character:
        details = ET.SubElement(root, "details")

        # Add physical appearance
        if 'physical_appearance' in character['details']:
            physical = ET.SubElement(details, "physical_appearance")
            for key, value in character['details']['physical_appearance'].items():
                ET.SubElement(physical, key).text = value

        # Add wardrobe
        if 'wardrobe' in character['details']:
            wardrobe = ET.SubElement(details, "wardrobe")
            for key, value in character['details']['wardrobe'].items():
                ET.SubElement(wardrobe, key).text = value

        # Add personality
        if 'personality' in character['details']:
            personality = ET.SubElement(details, "personality")
            for key, value in character['details']['personality'].items():
                ET.SubElement(personality, key).text = value

        # Add background details
        if 'background_details' in character['details']:
            background = ET.SubElement(details, "background_details")
            for key, value in character['details']['background_details'].items():
                ET.SubElement(background, key).text = value

        # Add artistic notes
        if 'artistic_notes' in character['details']:
            artistic = ET.SubElement(details, "artistic_notes")
            for key, value in character['details']['artistic_notes'].items():
                ET.SubElement(artistic, key).text = value

    # Convert to pretty XML
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    filename = f"char_xml/{character['first']}_{character['last']}_{int(time.time())}.xml"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(xml_str)

    print_colored(f"Character saved to {filename}", "green")
    return filename


def save_character_json(character):
    """Save character data as JSON file."""
    filename = f"char_json/{character['first']}_{character['last']}_{int(time.time())}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(character, f, indent=2, ensure_ascii=False)

    print_colored(f"Character saved to {filename}", "green")
    return filename


def display_character_summary(character):
    """Display a summary of the character for the user."""
    print_colored("\n===== CHARACTER SUMMARY =====", "magenta")
    print_colored(f"Name: {character['first']} {character['last']}", "cyan")
    print_colored(f"Age: {character['age']} | Gender: {character['gender']} | Ethnicity: {character['ethnicity']}",
                  "cyan")
    print_colored(f"Occupation: {character['occupation']}", "cyan")

    print_colored("\nBackground:", "yellow")
    print(character['background'])

    print_colored("\nKey Visual Traits:", "yellow")
    print(character['key_visual_traits'])

    if 'details' in character and 'physical_appearance' in character['details']:
        print_colored("\nPhysical Appearance Highlights:", "yellow")
        if 'face' in character['details']['physical_appearance']:
            print(f"Face: {character['details']['physical_appearance']['face'][:100]}...")
        if 'hair' in character['details']['physical_appearance']:
            print(f"Hair: {character['details']['physical_appearance']['hair'][:100]}...")

    if 'details' in character and 'wardrobe' in character['details'] and 'signature_item' in character['details'][
        'wardrobe']:
        print_colored("\nSignature Item:", "yellow")
        print(character['details']['wardrobe']['signature_item'])

    print_colored("=============================\n", "magenta")


def main():
    """Main function to run the character creator."""
    print_colored("\n🎭 WELCOME TO THE CHARACTER CREATOR 🎭", "magenta")
    print_colored("This tool helps you generate detailed character profiles for artistic rendering.", "cyan")

    # Check for API key
    if not HERMES_API_KEY:
        print_colored("ERROR: HERMES_API_KEY environment variable not set.", "red")
        print_colored("Please set your API key with: export HERMES_API_KEY='your_api_key'", "yellow")
        return

    # Get user preferences
    preferences = get_user_preferences()

    # Generate initial characters
    characters = generate_characters(client, preferences)

    if not characters:
        print_colored("No characters were generated. Please try again.", "red")
        return

    print_colored(f"\nSuccessfully generated {len(characters)} initial character profiles!", "green")
    print_colored("Now expanding each character with detailed information...", "cyan")

    # Expand each character with detailed information
    expanded_characters = []
    for character in characters:
        try:
            expanded_character = expand_character_details(client, character, preferences)
            expanded_characters.append(expanded_character)

            # Display a summary of the character
            display_character_summary(expanded_character)

            # Save the character based on user preference
            if preferences["output_format"] in ["xml", "both"]:
                xml_file = save_character_xml(expanded_character)

            if preferences["output_format"] in ["json", "both"]:
                json_file = save_character_json(expanded_character)

            # Give the model a short break between characters
            time.sleep(1)

        except Exception as e:
            print_colored(f"Error processing character {character['first']} {character['last']}: {e}", "red")

    print_colored(f"\n✅ Successfully created {len(expanded_characters)} detailed character profiles!", "green")
    print_colored("Files have been saved in the char_xml and/or char_json directories.", "cyan")
    print_colored("These profiles contain rich details for artists to create complete character designs.", "cyan")
    print_colored("\nThank you for using the Character Creator! 🎨", "magenta")


if __name__ == "__main__":
    main()
