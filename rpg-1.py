from rpg_tools import RPGCharacter


def main():
    # Create a new character
    character = RPGCharacter(name="Alice")

    # Set up initial personality and traits
    character.define("personality_traits", [
        "Ambitious and hardworking",
        "Enjoys learning new things",
        "Values self-improvement"
    ])

    character.define("occupation", "Apprentice Adventurer")
    character.define("age", "22")

    # Initial thoughts
    character.think("I should start my day and work on improving myself.")

    # Work a bit
    character.act()  # Character will decide to work based on their energy/hunger

    # Display status after working
    print("Status after working:")
    print(character.get_status())

    # Make character hungry and tired
    character.stats.hunger = 70
    character.stats.energy = 25

    # Character should decide to eat and rest
    character.think("I'm feeling quite hungry and tired. I should take care of myself.")
    character.act()

    # Display final status
    print("\nFinal status:")
    print(character.get_status())


if __name__ == "__main__":
    main()