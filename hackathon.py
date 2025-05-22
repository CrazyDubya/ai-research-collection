# hackathon.py
import tinytroupe
from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld, TinySocialNetwork
from tinytroupe.extraction import default_extractor as extractor
from examples import *
import pandas as pd


def run_hackathon():
    # Variables for Broadcasting
    situation = \
        """
        This is a coding challenge to create five standalone HTML pages that demonstrate various web 
        development techniques using HTML, CSS, and JavaScript. The goal is to showcase what can be 
        achieved with minimal local resources by leveraging CDNs for external libraries and assets.
        """

    general_description = \
        """
        Assume you are a team of web developers participating in a coding challenge. Your task is to 
        create five impressive standalone HTML pages, each focusing on a different aspect of web 
        development. You should aim to use inline styles and scripts, and rely on CDNs for any 
        external dependencies to keep the pages self-contained and easily shareable.
        """

    task = \
        """
        Create five HTML pages with the following themes:
        1. An interactive data visualization page
        2. A page with engaging animations and transitions
        3. A responsive page that adapts to different screen sizes
        4. A page that demonstrates form validation and submission
        5. An accessible page that follows web accessibility guidelines
    
        Each page should be a complete, standalone HTML file with inline CSS and JavaScript. Use CDNs 
        to include any external libraries or assets. Provide clear comments in the code to explain 
        the key techniques and approaches used.
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

    coding_team = TinyWorld("Coding Team", [lisa, oscar, lila, marcos, trixie, sadie, roxy, mona, candy, honey])
    coding_team.make_everyone_accessible()

    lisa.listen("Focus on creating an impressive data visualization using D3.js")
    lila.listen("Ensure the pages are accessible and follow WCAG guidelines")
    oscar.listen("Implement a responsive layout using CSS Grid and Flexbox")
    coding_team.broadcast(situation)
    coding_team.broadcast(general_description)
    coding_team.broadcast(task)
    coding_team.broadcast("Work together as a team to create five standalone HTML pages")
    coding_team.run(10)

    extraction_objective = "Compose the five HTML pages"
    fields = [
        "data_viz_html",
        "animation_html",
        "responsive_html",
        "form_html",
        "accessibility_html"
    ]
    verbose = True
    html_pages = extractor.extract_results_from_world(extraction_objective, fields, verbose)

    # Convert the results to a DataFrame and return
    df = pd.DataFrame(html_pages)
    print(df)
    return df


if __name__ == "__main__":
    run_hackathon()