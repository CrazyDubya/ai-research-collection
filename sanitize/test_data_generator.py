"""
Text Attack Vector Generator (Python)
Generates sophisticated test cases for the text sanitizer
"""

import os
import asyncio
import aiofiles
from pathlib import Path


async def generate_attack_test_data():
    """
    Generate sophisticated attack test cases
    """
    print('Generating sophisticated attack test cases...')
    attack_vectors = {}

    # Create output directory
    output_dir = Path('test_data')
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Basic XSS attacks
    attack_vectors['xss_basic'] = '<script>alert("XSS");</script>'
    attack_vectors['xss_img'] = '<img src="x" onerror="alert(\'XSS\')" />'
    attack_vectors['xss_iframe'] = '<iframe src="javascript:alert(\'XSS\')"></iframe>'

    # 2. Advanced XSS with encoding tricks
    attack_vectors['xss_encoded'] = '&lt;script&gt;alert("XSS");&lt;/script&gt;'
    attack_vectors['xss_unicode'] = '<scr\u0069pt>alert("XSS");</scr\u0069pt>'
    attack_vectors[
        'xss_data_url'] = '<a href="data:text/html;base64,PHNjcmlwdD5hbGVydCgnWFNTJyk7PC9zY3JpcHQ+">Click me</a>'

    # 3. Zero-width character attacks
    zero_width_space = '\u200B'
    zero_width_non_joiner = '\u200C'
    zero_width_joiner = '\u200D'
    left_to_right_mark = '\u200E'
    right_to_left_mark = '\u200F'

    attack_vectors[
        'zero_width_simple'] = f"Hidden{zero_width_space}{zero_width_space}{zero_width_space}text{zero_width_space}between{zero_width_space}words"
    attack_vectors[
        'zero_width_complex'] = f"Text with {zero_width_space}hidden{zero_width_non_joiner} script{left_to_right_mark} tags{right_to_left_mark}: <{zero_width_joiner}script{zero_width_space}>alert(\"Hidden\");<{zero_width_space}/script{zero_width_non_joiner}>"

    # 4. Bidirectional text attacks
    rtl_override = '\u202E'  # RIGHT-TO-LEFT OVERRIDE
    ltr_override = '\u202D'  # LEFT-TO-RIGHT OVERRIDE
    pop_directional_formatting = '\u202C'  # POP DIRECTIONAL FORMATTING

    attack_vectors[
        'bidi_simple'] = f"Normal text {rtl_override}txet desrever{pop_directional_formatting} back to normal"
    attack_vectors[
        'bidi_disguise'] = f"Safe filename {rtl_override}gpj.elifcod{pop_directional_formatting}.txt"  # Looks like safe.txt but is actually safe.docfile.jpg
    attack_vectors[
        'bidi_complex'] = f"{ltr_override}Start {rtl_override}elddim{pop_directional_formatting} end{pop_directional_formatting}"

    # 5. Control character injection
    control_chars = ''.join(chr(i) for i in range(0, 32))
    attack_vectors['control_chars'] = f"Text with{control_chars[:10]}control{control_chars[10:20]}characters"
    attack_vectors['control_chars_hidden'] = f"Text with hidden{chr(0) * 20}control{chr(31) * 20}characters"

    # 6. SQL injection attacks
    attack_vectors['sql_injection_simple'] = "Robert'); DROP TABLE Students;--"
    attack_vectors['sql_injection_complex'] = "1' OR '1'='1"
    attack_vectors['sql_injection_union'] = "1' UNION SELECT username, password FROM users--"

    # 7. Emoji attacks (unusual sequences and modifiers)
    attack_vectors['emoji_simple'] = "Text with emojis 😀😈👾 that should be normalized"
    attack_vectors[
        'emoji_zalgo'] = "Z͉̩͖̠͈̞̟͍̬͓̦̜̻͖̹͔̬̭͢͝ͅa̶̵͓̖̦̗̙͖̮̥͔͕͘͟l̨̢̜̘̖͓̜̘̟̭̘̣̰̟̘̫ͅg͏̶͎̟̣̬̥̣͎̝̀͜͜o̢̢̡̰̖͉̞̙̣̮̥̰̥͉͕͇̕ͅͅ ̷̛̻̪̘͈͎̀͝t̛̮̲̩̟͟e̵̷̢̯̗̰̹̠̰̻̥̻̞̭̝͙̩͘x̴̡̪̫̜̟̥͙̬̀t̶̡͇̫̫̞̺̞̙̞̟̲͎̫͜ͅͅ with emojis 👨‍👩‍👧‍👦👨‍❤️‍💋‍👨"
    attack_vectors['emoji_flags'] = "Flag sequences 🇺🇸🇬🇧🇨🇦🇦🇺 and skin tones 👍🏻👍🏼👍🏽👍🏾👍🏿"

    # 8. Homograph attacks (lookalike characters)
    attack_vectors['homograph_cyrillic'] = "www.аpple.com"  # 'а' is Cyrillic, not Latin
    attack_vectors['homograph_greek'] = "www.gοοgle.com"  # 'ο' is Greek omicron, not 'o'
    attack_vectors['homograph_mixed'] = "Μicrosoft.com"  # First letter is Greek Mu, not Latin M

    # 9. Mixed attacks combining multiple vectors
    attack_vectors[
        'mixed_complex'] = f"{rtl_override}Normal text <{zero_width_space}script{zero_width_joiner}>{zero_width_space}alert(\"{control_chars[0]}XSS{control_chars[1]}\");</{zero_width_space}script{zero_width_non_joiner}>{pop_directional_formatting} with SQL: {attack_vectors['sql_injection_simple']}"

    # 10. Large payloads (10KB, 100KB)
    base_xss_payload = '<script>alert("XSS")</script>'
    attack_vectors['large_10kb'] = base_xss_payload * 200  # ~10KB
    attack_vectors['large_100kb'] = base_xss_payload * 2000  # ~100KB

    # 11. JavaScript obfuscation attack
    attack_vectors['js_obfuscation'] = 'javascript:eval("\\x61\\x6c\\x65\\x72\\x74\\x28\\x27\\x58\\x53\\x53\\x27\\x29")'

    # 12. HTML attribute attacks
    attack_vectors[
        'attr_attack'] = '<div onclick="alert(\'XSS\')" onmouseover="alert(\'XSS\')" onfocus="alert(\'XSS\')">Click me</div>'

    # 13. CSS attacks
    attack_vectors['css_attack'] = '<style>@import url("javascript:alert(\'XSS\')");</style>'

    # 14. SVG-based attacks
    attack_vectors['svg_attack'] = '<svg><script>alert(\'XSS\')</script></svg>'

    # Write attack vectors to files asynchronously
    write_tasks = []
    for name, data in attack_vectors.items():
        file_path = output_dir / f"{name}.txt"
        write_tasks.append(write_file(file_path, data))

    await asyncio.gather(*write_tasks)

    print(f"Generated {len(attack_vectors)} attack vectors in {output_dir}")
    return attack_vectors


async def write_file(file_path: Path, content: str) -> None:
    """Write content to a file asynchronously"""
    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
        await f.write(content)


if __name__ == '__main__':
    asyncio.run(generate_attack_test_data())