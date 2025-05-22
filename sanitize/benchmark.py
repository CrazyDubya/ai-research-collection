"""
Optimized Text Sanitizer Benchmark (Python)
Efficiently tests sanitizer performance and effectiveness against various attacks
"""

import os
import re
import json
import time
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

# Import sanitizer functions
from python_text_sanitizer import (
    sanitize_base,
    sanitize_html,
    sanitize_sql,
    retype_text,
    ultimate_sanitizer
)

# Benchmark results storage
benchmark_results = {
    "summary": {},
    "timings": [],
    "attackResults": []
}

# Track current test size for calculations
current_test_size = 0


def measure_step(name: str, fn: Callable) -> Any:
    """
    Measure the performance of a function

    Args:
        name: Name of the step
        fn: Function to measure

    Returns:
        Result of the function
    """
    global benchmark_results, current_test_size

    t0 = time.time()
    result = fn()
    t1 = time.time()

    duration_ms = (t1 - t0) * 1000  # Convert to milliseconds

    benchmark_results["timings"].append({
        "step": name,
        "duration": duration_ms,
        "charsPerMs": current_test_size / duration_ms if duration_ms > 0 else 0
    })

    print(f"{name}: {duration_ms:.3f}ms")
    return result


def analyze_attack_effect(name: str, original: str, sanitized: str) -> Dict[str, Any]:
    """
    Analyze if an attack was successfully neutralized

    Args:
        name: Attack type name
        original: Original text
        sanitized: Sanitized text

    Returns:
        Analysis results
    """
    result = {
        "attackType": name,
        "neutralized": True,
        "details": []
    }

    # Analyze text properties
    original_props = {
        "hasScriptTags": bool(re.search(r'<script|</script|javascript:', original, re.I)),
        "hasHtmlTags": bool(re.search(r'<[^>]*>', original, re.I)),
        "hasControlChars": bool(re.search(r'[\x00-\x1F\x7F-\x9F]', original)),
        "hasZeroWidthChars": bool(re.search(r'[\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF]', original)),
        "hasBidiChars": bool(re.search(r'[\u061C\u200E\u200F\u202A-\u202E\u2066-\u2069]', original)),
        "hasEmojis": bool(
            re.search(r'[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U00002600-\U000026FF\U00002700-\U000027BF]',
                      original)),
        "hasSqlPatterns": bool(
            re.search(r'[\'";].*(?:--|drop|select|insert|update|delete|alter|create)\s+\w+', original, re.I))
    }

    sanitized_props = {
        "hasScriptTags": bool(re.search(r'<script|</script|javascript:', sanitized, re.I)),
        "hasHtmlTags": bool(re.search(r'<[^>]*>', sanitized, re.I)),
        "hasControlChars": bool(re.search(r'[\x00-\x1F\x7F-\x9F]', sanitized)),
        "hasZeroWidthChars": bool(re.search(r'[\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF]', sanitized)),
        "hasBidiChars": bool(re.search(r'[\u061C\u200E\u200F\u202A-\u202E\u2066-\u2069]', sanitized)),
        "hasEmojis": bool(
            re.search(r'[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U00002600-\U000026FF\U00002700-\U000027BF]',
                      sanitized)),
        "hasSqlPatterns": bool(
            re.search(r'[\'";].*(?:--|drop|select|insert|update|delete|alter|create)\s+\w+', sanitized, re.I))
    }

    # Check specific attack vectors
    if 'xss' in name:
        if sanitized_props["hasScriptTags"] or sanitized_props["hasHtmlTags"]:
            result["neutralized"] = False
            result["details"].append('HTML/Script tags still present')
        else:
            result["details"].append('XSS attack neutralized')

    if 'zero_width' in name:
        if sanitized_props["hasZeroWidthChars"]:
            result["neutralized"] = False
            result["details"].append('Zero-width characters still present')
        else:
            result["details"].append('Zero-width characters removed')

    if 'bidi' in name:
        if sanitized_props["hasBidiChars"]:
            result["neutralized"] = False
            result["details"].append('Bidirectional control characters still present')
        else:
            result["details"].append('Bidirectional control characters removed')

    if 'control' in name:
        if sanitized_props["hasControlChars"]:
            result["neutralized"] = False
            result["details"].append('Control characters still present')
        else:
            result["details"].append('Control characters removed')

    if 'sql' in name:
        if sanitized_props["hasSqlPatterns"]:
            result["neutralized"] = False
            result["details"].append('SQL injection patterns not properly escaped')
        else:
            result["details"].append('SQL injection patterns neutralized')

    # For mixed attacks, check all vectors
    if 'mixed' in name:
        issues = []

        if sanitized_props["hasScriptTags"] or sanitized_props["hasHtmlTags"]:
            issues.append('HTML/Script tags')
        if sanitized_props["hasControlChars"]:
            issues.append('Control characters')
        if sanitized_props["hasZeroWidthChars"]:
            issues.append('Zero-width characters')
        if sanitized_props["hasBidiChars"]:
            issues.append('Bidirectional characters')
        if sanitized_props["hasSqlPatterns"]:
            issues.append('SQL patterns')

        if issues:
            result["neutralized"] = False
            result["details"].append(f'Mixed attack not fully neutralized. Issues: {", ".join(issues)}')
        else:
            result["details"].append('Mixed attack neutralized successfully')

    return result


async def generate_test_data() -> Dict[str, str]:
    """
    Generate test data of different sizes and attack vectors

    Returns:
        Dictionary of test data
    """
    print('Generating test data...')
    test_data = {}
    output_dir = Path('benchmark_results')

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)

    # Normal text samples
    normal_text = 'This is a normal text sample with no malicious content. It contains regular punctuation, numbers (123456), and symbols (@#$%). This text should pass through the sanitizer with minimal changes.'

    # Generate various sizes of normal text
    test_data['normal_100b'] = normal_text  # ~100 bytes
    test_data['normal_1kb'] = normal_text * 10  # ~1KB
    test_data['normal_10kb'] = normal_text * 100  # ~10KB
    test_data['normal_100kb'] = normal_text * 1000  # ~100KB
    test_data['normal_1mb'] = normal_text * 10000  # ~1MB
    test_data['normal_10mb'] = normal_text * 100000  # ~10MB

    # Attack vectors
    test_data[
        'xss_attack'] = 'Here is some text <script>alert("XSS")</script> with embedded malicious code <img src="x" onerror="alert(\'XSS\')" /> and more <iframe src="javascript:alert(\'XSS\')"></iframe>.'

    test_data[
        'zero_width'] = 'This text has hidden\u200B\u200B\u200B\u200B\u200D\u200D\u200D\u200D zero-width\u200B\u200B\u200B\u200B\u200D\u200D\u200D\u200D characters\u200B\u200B\u200B\u200B\u200D\u200D\u200D\u200D between\u200B\u200B\u200B\u200B\u200D\u200D\u200D\u200D words.'

    test_data['bidi_attack'] = 'Normal text \u202Edetrevni eb lliw siht\u202C but this is normal again.'

    test_data[
        'emoji_attack'] = 'Text with emojis 😀😈👾 and unusual combinations like 👨‍👩‍👧‍👦 that may need special handling.'

    test_data['control_chars'] = 'Text with control\x00chars\x1Fthat\x07should\x1Dbe\x0Fremoved.'

    test_data['sql_injection'] = "Text with SQL injection patterns like: '; DROP TABLE users; --"

    test_data[
        'mixed_attack'] = 'Normal text <script>alert("XSS")</script> with\u202E SQL\'injection\u202C\x00and\x1Fhidden\u200B\u200B\u200B\u200B\u200D\u200D\u200D\u200Dcharacters plus 😈emojis😈.'

    # URI-based attack
    test_data[
        'uri_attack'] = 'Link to <a href="javascript:alert(\'XSS\')">click me</a> or <a href="data:text/html;base64,PHNjcmlwdD5hbGVydCgnWFNTJyk7PC9zY3JpcHQ+">data URL</a>'

    # CSS attack
    test_data[
        'css_attack'] = '<div style="background:url(javascript:alert(\'XSS\'))">Styled div</div><style>@import url("javascript:alert(\'XSS\')");</style>'

    # Template injection
    test_data[
        'template_attack'] = 'Template with ${alert(\'XSS\')} or {{constructor.constructor(\'alert(\\\'XSS\\\')\')()}}'

    # Homograph attack
    test_data['homograph_attack'] = 'www.аpple.com'  # 'а' is Cyrillic, not Latin

    # Write test data to files asynchronously
    write_tasks = []
    for name, data in test_data.items():
        file_path = output_dir / f"{name}.txt"
        write_tasks.append(write_file(file_path, data))

    await asyncio.gather(*write_tasks)
    return test_data


async def write_file(file_path: Path, content: str) -> None:
    """Write content to a file asynchronously"""
    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
        await f.write(content)


async def run_benchmark(name: str, text: str, max_length: int = 10000000) -> Dict[str, Any]:
    """
    Run benchmark on a text sample

    Args:
        name: Test case name
        text: Text to process
        max_length: Maximum allowed length

    Returns:
        Benchmark results
    """
    global current_test_size, benchmark_results

    print(f"\nRunning benchmark on: {name}")
    print(f"Input size: {len(text)} characters")

    current_test_size = len(text)

    sanitized = ""

    # Run the sanitizer
    start_time = time.time()

    if len(text) > 500000:
        # For large texts, skip sub-step measurements
        sanitized = ultimate_sanitizer(text, max_length)
    else:
        # For smaller texts, measure individual steps
        sanitized = measure_step('1. Length limit', lambda: text[:max_length])
        sanitized = measure_step('2. Base sanitization', lambda: sanitize_base(sanitized))
        sanitized = measure_step('3. HTML sanitization', lambda: sanitize_html(sanitized))
        sanitized = measure_step('4. SQL sanitization', lambda: sanitize_sql(sanitized))
        sanitized = measure_step('5. Retyping', lambda: retype_text(sanitized))

    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000

    # Record results
    result = {
        "name": name,
        "inputSize": len(text),
        "outputSize": len(sanitized),
        "totalTime": total_time_ms,
        "throughput": len(text) / total_time_ms if total_time_ms > 0 else 0
    }

    benchmark_results["summary"][name] = result

    # For attack vectors, check if they were neutralized
    if ('attack' in name or 'injection' in name or
            'zero_width' in name or 'bidi' in name or
            'control' in name or 'uri' in name or
            'css' in name or 'template' in name or
            'homograph' in name):
        attack_result = analyze_attack_effect(name, text, sanitized)
        benchmark_results["attackResults"].append(attack_result)

        print(f"Attack neutralized: {'YES' if attack_result['neutralized'] else 'NO'}")
        print(f"Attack details: {', '.join(attack_result['details'])}")

    print(f"Total time: {result['totalTime']:.3f}ms")
    print(f"Throughput: {result['throughput']:.2f} characters/ms")
    print(f"Characters removed: {len(text) - len(sanitized)}")

    return result


async def generate_report() -> None:
    """Generate a benchmark report"""
    output_dir = Path('benchmark_results')

    # Save JSON results
    async with aiofiles.open(output_dir / 'benchmark_report.json', 'w', encoding='utf-8') as f:
        await f.write(json.dumps(benchmark_results, indent=2))

    # Generate HTML report
    html_report = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Python Text Sanitizer Benchmark Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
    h1, h2, h3 {{ color: #2c3e50; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; }}
    tr:nth-child(even) {{ background-color: #f9f9f9; }}
    .success {{ color: green; font-weight: bold; }}
    .failure {{ color: red; font-weight: bold; }}
    .chart {{ height: 400px; margin: 20px 0; }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <div class="container">
    <h1>Python Text Sanitizer Benchmark Report</h1>
    <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>

    <h2>Performance Summary</h2>
    <table>
      <tr>
        <th>Test Case</th>
        <th>Input Size</th>
        <th>Output Size</th>
        <th>Total Time (ms)</th>
        <th>Throughput (chars/ms)</th>
      </tr>
      {"".join([f"""
          <tr>
            <td>{name}</td>
            <td>{result["inputSize"]:,}</td>
            <td>{result["outputSize"]:,}</td>
            <td>{result["totalTime"]:.3f}</td>
            <td>{result["throughput"]:.2f}</td>
          </tr>
        """ for name, result in benchmark_results["summary"].items()])}
    </table>

    <div class="chart">
      <canvas id="performanceChart"></canvas>
    </div>

    <h2>Step-by-Step Timing Analysis</h2>
    <table>
      <tr>
        <th>Step</th>
        <th>Duration (ms)</th>
        <th>Chars/ms</th>
      </tr>
      {"".join([f"""
          <tr>
            <td>{timing["step"]}</td>
            <td>{timing["duration"]:.3f}</td>
            <td>{timing["charsPerMs"]:.2f}</td>
          </tr>
        """ for timing in benchmark_results["timings"]])}
    </table>

    <h2>Attack Vector Analysis</h2>
    <table>
      <tr>
        <th>Attack Type</th>
        <th>Neutralized</th>
        <th>Details</th>
      </tr>
      {"".join([f"""
        <tr>
          <td>{result["attackType"]}</td>
          <td class="{'success' if result["neutralized"] else 'failure'}">
            {'YES' if result["neutralized"] else 'NO'}
          </td>
          <td>{', '.join(result["details"])}</td>
        </tr>
      """ for result in benchmark_results["attackResults"]])}
    </table>

    <script>
      // Create performance chart
      const ctx = document.getElementById('performanceChart').getContext('2d');
      const perfData = {json.dumps([
        {"name": name, "inputSize": result["inputSize"], "throughput": float(f"{result['throughput']:.2f}")}
        for name, result in benchmark_results["summary"].items()
        if '10mb' not in name  # Exclude very large test for better chart scaling
    ])};

      new Chart(ctx, {{
        type: 'bar',
        data: {{
          labels: perfData.map(d => d.name),
          datasets: [{{
            label: 'Throughput (chars/ms)',
            data: perfData.map(d => d.throughput),
            backgroundColor: 'rgba(54, 162, 235, 0.5)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1
          }}]
        }},
        options: {{
          scales: {{
            y: {{
              beginAtZero: true,
              title: {{
                display: true,
                text: 'Throughput (chars/ms)'
              }}
            }},
            x: {{
              title: {{
                display: true,
                text: 'Test Case'
              }}
            }}
          }},
          plugins: {{
            title: {{
              display: true,
              text: 'Sanitizer Performance by Test Case'
            }}
          }}
        }}
      }});
    </script>
  </div>
</body>
</html>"""

    async with aiofiles.open(output_dir / 'benchmark_report.html', 'w', encoding='utf-8') as f:
        await f.write(html_report)


async def run_benchmark_suite() -> None:
    """Run the complete benchmark suite"""
    print('Starting Python Text Sanitizer Benchmark')
    print('========================================')

    try:
        # Generate test data
        test_data = await generate_test_data()

        # Run benchmarks for each test case
        print('\nBenchmarking normal text samples:')
        await run_benchmark('normal_100b', test_data['normal_100b'])
        await run_benchmark('normal_1kb', test_data['normal_1kb'])
        await run_benchmark('normal_10kb', test_data['normal_10kb'])
        await run_benchmark('normal_100kb', test_data['normal_100kb'])
        await run_benchmark('normal_1mb', test_data['normal_1mb'])

        # Only run the 10MB test if explicitly requested (it can take a while)
        import sys
        if '--run-10mb' in sys.argv:
            await run_benchmark('normal_10mb', test_data['normal_10mb'])
        else:
            print('\nSkipping 10MB test. Use --run-10mb flag to run it.')

        print('\nBenchmarking attack vectors:')
        await run_benchmark('xss_attack', test_data['xss_attack'])
        await run_benchmark('zero_width', test_data['zero_width'])
        await run_benchmark('bidi_attack', test_data['bidi_attack'])
        await run_benchmark('emoji_attack', test_data['emoji_attack'])
        await run_benchmark('control_chars', test_data['control_chars'])
        await run_benchmark('sql_injection', test_data['sql_injection'])
        await run_benchmark('mixed_attack', test_data['mixed_attack'])
        await run_benchmark('uri_attack', test_data['uri_attack'])
        await run_benchmark('css_attack', test_data['css_attack'])
        await run_benchmark('template_attack', test_data['template_attack'])
        await run_benchmark('homograph_attack', test_data['homograph_attack'])

        # Create a comprehensive benchmark report
        await generate_report()

        print('\nBenchmark completed!')
        print('Results saved to benchmark_results/benchmark_report.json')
        print('Detailed report saved to benchmark_results/benchmark_report.html')
    except Exception as error:
        print(f'Benchmark error: {error}')


if __name__ == '__main__':
    asyncio.run(run_benchmark_suite())