"""
Optimized Ultimate Text Sanitizer (Python)
Ensures only truly sanitized text is returned, with any surprises left behind
"""

import re
import unicodedata
from typing import Dict, Optional, Union
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from html import escape
import bleach

app = Flask(__name__)

# Configure rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per 15 minutes"]
)

# Pre-compiled regex patterns (module scope for better performance)
UNSAFE_CHARS_REGEX = re.compile(
    r'[\x00-\x1F\x7F-\x9F\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF\u061C\u200E\u200F\u2066-\u2069]')
WHITESPACE_REGEX = re.compile(r'\s+')
SAFE_CHAR_PATTERN = re.compile(r'^[A-Za-z0-9.,;:\'"!?()\[\]{}\s@#$%^&*_+=|\\\/~`<>-]$')

# Common character replacement mappings
ACCENT_MAPPINGS = {
    'á': 'a', 'à': 'a', 'â': 'a', 'ä': 'a', 'ã': 'a', 'å': 'a',
    'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
    'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
    'ó': 'o', 'ò': 'o', 'ô': 'o', 'ö': 'o', 'õ': 'o',
    'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
    'ý': 'y', 'ÿ': 'y',
    'ñ': 'n', 'ç': 'c'
}


def sanitize_base(text: str) -> str:
    """
    Core sanitization module - removes control characters and normalizes whitespace

    Args:
        text: Input text to sanitize

    Returns:
        Text with control characters and excessive whitespace removed
    """
    if not isinstance(text, str):
        return ''

    # Trim text
    result = text.strip()

    # Remove control characters, zero-width chars, and bidirectional controls in a single pass
    result = UNSAFE_CHARS_REGEX.sub('', result)

    # Normalize whitespace (replace multiple spaces, tabs, etc. with single space)
    result = WHITESPACE_REGEX.sub(' ', result)

    # Unicode normalization to handle emoji and special characters
    result = unicodedata.normalize('NFKC', result)

    return result


def sanitize_html(text: str) -> str:
    """
    HTML sanitization module - removes HTML/script tags

    Args:
        text: Input text to sanitize

    Returns:
        Text with HTML content removed
    """
    # First escape all HTML entities
    escaped = escape(text)

    # Then use bleach to remove any remaining tags
    # Bleach is a Python library specifically designed for sanitizing HTML
    return bleach.clean(
        escaped,
        tags=[],  # No tags allowed
        strip=True,
        strip_comments=True
    )


def sanitize_sql(text: str) -> str:
    """
    SQL sanitization module - escapes SQL patterns
    Note: This is a basic defense. Use parameterized queries in production!

    Args:
        text: Input text to sanitize

    Returns:
        Text with SQL patterns escaped
    """
    # Basic SQL pattern escaping
    result = text
    result = result.replace("'", "''")
    result = result.replace("%", "\\%")
    result = result.replace("_", "\\_")
    return result


def is_control_char(code: int) -> bool:
    """
    Check if a character code is a control character

    Args:
        code: Character code

    Returns:
        True if it's a control character
    """
    return (code <= 31) or (127 <= code <= 159)


def retype_text(text: str) -> str:
    """
    Character retyping module - rebuilds string with only safe characters

    Args:
        text: Input text to sanitize

    Returns:
        Safely retyped text
    """
    buffer = []

    for char in text:
        # Add the character if it's safe
        if SAFE_CHAR_PATTERN.match(char):
            buffer.append(char)
        else:
            # For non-ASCII but printable characters
            code = ord(char)
            if code > 127 and code < 65536 and not is_control_char(code):
                # Try to replace with ASCII equivalent
                replacement = ACCENT_MAPPINGS.get(char, char)
                buffer.append(replacement)
            # Skip other characters

    return ''.join(buffer)


def ultimate_sanitizer(text: str, max_length: int = 5000) -> str:
    """
    Main sanitizer function - combines all sanitization steps

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length

    Returns:
        Thoroughly sanitized text
    """
    if not isinstance(text, str):
        return ''

    # Limit length first (quick operation)
    sanitized = text[:max_length]

    # Apply the sanitization pipeline
    sanitized = sanitize_base(sanitized)
    sanitized = sanitize_html(sanitized)
    sanitized = sanitize_sql(sanitized)

    # Final barrier - retype to ensure only safe characters remain
    sanitized = retype_text(sanitized)

    return sanitized


@app.route('/api/sanitize', methods=['POST'])
@limiter.limit("100 per 15 minutes")
def sanitize_endpoint():
    """
    POST endpoint to sanitize text
    Route: /api/sanitize
    """
    # Validate request
    if not request.is_json:
        return jsonify({
            'success': False,
            'message': 'Request must be JSON'
        }), 400

    data = request.get_json()

    if 'text' not in data:
        return jsonify({
            'success': False,
            'message': 'Text input is required'
        }), 400

    try:
        text = data['text']
        max_length = data.get('maxLength', 5000)  # Default max length is 5000

        # Validate max_length
        if not isinstance(max_length, int) or not (1 <= max_length <= 10000):
            return jsonify({
                'success': False,
                'message': 'Max length must be between 1 and 10000 characters'
            }), 400

        # Check size before processing
        approximate_byte_size = len(text.encode('utf-8'))
        if approximate_byte_size > 100 * 1024:  # 100KB limit
            return jsonify({
                'success': False,
                'message': 'Input exceeds maximum allowed size (100KB)'
            }), 413

        # Sanitize the text
        sanitized = ultimate_sanitizer(text, max_length)

        # Return the sanitized text
        return jsonify({
            'success': True,
            'originalLength': len(text),
            'sanitizedLength': len(sanitized),
            'sanitizedText': sanitized
        }), 200
    except Exception as error:
        print(f'Sanitization error: {error}')
        return jsonify({
            'success': False,
            'message': 'Server error during text sanitization'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    app.run(port=3000, debug=False)