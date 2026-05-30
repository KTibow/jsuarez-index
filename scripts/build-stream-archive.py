#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["requests>=2.32"]
# ///
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import time

import requests


DEFAULT_MODEL = 'deepseek-v4-flash'
DEFAULT_ENDPOINT = 'https://crof.ai/v1'
SYSTEM_PROMPT_PATH = Path(__file__).with_name('build-stream-archive-system-prompt.txt')
TRANSCRIPTS_DIR = Path('src/data/transcripts')
CATALOG_DIR = Path('src/data/catalog')
LEGACY_STREAMS_DIR = Path('streams')
MAX_WORKERS = int(os.environ.get('ARCHIVE_WORKERS', '3'))
MAX_TRANSCRIPT_CHARS = int(os.environ.get('ARCHIVE_MAX_TRANSCRIPT_CHARS', '180000'))

TIMING_RE = re.compile(r'^\d\d:\d\d:\d\d\.\d{3} --> \d\d:\d\d:\d\d\.\d{3}')
GENERIC_TITLE_RE = re.compile(
    r'^(fixing reinforcement learning with pufferlib - live dev|'
    r'live dev - pufferlib is fixing reinforcement learning|'
    r'reinforcement learning dev on pufferlib|'
    r'reinforcement learning dev with joseph suarez|'
    r'reinforcement learning research live|'
    r'live reinforcement learning dev|'
    r'live rl dev on pufferlib(?: with joseph suarez)?|'
    r'pufferlib - live reinforcement learning dev|'
    r'pufferlib rl dev with joseph suarez)$',
    re.IGNORECASE,
)
PUFFER_VERSION_RE = re.compile(r'\bPuffer(?:Lib)?\s+v?\s*(\d+(?:\.\d+)?)\b', re.IGNORECASE)


def run_git(args: list[str]) -> str:
    return subprocess.check_output(['git', *args], text=True)


def transcript_paths() -> list[str]:
    paths = sorted(str(path) for path in TRANSCRIPTS_DIR.glob('*.vtt'))
    if paths:
        return paths
    paths = sorted(str(path) for path in LEGACY_STREAMS_DIR.glob('*.vtt'))
    if paths:
        return paths
    try:
        paths = run_git(['ls-tree', '-r', '--name-only', 'HEAD', str(TRANSCRIPTS_DIR)]).splitlines()
        if paths:
            return paths
        return run_git(['ls-tree', '-r', '--name-only', 'HEAD', str(LEGACY_STREAMS_DIR)]).splitlines()
    except subprocess.CalledProcessError:
        return []


def read_stream(path: str) -> str:
    file_path = Path(path)
    if file_path.exists():
        return file_path.read_text(encoding='utf-8')
    return run_git(['show', f'HEAD:{path}'])


def parse_metadata(text: str, fallback_id: str) -> dict[str, str]:
    metadata = {'id': fallback_id, 'url': f'https://www.youtube.com/watch?v={fallback_id}'}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line != 'metadata:':
            continue
        for metadata_line in lines[index + 1:]:
            if not metadata_line.strip():
                break
            key, separator, value = metadata_line.partition(':')
            if separator:
                metadata[key.strip()] = value.strip()
        break
    return metadata


def transcript_body(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or TIMING_RE.match(stripped):
            continue
        if stripped in {'WEBVTT', 'NOTE', 'metadata:'}:
            continue
        if re.match(r'^(title|id|url|date|language): ', stripped):
            continue
        lines.append(stripped)
    return '\n'.join(lines)


def catalog_path(video_id: str) -> Path:
    return CATALOG_DIR / f'{video_id}.json'


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def fallback_archive(metadata: dict[str, str]) -> dict:
    title = metadata.get('title') or metadata['id']
    return {
        'title': title,
        'summary': 'Transcript indexed. Run with CROF_KEY to generate a specific title, summary, and topics.',
        'topics': heuristic_topics(title),
    }


def heuristic_topics(title: str) -> list[str]:
    lowered = title.lower()
    topics = []
    checks = [
        ('PufferLib', 'puffer'),
        ('CUDA', 'cuda'),
        ('C', ' c'),
        ('Cython', 'cython'),
        ('WASM', 'wasm'),
        ('MOBA', 'moba'),
        ('GPUDrive', 'gpudrive'),
        ('World Models', 'world model'),
        ('Hyperparameter Tuning', 'hyperparam'),
        ('Imitation Learning', 'imitation'),
        ('Off-policy RL', 'off-policy'),
    ]
    for label, needle in checks:
        if needle in lowered and label not in topics:
            topics.append(label)
    return topics[:5] or ['Stream']


def parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def normalize_puffer_versions(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        version = match.group(1)
        if version.endswith('.0'):
            version = version[:-2]
        return f'Puffer {version}'

    return PUFFER_VERSION_RE.sub(replace, text)


def call_llm(metadata: dict[str, str], body: str) -> dict:
    api_key = os.environ.get('CROF_KEY')
    if not api_key:
        return fallback_archive(metadata)

    endpoint = DEFAULT_ENDPOINT
    model = DEFAULT_MODEL
    original_title = metadata.get('title', metadata['id'])
    transcript = body[:MAX_TRANSCRIPT_CHARS]
    generic_note = 'The original title is generic; ignore it.' if GENERIC_TITLE_RE.match(original_title) else ''

    payload = {
        'model': model,
        'temperature': 0.2,
        'response_format': {'type': 'json_object'},
        'messages': [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT_PATH.read_text(encoding='utf-8'),
            },
            {
                'role': 'user',
                'content': (
                    f'Original title: {original_title}\n'
                    f'Date: {metadata.get("date", "unknown")}\n'
                    f'{generic_note}\n\n'
                    f'Transcript:\n{transcript}'
                ),
            },
        ],
    }

    last_error = None
    for attempt in range(4):
        try:
            response = requests.post(
                f'{endpoint}/chat/completions',
                headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            data = response.json()
            content = data['choices'][0]['message']['content']
            parsed = parse_json_object(content)
            return normalize_llm_archive(parsed, metadata)
        except (requests.RequestException, KeyError, json.JSONDecodeError, TypeError, ValueError) as error:
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f'LLM request failed for {metadata["id"]}: {last_error}')


def normalize_llm_archive(data: dict, metadata: dict[str, str]) -> dict:
    title = normalize_puffer_versions(str(data.get('title') or metadata.get('title') or metadata['id']).strip())
    summary = normalize_puffer_versions(str(data.get('summary') or '').strip())
    topics = data.get('topics') or []
    if not isinstance(topics, list):
        topics = []
    cleaned_topics = []
    for topic in topics:
        topic = normalize_puffer_versions(str(topic).strip())
        if topic and topic.lower() not in {item.lower() for item in cleaned_topics}:
            cleaned_topics.append(topic)
    return {
        'title': title[:140],
        'summary': summary or fallback_archive(metadata)['summary'],
        'topics': cleaned_topics[:6] or fallback_archive(metadata)['topics'],
    }


def materialize_transcript(video_id: str, text: str) -> None:
    target = TRANSCRIPTS_DIR / f'{video_id}.vtt'
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')


def build_entry(path: str) -> tuple[dict, bool]:
    video_id = Path(path).stem.removesuffix('.en')
    text = read_stream(path)
    metadata = parse_metadata(text, video_id)
    materialize_transcript(metadata['id'], text)

    path = catalog_path(metadata['id'])
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8')), False

    body = transcript_body(text)
    archive = call_llm(metadata, body)
    entry = {
        'id': metadata['id'],
        'date': metadata.get('date', ''),
        'url': metadata.get('url') or f'https://www.youtube.com/watch?v={metadata["id"]}',
        'transcript': f'/src/data/transcripts/{metadata["id"]}.vtt',
        'originalTitle': metadata.get('title', metadata['id']),
        **archive,
    }
    write_json(path, entry)
    return entry, True


def main() -> None:
    paths = transcript_paths()
    if not paths:
        raise SystemExit(f'No stream caption files were found in {TRANSCRIPTS_DIR}/ or git HEAD')

    entries = []
    failures = []
    written = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(build_entry, path): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                entry, did_write = future.result()
                entries.append(entry)
                if did_write:
                    written += 1
            except Exception as error:
                failures.append(f'{path}: {error}')

    if failures:
        raise SystemExit('Failed to build archive entries:\n' + '\n'.join(failures))

    print(f'Indexed {len(entries)} streams; wrote {written} new catalog file(s) to {CATALOG_DIR}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
