#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from html import unescape
from pathlib import Path
import json
import os
import re


TIMING_RE = re.compile(r'^(\d\d:\d\d:\d\d\.\d{3}) --> (\d\d:\d\d:\d\d\.\d{3})(?: .*)?$')
INLINE_TIMESTAMP_RE = re.compile(r'<\d\d:\d\d:\d\d\.\d{3}>')
TAG_RE = re.compile(r'<[^>]+>')
CAPTION_RE = re.compile(r'^([A-Za-z0-9_-]{11})(?:\.en)?\.vtt$')


def clean_caption_text(text: str) -> str:
    text = INLINE_TIMESTAMP_RE.sub('', text)
    text = TAG_RE.sub('', text)
    text = unescape(text)
    return ' '.join(text.split())


def clean_vtt_body(text: str) -> str:
    lines = text.splitlines()
    cues: list[tuple[str, str, str]] = []
    index = 0

    while index < len(lines):
        match = TIMING_RE.match(lines[index])
        if not match:
            index += 1
            continue

        start, end = match.groups()
        index += 1
        cue_lines = []
        while index < len(lines) and lines[index].strip():
            cue_lines.append(lines[index])
            index += 1

        cue_text = clean_caption_text(' '.join(cue_lines))
        if not cue_text:
            continue

        if cues:
            previous_text = cues[-1][2]
            if cue_text == previous_text:
                continue
            if cue_text.startswith(previous_text + ' '):
                cue_text = cue_text[len(previous_text):].strip()
                if not cue_text:
                    continue

        cues.append((start, end, cue_text))

    body = []
    for start, end, cue_text in cues:
        body.append(f'{start} --> {end}')
        body.append(cue_text)
        body.append('')
    return '\n'.join(body)


def load_metadata(metadata_file: Path) -> dict[str, dict]:
    metadata_by_id = {}
    if not metadata_file.exists():
        return metadata_by_id

    for line in metadata_file.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        metadata = json.loads(line)
        video_id = metadata.get('id')
        if video_id:
            metadata_by_id[video_id] = metadata
    return metadata_by_id


def existing_metadata(text: str) -> dict[str, str]:
    metadata = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line == 'metadata:':
            for metadata_line in lines[index + 1:]:
                if not metadata_line.strip():
                    break
                key, separator, value = metadata_line.partition(':')
                if separator:
                    metadata[key.strip()] = value.strip()
            break
    return metadata


def metadata_date(metadata: dict) -> str | None:
    if metadata.get('date'):
        return str(metadata['date'])
    timestamp = metadata.get('release_timestamp') or metadata.get('timestamp')
    if timestamp is None and metadata.get('upload_date'):
        timestamp = datetime.strptime(metadata['upload_date'], '%Y%m%d').replace(tzinfo=timezone.utc).timestamp()
    if timestamp is None:
        return None
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()


def normalize() -> int:
    streams_dir = Path('streams')
    metadata_file = Path(os.environ.get('RUNNER_TEMP') or '/tmp/jsuarez-index-sync') / 'stream-info.jsonl'
    caption_files = sorted(streams_dir.glob('*.vtt'))
    if not caption_files:
        raise SystemExit('No stream caption files were found')

    metadata_by_id = load_metadata(metadata_file)
    empty = []
    normalized = 0

    for caption_file in caption_files:
        id_match = CAPTION_RE.match(caption_file.name)
        if not id_match:
            continue
        video_id = id_match.group(1)

        if caption_file.stat().st_size == 0:
            empty.append(str(caption_file))
            continue

        text = caption_file.read_text(encoding='utf-8')
        if not text.startswith('WEBVTT'):
            empty.append(str(caption_file))
            continue

        existing = existing_metadata(text)
        metadata = metadata_by_id.get(video_id, {})
        title = metadata.get('title') or existing.get('title') or caption_file.stem
        webpage_url = metadata.get('webpage_url') or existing.get('url') or f'https://www.youtube.com/watch?v={video_id}'

        header = [
            'WEBVTT',
            '',
            'NOTE',
            'metadata:',
            f'title: {title}',
            f'id: {metadata.get("id") or existing.get("id") or video_id}',
            f'url: {webpage_url}',
        ]
        date = metadata_date(metadata) or metadata_date(existing)
        if date:
            header.append(f'date: {date}')
        header.extend(['language: en', ''])

        target_file = streams_dir / f'{video_id}.vtt'
        body = clean_vtt_body(text)
        if not body:
            empty.append(str(caption_file))
            continue
        target_file.write_text('\n'.join(header) + '\n' + body, encoding='utf-8')
        if target_file != caption_file:
            caption_file.unlink()
        normalized += 1

    if empty:
        raise SystemExit('Invalid or empty caption files:\n' + '\n'.join(empty))
    return normalized


def main() -> None:
    normalized = normalize()
    print(f'Normalized {normalized} stream caption file(s)')


if __name__ == '__main__':
    main()
