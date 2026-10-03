"""Portable UFC data and bounded automatic quote collection, with atomic caches."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import io
import json
import os
from pathlib import Path
import tempfile

import pandas as pd
import requests

from src.app.odds_api import _request, resolve_api_key
from src.app.tennis_strategy import utc
from src.app import ufc_markov as markov
from src.app.value_methods import BOOKMAKERS, FOOTBALL_SPORTS, MMA_KEY

DEFAULT_LEAGUES = ['soccer_france_ligue_one', 'soccer_epl', 'soccer_germany_bundesliga',
                   'soccer_italy_serie_a', 'soccer_spain_la_liga']
DAILY_REQUEST_CAP = 12
QUOTA_RESERVE = 20
AUTO_SCAN_SECONDS = 3600


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(value, file, ensure_ascii=False, allow_nan=False)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def load_ufc(root):
    document = read_json(Path(root) / 'models/value_methods/ufc.json')
    if not isinstance(document, dict):
        raise ValueError('Paquet UFC indisponible : actualiser les données.')
    if markov.fingerprint({k: v for k, v in document.items() if k != 'sha256'}) != document.get('sha256'):
        raise ValueError('Paquet UFC incohérent : actualiser les données.')
    markov.verify_bundle(document['bundle'])
    return document['bundle'], document['cards']


def _historical_seed(root):
    folder = Path(root) / 'models/value_methods'
    portable = folder / 'ufc_history.csv.gz'
    if portable.exists():
        return pd.read_csv(portable, dtype={'fighter_1_id': str, 'fighter_2_id': str, 'fight_id': str}, parse_dates=['event_date'])
    local = Path(root) / 'predictor_ufc/data/rigorous/processed/fights.parquet'
    if local.exists():
        return pd.read_parquet(local)
    from predictor_ufc.rigorous.data_pipeline import UFCSTATS_COMPETITIONS_URL, canonicalise_source_fights
    response = requests.get(UFCSTATS_COMPETITIONS_URL, timeout=40)
    response.raise_for_status()
    return canonicalise_source_fights(pd.read_csv(io.BytesIO(response.content)))


def refresh_ufc(root, now=None, progress=lambda message: None):
    """Free official data only. No odds requests, hyperparameter search or wagers."""
    from predictor_ufc.rigorous.data_pipeline import (UFCStatsClient, get_completed_events,
        _parse_event_index, _parse_fight_detail, canonicalise_supplemental_fights)
    from predictor_ufc.rigorous.upcoming import fetch_upcoming_events, fetch_card
    now = utc(now)
    folder = Path(root) / 'models/value_methods'
    with lock(folder / '.refresh.lock'):
        history = _historical_seed(root)
        history['event_date'] = pd.to_datetime(history.event_date).dt.tz_localize(None)
        cutoff = now.normalize().tz_localize(None)
        history = history[history.event_date.lt(cutoff)].copy()
        if history.empty:
            raise ValueError('Historique UFC vide.')
        client = UFCStatsClient()
        completed = get_completed_events(client)
        official_completed = sorted([e for e in completed if e.date < cutoff], key=lambda e: e.date)
        if not official_completed:
            raise ValueError('Aucune carte UFC terminée vérifiable avant aujourd’hui.')
        latest = max(history.event_date)
        missing = [e for e in official_completed if e.date > latest]
        if len(missing) > 8:
            raise ValueError('Plus de huit cartes manquantes : actualisation complète des données UFC requise.')
        extra = []
        for event in missing:
            progress(f'UFCStats : {event.name} ({event.date.date()})')
            _, fights = _parse_event_index(client, event)
            if not fights:
                raise ValueError('Carte terminée sans statistiques lisibles : actualisation interrompue.')
            extra.extend(_parse_fight_detail(client, fight) for fight in fights)
        if extra:
            history = pd.concat([history, canonicalise_supplemental_fights(extra)], ignore_index=True)
        # Direct and historical extracts use different fight IDs for the same bout.
        history['_identity'] = [f"{pd.Timestamp(d).date()}|{'|'.join(sorted([str(a), str(b)]))}"
                               for d, a, b in zip(history.event_date, history.fighter_1_id, history.fighter_2_id)]
        history = history.drop_duplicates('_identity', keep='last').drop(columns='_identity')
        if history.event_date.max() < official_completed[-1].date:
            raise ValueError('La dernière carte UFC terminée manque à l’historique.')
        official = []
        for event_name, date, url in fetch_upcoming_events(client, limit=3):
            fights = fetch_card(client, event_name, date, url)
            for fight in fights:
                official.append({**asdict(fight), 'event_date': date.date().isoformat(), 'source_url': url})
        checked = utc().isoformat()
        bundle = markov.build_bundle(history, now, checked)
        # Save only the portable columns consumed by this implementation.
        columns = ['fight_id', 'event_date', 'weight_class', 'method', 'duration_secs', 'y',
                   'fighter_1', 'fighter_2', 'fighter_1_id', 'fighter_2_id']
        columns += [f'p{s}_{k}' for s in (1, 2) for k in ['sig_lnd', 'sig_att', 'td_lnd', 'td_att', 'sub_att', 'rev', 'ctrl_secs']]
        portable = history[columns].sort_values(['event_date', 'fight_id'])
        folder.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(suffix='.csv.gz', dir=folder)
        os.close(handle)
        try:
            portable.to_csv(temporary, index=False, compression={'method': 'gzip', 'mtime': 0})
            import hashlib
            history_hash = hashlib.sha256(Path(temporary).read_bytes()).hexdigest()
            os.replace(temporary, folder / 'ufc_history.csv.gz')
        finally:
            Path(temporary).unlink(missing_ok=True)
        document = {'bundle': bundle, 'cards': {'checked_at': checked, 'fights': official},
                    'history_sha256': history_hash, 'official_last_completed_date': official_completed[-1].date.date().isoformat()}
        document['sha256'] = markov.fingerprint(document)
        atomic_json(folder / 'ufc.json', document)
        progress(f"Modèle Markov : {bundle['history_fights_used']} combats, {bundle['decision_training_count']} décisions chronologiques.")
        return document


def refresh_due(root, now=None):
    try:
        bundle, _ = load_ufc(root)
        age = utc(now) - utc(bundle['checked_at'])
        return not pd.Timedelta(0) <= age <= pd.Timedelta(hours=24)
    except (ValueError, KeyError, TypeError):
        return True


def collect_live(root, sport, leagues=None, now=None, force=False):
    """One shared cache and budget for all users; never logs provider secrets."""
    now = utc(now)
    if sport not in {'football', 'ufc'}:
        raise ValueError('Sport inconnu.')
    keys = list(dict.fromkeys(leagues if leagues is not None else DEFAULT_LEAGUES)) if sport == 'football' else [MMA_KEY]
    if not keys or (sport == 'football' and any(k not in FOOTBALL_SPORTS for k in keys)):
        raise ValueError('Compétitions football invalides.')
    folder = Path(root) / 'bets/value_methods_runtime'
    cache_path = folder / f'{sport}.json'
    with lock(folder / '.quotes.lock'):
        previous = read_json(cache_path)
        if not force and previous and previous.get('sports') == keys and pd.Timedelta(0) <= now - utc(previous['at']) < pd.Timedelta(seconds=AUTO_SCAN_SECONDS):
            return previous
        snapshot = {'at': now.isoformat(), 'sport': sport, 'sports': keys, 'events': [], 'errors': [],
                    'coverage': [], 'remaining': None, 'requests': 0, 'no_promotions': True}
        budget = read_json(folder / 'budget.json', {})
        day = now.tz_convert('Europe/Paris').date().isoformat()
        if budget.get('day') != day:
            budget = {'day': day, 'used': 0}
        snapshot['daily_used'] = budget['used']
        if budget['used'] >= DAILY_REQUEST_CAP:
            snapshot['errors'].append('Plafond quotidien partagé atteint (12 consultations). Reprise demain.')
            atomic_json(cache_path, snapshot)
            return snapshot
        try:
            key, _ = resolve_api_key(Path(root))
            if not key:
                snapshot['errors'].append('Clé de cotes absente. Configurer ODDS_API_KEY dans les secrets de l’application.')
                atomic_json(cache_path, snapshot)
                return snapshot
            catalogue = _request('sports/', key, {})  # Zero-credit endpoint.
            snapshot['remaining'] = catalogue.remaining
            if not catalogue.ok:
                snapshot['errors'].append(catalogue.error)
            else:
                active = {s['key'] for s in catalogue.events if s.get('active')}
                for sport_key in keys:
                    if sport_key not in active:
                        snapshot['coverage'].append({'competition': sport_key, 'status': 'inactive', 'events': 0})
                        continue
                    remaining = snapshot['remaining']
                    if remaining is None or remaining <= QUOTA_RESERVE or budget['used'] >= DAILY_REQUEST_CAP:
                        snapshot['errors'].append('Collecte limitée : réserve mensuelle de 20 crédits, quota inconnu ou plafond quotidien atteint.')
                        break
                    # Reserve the credit before the request; a failed request never refunds the local budget.
                    budget['used'] += 1
                    atomic_json(folder / 'budget.json', budget)
                    response = _request(f'sports/{sport_key}/odds', key,
                        {'bookmakers': ','.join([*BOOKMAKERS, 'pinnacle']), 'markets': 'h2h', 'oddsFormat': 'decimal'})
                    snapshot['requests'] += 1
                    snapshot['remaining'] = response.remaining  # Missing headers fail closed on subsequent calls.
                    if response.ok:
                        accepted = [e for e in response.events if isinstance(e, dict) and e.get('sport_key') == sport_key]
                        snapshot['events'].extend(accepted)
                        snapshot['coverage'].append({'competition': sport_key, 'status': 'ok', 'events': len(accepted)})
                    else:
                        snapshot['errors'].append(f'{FOOTBALL_SPORTS.get(sport_key, "UFC")} : {response.error}')
                        snapshot['coverage'].append({'competition': sport_key, 'status': 'error', 'events': 0})
                        if '401' in response.error or '429' in response.error:
                            break
        except Exception as error:
            snapshot['errors'].append(f'Collecte interrompue ({type(error).__name__}).')
        snapshot['daily_used'] = budget['used']
        atomic_json(cache_path, snapshot)
        return snapshot
