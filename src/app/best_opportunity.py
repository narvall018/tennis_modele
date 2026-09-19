"""La même question posée à tout le dépôt : quelle est la meilleure opportunité, tous sports confondus.

Chaque stratégie de ce dépôt répond pour son sport et avec sa propre unité. Les
comparer exige de fixer d'abord ce qui l'emporte, avant de voir les nombres :

* **Palier 1, arithmétique.** Un arbitrage entre bookmakers ne demande aucune
  prévision : si les prix des deux issues se contredisent assez, la mise répartie
  rend un profit quel que soit le résultat. Sous ses garde-fous (fraîcheur,
  absence d'exchange, comptes réellement ouvrables), c'est un gain calculé, pas
  estimé.
* **Palier 2, modèles.** Les sélections ATP et WTA sont des espérances *estimées*
  par des règles figées dont le filtre de validation a échoué. Leur meilleur
  nombre reste une opinion sur des prix, pas un gain.

Un palier 1 exploitable passe donc toujours devant un palier 2, même si le
pourcentage affiché est plus petit : 1,5 % garanti n'est pas comparable à 8 %
espérés par un modèle non validé. La règle est écrite ici, en amont du scan, pour
qu'aucun classement ne soit choisi après coup.

Deux sports n'ont aucun droit de sélection : le football et l'UFC. Leurs propres
métadonnées disent qu'ils ne battent pas le marché — le module les lit plutôt que
de les paraphraser. Ils restent couverts par le palier 1, qui ne demande rien à un
modèle.

Rien ici n'appelle le réseau ni ne mise : les évènements arrivent déjà relevés, et
le résultat est un classement à lire.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from src.app import atp_reference_strategy as atp
from src.app import wta_kernel_strategy as wta
from src.backtesting.arbitrage import classify, scan

ARBITRAGE_TIER = 1
MODEL_TIER = 2
SURFACES = ('Hard', 'Clay', 'Grass')
UNIT_FRACTION = .0025      # Mise unitaire des carnets : 0,25 % de la bankroll du jour.
DAILY_CAP_FRACTION = .02   # Plafond quotidien des carnets, non réinvesti.


def _start(value):
    try:
        return atp.utc(value).isoformat()
    except (ValueError, TypeError):
        return ''


def arbitrage_rows(events_by_sport, now=None):
    """Palier 1 pour tous les sports fournis, garde-fous reportés et non masqués."""
    opportunities = []
    for sport, events in sorted(events_by_sport.items()):
        opportunities += scan(events, sport, pd.Timestamp(now) if now is not None else None)
    frame = classify(opportunities)
    rows = []
    by_key = {item.key(): item for item in opportunities}
    for record in (frame.to_dict('records') if not frame.empty else []):
        item = by_key.get(f"{record['sport']}|{record['événement']}")
        blocked = [label for label, ok in [('fraîcheur', record['assez_frais']),
                                           ('exchange', record['sans_exchange']),
                                           ('accessibilité', record['books_accessibles'])] if not ok]
        rows.append({
            'tier': ARBITRAGE_TIER, 'tier_label': 'arbitrage arithmétique',
            'strategy_id': 'arbitrage_cross_book_scan',
            'sport': record['sport'], 'competition': record['sport'],
            'event': record['événement'], 'start': _start(item.commence_time if item else ''),
            'metric': float(record['gain_garanti']),
            'metric_kind': 'gain_garanti_si_les_deux_jambes_passent',
            'pick': 'les deux issues', 'books': record['books'], 'odds': record['cotes'],
            'exploitable': bool(record['exploitable']),
            'blocked_reason': '' if record['exploitable'] else 'écarté : ' + ', '.join(blocked),
            'real_money_authorised': False,
            'detail': {'legs': item.legs if item else [], 'overround': float(record['overround']),
                       'worst_staleness_minutes': float(record['fraîcheur_pire_min']),
                       'seen_at_utc': record['vu_a']},
        })
    return rows


def atp_rows(events, now=None):
    """Palier 2, ATP : prix de référence Pinnacle contre un opérateur français."""
    rows = []
    for event in events:
        if not str(event.get('sport_key', '')).startswith('tennis_atp_'):
            continue
        result = atp.analyse_event(event, now)
        candidate = result.get('candidate')
        if not candidate:
            continue
        side = candidate['selected_side']
        rows.append({
            'tier': MODEL_TIER, 'tier_label': 'modèle non validé (papier)',
            'strategy_id': candidate['strategy_id'],
            'sport': event.get('sport_key', ''), 'competition': result['competition'],
            'event': result['match'], 'start': _start(candidate['fixture']['start']),
            'metric': float(candidate['expected_returns'][side]),
            'metric_kind': 'ev_estimee_apres_decote_modele_non_valide',
            'pick': candidate['pick'], 'books': candidate['fixture']['bookmaker'],
            'odds': f"{candidate['odds']:.2f}",
            'exploitable': True, 'blocked_reason': '',
            'real_money_authorised': False, 'detail': candidate,
        })
    return rows


def wta_rows(bundle, events, surfaces, now=None):
    """Palier 2, WTA : noyau expérimental, une surface confirmée par tournoi.

    `surfaces` associe un intitulé de compétition à sa surface. Fournir une entrée
    vaut confirmation du simple, tableau principal : le modèle refuse les
    qualifications, les doubles et le WTA 125, et rien ici ne devine une surface.
    Un tournoi absent de la table est signalé, jamais supposé.

    Un paquet périmé fait lever une erreur plutôt que rendre une liste vide : zéro
    sélection parce que le modèle est hors service ne se lit pas comme zéro
    sélection parce que le marché n'offre rien.
    """
    reasons = wta.freshness_reasons(bundle[0], now)
    if reasons:
        raise ValueError(' '.join(reasons))
    confirmed = {str(name).strip().casefold(): surface for name, surface in (surfaces or {}).items()}
    unknown, skipped, rows = set(), [], []
    for quote in wta.quotes(events, now):
        if any('/' in str(quote[side]) for side in ['player_1', 'player_2']):
            continue  # Un double n'est pas un simple ; le modèle n'en price aucun.
        surface = confirmed.get(quote['competition'].strip().casefold())
        if surface not in SURFACES:
            unknown.add(quote['competition'])
            continue
        fixture = {**quote, 'tour': 'WTA', 'surface': surface, 'singles_main_draw': True,
                   'tournament': quote['competition'], 'api_pair': dict(quote)}
        try:
            scored = wta.score_fixture(bundle, fixture, now)
        except (ValueError, KeyError, TypeError) as error:
            # Identité inconnue, homonymes ou statistiques insuffisantes : on dit
            # lesquelles plutôt que de laisser croire que le marché était vide.
            skipped.append({'event': f"{quote['player_1']} — {quote['player_2']}",
                            'bookmaker': quote['bookmaker'], 'raison': str(error)})
            continue
        if not scored['eligible']:
            continue
        side = scored['selected_side']
        rows.append({
            'tier': MODEL_TIER, 'tier_label': 'modèle non validé (papier)',
            'strategy_id': scored['strategy_id'],
            'sport': quote['sport_key'], 'competition': quote['competition'],
            'event': f"{quote['player_1']} — {quote['player_2']}", 'start': _start(quote['start']),
            'metric': float(scored['expected_returns'][side]),
            'metric_kind': 'ev_estimee_apres_decote_modele_non_valide',
            'pick': scored['pick'], 'books': quote['bookmaker'], 'odds': f"{scored['odds']:.2f}",
            'exploitable': True, 'blocked_reason': '',
            'real_money_authorised': False, 'detail': scored,
        })
    return rows, {'surfaces_a_confirmer': sorted(unknown), 'ecartes': skipped}


def models_without_selection_rights(root):
    """Ce que le football et l'UFC disent d'eux-mêmes, lu dans leurs métadonnées."""
    notes = {}
    for sport, path in [('football', 'models/football/metadata.json'), ('ufc', 'models/ufc/metadata.json')]:
        try:
            meta = json.loads((Path(root)/path).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            notes[sport] = 'Métadonnées illisibles : aucune sélection possible.'
            continue
        note = str(meta.get('honest_note', '')).strip()
        versus = meta.get('versus_market') or {}
        if 'model_minus_market' in versus:
            note += f" Écart de log-loss face au marché : {versus['model_minus_market']:+.6f}."
        notes[sport] = note or 'Aucune preuve de supériorité sur le marché dans le paquet livré.'
    return notes


def rank(rows):
    """Palier d'abord, mesure ensuite ; départage déterministe, jamais aléatoire."""
    usable = [row for row in rows if row.get('exploitable')]
    return sorted(usable, key=lambda row: (row['tier'], -row['metric'], row['start'],
                                           row['sport'], row['event'], row['books']))


def best(rows):
    ordered = rank(rows)
    return ordered[0] if ordered else None


def stake_plan(row, bankroll_cents):
    """Ce que les règles figées autorisent à engager, sans rien réinventer.

    Arbitrage : la répartition qui rend le même montant sur chaque issue, plafonnée
    au budget quotidien du carnet. Modèle : l'unité de 0,25 %, arrondie au centime
    inférieur comme dans les carnets. Aucun réinvestissement intrajournalier.
    """
    cap = math.floor(bankroll_cents*DAILY_CAP_FRACTION)
    if row['tier'] == ARBITRAGE_TIER:
        legs = row['detail']['legs']
        overround = row['detail']['overround']
        if not legs or overround <= 0:
            return {'total_cents': 0, 'legs': [], 'cap_cents': cap}
        shares = [math.floor(cap/leg['price']/overround) for leg in legs]
        return {'total_cents': sum(shares), 'cap_cents': cap,
                'legs': [{'outcome': leg['outcome'], 'bookmaker': leg['bookmaker'],
                          'odds': leg['price'], 'stake_cents': share}
                         for leg, share in zip(legs, shares)]}
    unit = min(math.floor(bankroll_cents*UNIT_FRACTION), cap)
    return {'total_cents': unit, 'cap_cents': cap,
            'legs': [{'outcome': row['pick'], 'bookmaker': row['books'],
                      'odds': float(row['odds']), 'stake_cents': unit}]}
