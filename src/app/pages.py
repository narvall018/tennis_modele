"""The three new app sections: predictions, staking, and measured performance.

Kept out of ``unified_app.py`` so the betting ledger and the research output stay
separable — the ledger is a personal record, this is what the studies concluded,
and mixing them would let one drift into looking like the other.

A design rule runs through all three: the honest number is the headline, not a
footnote. Eight studies found no demonstrable edge, so a page that buries that
under a green figure would be the one dishonest thing in this repository.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.app.evidence import all_evidence, global_verdict, model_registry
from src.app.github_ledger import (
    append_bet,
    ensure_branch,
    ledger_frame,
    load_config,
)
from src.app.ledger import record_recommendation
from src.app.maintenance import TASKS, artefact_status, run_task
from src.app.predictions import (
    betting_candidates,
    football_predictions,
    tennis_predictions,
    ufc_predictions,
)
from src.app.staking import PLANS, apply_daily_cap, stake_for_bet


@st.cache_data(ttl=1800, show_spinner="Chargement des rencontres…")
def _cached_football(root_text: str):
    return football_predictions(Path(root_text))


@st.cache_data(ttl=1800, show_spinner="Lecture des cartes UFC programmées…")
def _cached_ufc(root_text: str):
    return ufc_predictions(Path(root_text))


@st.cache_data(ttl=1800, show_spinner="Interrogation des cotes tennis…")
def _cached_tennis(root_text: str):
    return tennis_predictions(Path(root_text))


CSS = """
<style>
.kpi-row { display:flex; gap:0.75rem; flex-wrap:wrap; margin:0.5rem 0 1rem; }
.kpi { flex:1 1 170px; border:1px solid rgba(128,128,128,0.25); border-radius:10px;
       padding:0.75rem 0.9rem; background:rgba(128,128,128,0.06); }
.kpi .label { font-size:0.75rem; opacity:0.75; text-transform:uppercase;
              letter-spacing:0.04em; }
.kpi .value { font-size:1.35rem; font-weight:650; margin-top:0.15rem; }
.kpi .note { font-size:0.72rem; opacity:0.65; margin-top:0.2rem; }
.verdict { border-left:4px solid #b4462e; background:rgba(180,70,46,0.08);
           padding:0.9rem 1.1rem; border-radius:0 10px 10px 0; margin:0.6rem 0 1.2rem; }
.verdict h4 { margin:0 0 0.35rem; font-size:1.02rem; }
.verdict p { margin:0; font-size:0.9rem; line-height:1.5; }
.sport-card { border:1px solid rgba(128,128,128,0.25); border-radius:12px;
              padding:1rem 1.15rem; margin-bottom:0.9rem; }
.sport-card h4 { margin:0 0 0.2rem; }
.badge { display:inline-block; font-size:0.7rem; font-weight:600; padding:0.15rem 0.5rem;
         border-radius:999px; background:rgba(180,70,46,0.15); color:#b4462e;
         letter-spacing:0.03em; }
</style>
"""


def _kpi(label: str, value: str, note: str = "") -> str:
    return (
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="note">{note}</div></div>'
    )


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:+.4f}" if abs(value) < 1 else f"{value:,.2f}"
    if isinstance(value, list) and len(value) == 2:
        return f"[{value[0]:+.2%} ; {value[1]:+.2%}]"
    return str(value)


def render_verdict_banner() -> None:
    verdict = global_verdict()
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        f'<div class="verdict"><h4>⚠️ {verdict["status"]}</h4>'
        f'<p>{verdict["summary"]}</p></div>',
        unsafe_allow_html=True,
    )


def render_performance_page(root: Path) -> None:
    st.title("Performances attendues")
    render_verdict_banner()

    verdict = global_verdict()
    st.markdown(
        '<div class="kpi-row">'
        + _kpi("Pistes explorées", str(verdict["avenues_explored"]), "toutes fermées")
        + _kpi("Sports couverts", str(verdict["sports_covered"]), "tennis, football, UFC")
        + _kpi("Gain espéré", "0 %", "aucun intervalle n'exclut zéro")
        + _kpi("Argent réel", "Non autorisé", "papier uniquement")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Ce qui a été mesuré, sport par sport")
    st.caption(
        "Chiffres lus directement dans les rapports d'étude, jamais ressaisis: "
        "ils ne peuvent pas diverger des mesures qui les produisent."
    )
    for evidence in all_evidence(root):
        with st.container(border=True):
            left, right = st.columns([3, 2])
            with left:
                st.markdown(f"#### {evidence.sport}")
                st.markdown(f'<span class="badge">{evidence.status}</span>',
                            unsafe_allow_html=True)
                st.markdown(f"**{evidence.headline}**")
                st.write(evidence.detail)
                if evidence.report_paths:
                    st.caption("Rapports : " + " · ".join(evidence.report_paths))
            with right:
                if evidence.metrics:
                    table = pd.DataFrame(
                        [{"Mesure": key, "Valeur": _format(value)}
                         for key, value in evidence.metrics.items()]
                    )
                    st.dataframe(table, hide_index=True, use_container_width=True)
                else:
                    st.info("Rapport d'étude introuvable; relancer le pipeline du sport.")

    st.subheader("Modèles retenus")
    st.caption(
        "Chaque sport utilise un modèle de descripteurs purs: la cote n'est jamais une "
        "entrée. Un modèle nourri au prix ne pourrait que l'approuver, alors que l'app a "
        "besoin d'une opinion à comparer au prix. Le vainqueur est choisi en walk-forward "
        "sur une fenêtre de développement, puis mesuré sur des périodes jamais classées."
    )
    registry = pd.DataFrame(model_registry(root))
    st.dataframe(
        registry.drop(columns=["Note"]),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Log-loss (hors échantillon)": st.column_config.NumberColumn(format="%.5f"),
            "AUC": st.column_config.NumberColumn(format="%.4f"),
            "Utilise la cote": st.column_config.CheckboxColumn(),
        },
    )
    st.caption(
        "Un meilleur modèle ici ne crée aucun avantage: la barre qui compte est le gain "
        "*conditionnel au prix*, et elle n'est franchie par aucun sport."
    )

    st.subheader("Les biais réellement trouvés")
    st.caption(
        "Ils existent et sont reproductibles. Aucun n'est assez grand pour payer "
        "la marge qu'il faudrait franchir — c'est tout le résultat du projet."
    )
    for bias in verdict["biases_found"]:
        st.markdown(f"- {bias}")

    with st.expander("Pourquoi aucun gain n'est annoncé ici"):
        st.markdown(
            "Une page de performances attendues n'a de valeur que si elle peut afficher "
            "un mauvais chiffre. Les mesures de ce dépôt donnent, dans le meilleur des "
            "cas, **+0,74 % de ROI avec un intervalle de confiance à 90 % de "
            "[−0,7 % ; +2,1 %]** — indistinguable de zéro, et annulé par deux points de "
            "friction d'exécution. Afficher une projection positive reviendrait à "
            "inventer un chiffre que rien ne soutient."
        )


def render_predictions_page(root: Path) -> None:
    st.title("Matchs à venir")
    render_verdict_banner()

    # Streamlit renders every tab's body on each run, so st.tabs here would fetch
    # all three sports — including the UFC state rebuild — even to look at one.
    # A selector keeps the work to the sport actually asked for.
    sport = st.radio(
        "Sport", ["⚽ Football", "🥊 UFC", "🎾 Tennis"],
        horizontal=True, label_visibility="collapsed",
    )
    st.caption(
        "Les probabilités viennent de modèles qui n'ont jamais vu la cote. Un écart "
        "large signale le plus souvent que c'est le modèle qui se trompe, pas le prix: "
        "mesuré, le modèle football vaut −0,00080 de log-loss contre le marché."
    )

    slot = f"charger_{sport}"
    if not st.session_state.get(slot):
        left, right = st.columns([1, 3])
        with left:
            if st.button("Charger", type="primary", use_container_width=True, key=f"btn_{slot}"):
                st.session_state[slot] = True
                st.rerun()
        with right:
            st.info(
                "Rien n'est téléchargé tant que vous ne cliquez pas. Football et UFC "
                "interrogent le web; le tennis consomme en plus du quota d'API."
            )
        return

    if sport.endswith("Football"):
        _render_football(_cached_football(str(root)))
    elif sport.endswith("UFC"):
        _render_ufc(_cached_ufc(str(root)))
    else:
        _render_tennis(_cached_tennis(str(root)))


def _render_tennis(block) -> None:
    if not block.available:
        _render_unavailable(block)
        return
    frame = block.rows
    st.markdown(
        '<div class="kpi-row">'
        + _kpi("Matchs cotés", str(len(frame)), block.meta.get("competitions", ""))
        + _kpi("Joueurs notés", f"{block.meta.get('rated', 0)}/{len(frame)}",
               "présents dans la table Elo")
        + _kpi("Requêtes restantes", str(block.meta.get("remaining_requests", "?")),
               "quota mensuel The Odds API")
        + _kpi("Clé", block.meta.get("key_source", "—"))
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(block.meta["model"])
    st.warning(
        "Les écarts sont larges parce que l'Elo seul est nettement moins bon que le "
        "prix — c'est attendu, et c'est pourquoi aucun de ces écarts n'est une "
        "recommandation. Le marché intègre blessures, forfaits et forme du jour."
    )
    ranked = int((frame["score"] > 0).sum()) if "score" in frame.columns else 0
    if ranked:
        st.success(
            f"{ranked} matchs classés par recommandation. Les autres sont écartés: "
            "espérance négative, joueur non noté, ou désaccord au prix trop grand."
        )
    view = frame.copy()
    if ranked:
        view = view[view["score"] > 0]
        view.insert(0, "rang", range(1, len(view) + 1))
    display = view[[column for column in [
        "rang", "début", "compétition", "favori", "adversaire",
        "pari", "cote_pari", "p_pari", "espérance", "score",
        "cote_favori", "cote_adversaire",
        "p_marché_favori", "p_modèle_favori", "écart", "books",
    ] if column in view.columns]].rename(
        columns={"cote_pari": "cote", "p_pari": "P(pari)", "espérance": "EV"}
    )
    st.dataframe(
        display, hide_index=True, use_container_width=True,
        column_config={
            "p_marché_favori": st.column_config.ProgressColumn(
                "P(favori) marché", min_value=0, max_value=1, format="%.2f"
            ),
            "p_modèle_favori": st.column_config.ProgressColumn(
                "P(favori) Elo", min_value=0, max_value=1, format="%.2f"
            ),
            "écart": st.column_config.NumberColumn("écart", format="%+.3f"),
            "cote_favori": st.column_config.NumberColumn("cote favori", format="%.2f"),
            "cote_adversaire": st.column_config.NumberColumn("cote adv.", format="%.2f"),
            "books": st.column_config.NumberColumn("books", format="%d"),
            "cote": st.column_config.NumberColumn("cote", format="%.2f"),
            "P(pari)": st.column_config.ProgressColumn(
                "P(pari)", min_value=0, max_value=1, format="%.2f"
            ),
            "EV": st.column_config.NumberColumn("EV", format="%+.3f"),
            "score": st.column_config.NumberColumn("score", format="%.3f"),
        },
    )
    st.caption(block.meta.get("prices", ""))


def _render_unavailable(block) -> None:
    st.info(block.unavailable_reason)
    if block.meta:
        st.caption(" · ".join(f"{key}: {value}" for key, value in block.meta.items()))


def _render_ufc(block) -> None:
    if not block.available:
        _render_unavailable(block)
        return
    frame = block.rows
    scored = int(frame["p_combattant_1"].notna().sum())
    priced = int(frame["cote_1"].notna().sum()) if "cote_1" in frame else 0
    st.markdown(
        '<div class="kpi-row">'
        + _kpi("Combats programmés", str(len(frame)), block.meta.get("events", ""))
        + _kpi("Notés par le modèle", str(scored), "combattants connus des deux côtés")
        + _kpi("Avec un prix", f"{priced}/{len(frame)}", "flux MMA The Odds API")
        + _kpi("Requêtes restantes", str(block.meta.get("remaining_requests") or "—"),
               "quota mensuel")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(block.meta["prices"])
    if block.meta.get("model_note"):
        st.error(block.meta["model_note"])
    if priced < len(frame):
        st.info(
            f"{len(frame) - priced} combats sans prix: les bookmakers ne cotent une "
            "carte que peu de jours avant, et le flux MMA couvre aussi d'autres "
            "organisations. Sans prix, aucun écart au marché n'est calculable."
        )
    st.caption(
        "Classé par confiance du modèle — l'écart de sa probabilité à 50 %. "
        "L'écart au marché, quand il existe, n'est pas un avantage: les trois phases "
        "rigoureuses UFC ont toutes été rejetées."
    )
    ranked = int((frame["score"] > 0).sum()) if "score" in frame.columns else 0
    if ranked:
        st.success(
            f"{ranked} combats classés par recommandation. Les autres sont écartés: "
            "espérance négative, combattant inconnu, ou désaccord au prix trop grand "
            "pour être crédible."
        )
    view = frame.copy()
    if "score" in view.columns:
        view = view[view["score"] > 0] if ranked else view
        view.insert(0, "rang", range(1, len(view) + 1))
    wanted = [
        "rang", "date", "combattant_1", "combattant_2", "catégorie",
        "pari", "cote_pari", "p_pari", "espérance", "score",
        "p_combattant_1", "cote_1", "cote_2", "p_marché_1", "écart",
    ]
    display = view[[column for column in wanted if column in view.columns]].rename(
        columns={"p_combattant_1": "P(combattant 1)", "cote_pari": "cote",
                 "p_pari": "P(pari)", "espérance": "EV"}
    )
    st.dataframe(
        display, hide_index=True, use_container_width=True,
        column_config={
            "P(combattant 1)": st.column_config.ProgressColumn(
                "P(combattant 1)", min_value=0, max_value=1, format="%.2f"
            ),
            "elo_1": st.column_config.NumberColumn("Elo 1", format="%.0f"),
            "elo_2": st.column_config.NumberColumn("Elo 2", format="%.0f"),
            "cote_1": st.column_config.NumberColumn("cote 1", format="%.2f"),
            "cote_2": st.column_config.NumberColumn("cote 2", format="%.2f"),
            "p_marché_1": st.column_config.NumberColumn("P(1) marché", format="%.3f"),
            "écart": st.column_config.NumberColumn("écart", format="%+.3f"),
            "cote": st.column_config.NumberColumn("cote", format="%.2f"),
            "P(pari)": st.column_config.ProgressColumn(
                "P(pari)", min_value=0, max_value=1, format="%.2f"
            ),
            "EV": st.column_config.NumberColumn("EV", format="%+.3f"),
            "score": st.column_config.NumberColumn("score", format="%.3f"),
        },
    )
    st.caption(f"Source : {block.meta['source']} · {block.meta['fetched_at_utc']}")


def _render_football(block) -> None:
    if not block.available:
        _render_unavailable(block)
        return

    frame = block.rows
    st.markdown(
        '<div class="kpi-row">'
        + _kpi("Rencontres", str(len(frame)), f"jusqu'au {block.meta.get('date_max', '?')}")
        + _kpi("Divisions", str(frame["division"].nunique()), "22 suivies")
        + _kpi("Prix de référence", "Moyenne", "Pinnacle absent du flux")
        + _kpi("Équipes inconnues", str(int((~frame["équipes_connues"]).sum())),
               "promues, sans historique")
        + "</div>",
        unsafe_allow_html=True,
    )

    columns = st.columns([2, 2, 2])
    with columns[0]:
        divisions = ["Toutes"] + sorted(frame["division"].unique())
        division = st.selectbox("Division", divisions)
    with columns[1]:
        only_positive = st.checkbox(
            "Uniquement les espérances positives", value=True,
            help="Un score nul signifie espérance négative ou équipe inconnue.",
        )
    with columns[2]:
        only_known = st.checkbox("Masquer les équipes sans historique", value=True)

    view = frame.copy()
    if division != "Toutes":
        view = view[view["division"] == division]
    if only_known:
        view = view[view["équipes_connues"]]
    if only_positive:
        view = view[view["score"] > 0]

    st.subheader(f"{len(view)} rencontres, meilleure recommandation en premier")
    st.caption(block.meta.get("ranking", ""))
    if view.empty:
        st.info(
            "Aucune rencontre ne ressort avec ces filtres. Une espérance négative "
            "partout est le résultat attendu quand le modèle ne bat pas le prix."
        )
        return

    view = view.copy()
    view.insert(0, "rang", range(1, len(view) + 1))
    display = view[[
        "rang", "date", "heure", "division", "domicile", "extérieur",
        "pari", "cote_pari", "p_pari", "espérance", "score",
        "p_domicile", "p_nul", "p_extérieur",
    ]].rename(columns={
        "p_domicile": "P(dom)", "p_nul": "P(nul)", "p_extérieur": "P(ext)",
        "cote_pari": "cote", "p_pari": "P(pari)", "espérance": "EV",
    })
    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "P(dom)": st.column_config.ProgressColumn("P(dom)", min_value=0, max_value=1,
                                                      format="%.2f"),
            "P(nul)": st.column_config.ProgressColumn("P(nul)", min_value=0, max_value=1,
                                                      format="%.2f"),
            "P(ext)": st.column_config.ProgressColumn("P(ext)", min_value=0, max_value=1,
                                                      format="%.2f"),
            "cote": st.column_config.NumberColumn("cote", format="%.2f"),
            "P(pari)": st.column_config.ProgressColumn(
                "P(pari)", min_value=0, max_value=1, format="%.2f"
            ),
            "EV": st.column_config.NumberColumn("EV", format="%+.3f"),
            "score": st.column_config.NumberColumn(
                "score", format="%.3f",
                help="Espérance escomptée par l'historique des deux équipes, "
                     "et nulle si le désaccord au prix dépasse 15 points.",
            ),
        },
    )
    st.caption(f"Source : {block.meta.get('source')} · récupéré {block.meta.get('fetched_at_utc')}")


def render_staking_page(root: Path, user_id: int | None = None) -> None:
    st.title("Taille de mise")
    render_verdict_banner()
    st.caption(
        "Kelly fractionné, plafonné par pari et par jour, avec un plancher d'écart. "
        "Sous ce plancher la mise proposée est zéro — c'est le comportement voulu quand "
        "l'avantage n'est pas distinguable de zéro."
    )

    columns = st.columns([2, 2, 2])
    with columns[0]:
        bankroll = st.number_input("Bankroll (€)", min_value=10.0, value=1000.0, step=50.0)
    with columns[1]:
        plan_name = st.selectbox("Plan", list(PLANS), index=0)
    with columns[2]:
        st.metric("Plancher d'écart", f"{PLANS[plan_name].minimum_edge:.0%}")
    plan = PLANS[plan_name]

    st.subheader("Simulateur")
    left, middle, right = st.columns(3)
    with left:
        probability = st.slider("Votre probabilité", 0.01, 0.99, 0.55, 0.01)
    with middle:
        odds = st.number_input("Cote proposée", min_value=1.01, value=2.10, step=0.05)
    with right:
        result = stake_for_bet(probability, odds, bankroll, plan)
        st.metric("Mise conseillée", f"{result['stake']:.2f} €",
                  f"{result.get('edge', 0):+.2%} d'écart")
    if result["reason"]:
        st.warning(result["reason"])
    else:
        st.success(
            f"Kelly complet {result['kelly_full']:.2%}, divisé par {plan.kelly_divisor:.0f}, "
            f"plafonné à {plan.max_fraction_per_bet:.2%} de la bankroll."
        )

    st.subheader("Appliqué aux trois sports")
    chosen = st.multiselect(
        "Sports à inclure", ["Football", "UFC", "Tennis"],
        default=["Football", "UFC", "Tennis"],
        help="Chaque sport interroge sa propre source; en décocher un évite son appel.",
    )

    loaders = {
        "Football": _cached_football, "UFC": _cached_ufc, "Tennis": _cached_tennis,
    }
    frames, unavailable = [], []
    for sport in chosen:
        block = loaders[sport](str(root))
        if not block.available:
            unavailable.append(f"{sport}: {block.unavailable_reason}")
            continue
        frames.append(betting_candidates(block))
    for message in unavailable:
        st.info(message)
    if not frames:
        st.info("Aucun sport chargé.")
        return

    candidates = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if candidates.empty:
        st.info(
            "Aucun candidat: partout l'espérance est négative, le désaccord au prix est "
            "trop grand pour être crédible, ou l'historique manque. C'est le résultat "
            "attendu quand un modèle ne bat pas le marché."
        )
        return

    candidates = candidates.sort_values("score", ascending=False).reset_index(drop=True)
    proposals = [
        stake_for_bet(row["p_pari"], row["cote_pari"], bankroll, plan)
        for _, row in candidates.iterrows()
    ]
    candidates["mise"] = [item["stake"] for item in proposals]
    candidates["motif"] = [item["reason"] for item in proposals]

    # The daily cap applies per calendar day, not to the whole list at once.
    candidates["jour"] = candidates["quand"].astype(str).str.slice(0, 10)
    for day, group in candidates.groupby("jour"):
        capped = apply_daily_cap(list(group["mise"]), bankroll, plan)
        candidates.loc[group.index, "mise"] = capped

    staked = candidates[candidates["mise"] > 0]
    by_sport = staked.groupby("sport")["mise"].sum().to_dict() if not staked.empty else {}
    st.markdown(
        '<div class="kpi-row">'
        + _kpi("Candidats examinés", str(len(candidates)), "score positif uniquement")
        + _kpi("Mises proposées", str(len(staked)), f"plancher {plan.minimum_edge:.0%} d'écart")
        + _kpi("Exposition totale", f"{staked['mise'].sum():.2f} €",
               f"plafond {plan.max_fraction_per_day:.0%} par jour")
        + _kpi("Sports concernés", str(len(by_sport)),
               " · ".join(f"{name} {value:.0f} €" for name, value in by_sport.items()))
        + "</div>",
        unsafe_allow_html=True,
    )

    if staked.empty:
        st.info(
            "Aucune mise proposée: tous les écarts sont sous le plancher. "
            "C'est le comportement voulu, pas une panne."
        )
        st.dataframe(
            candidates[["sport", "quand", "rencontre", "pari", "cote_pari",
                        "espérance", "motif"]].head(20),
            hide_index=True, use_container_width=True,
        )
        return

    st.dataframe(
        staked[["sport", "quand", "rencontre", "pari", "cote_pari", "p_pari",
                "espérance", "score", "mise"]],
        hide_index=True, use_container_width=True,
        column_config={
            "cote_pari": st.column_config.NumberColumn("cote", format="%.2f"),
            "p_pari": st.column_config.ProgressColumn(
                "P(pari)", min_value=0, max_value=1, format="%.2f"
            ),
            "espérance": st.column_config.NumberColumn("EV", format="%+.3f"),
            "score": st.column_config.NumberColumn("score", format="%.3f"),
            "mise": st.column_config.NumberColumn("mise (€)", format="%.2f"),
        },
    )
    st.warning(
        "Exercice de dimensionnement, pas une recommandation. Aucun avantage n'est "
        "démontré sur ces marchés: sur les huit pistes étudiées, le meilleur rendement "
        "mesuré est +0,74 % avec un intervalle contenant zéro. Le suivi doit rester "
        "sur papier."
    )

    _render_ledger_recording(root, staked, user_id)


def _render_ledger_recording(root: Path, staked: pd.DataFrame, user_id: int | None) -> None:
    """Write a proposed stake into the app's bet ledger."""
    st.subheader("Enregistrer dans le carnet")
    if user_id is None:
        st.info("Connectez-vous pour enregistrer une mise.")
        return
    st.caption(
        "La cote et la probabilité sont figées au moment de l'enregistrement. Un "
        "carnet dont les cotes suivent le marché ne peut plus répondre à la seule "
        "question qu'il sert à trancher: ce pari a-t-il payé ?"
    )

    labels = {
        f"{row.sport} · {row.rencontre} → {row.pari} @ {row.cote_pari:.2f} "
        f"({row.mise:.2f} €)": index
        for index, row in staked.iterrows()
    }
    chosen = st.selectbox("Pari à enregistrer", ["—"] + list(labels))
    if chosen == "—":
        return
    row = staked.loc[labels[chosen]]
    amount = st.number_input(
        "Mise (€)", min_value=0.01, value=float(max(row["mise"], 0.01)), step=0.5,
    )
    config, source = load_config(root)
    destinations = st.multiselect(
        "Où enregistrer", ["Carnet local (SQLite)", "Dépôt GitHub (durable)"],
        default=(["Carnet local (SQLite)", "Dépôt GitHub (durable)"]
                 if config.configured else ["Carnet local (SQLite)"]),
        help="Sur Streamlit Cloud le carnet local est effacé à chaque redéploiement; "
             "seule la copie GitHub survit.",
    )
    if not config.configured:
        st.info(
            "GitHub non configuré. Ajoutez dans les secrets Streamlit un token à "
            "permission « Contents: Read and write » :\n\n"
            '```toml\nGITHUB_TOKEN = "github_pat_..."\n'
            f'GITHUB_REPO = "{config.repository or "utilisateur/depot"}"\n```'
        )
    else:
        st.caption(
            f"GitHub: `{config.repository}` · branche `{config.branch}` · "
            f"token lu depuis les {source}. La branche du carnet n'est pas déployée, "
            "donc un enregistrement ne redémarre pas l'app."
        )

    if st.button("Enregistrer ce pari", type="primary"):
        payload = row.to_dict()
        payload["utilisateur"] = int(user_id)
        payload["mise"] = float(amount)
        results: list[str] = []
        failed = False
        if "Carnet local (SQLite)" in destinations:
            ok, message = record_recommendation(
                root / "bets" / "unified_app.db", int(user_id), payload, float(amount)
            )
            results.append(f"Local: {message}")
            failed |= not ok
        if "Dépôt GitHub (durable)" in destinations:
            created, branch_message = ensure_branch(config)
            if not created:
                results.append(f"GitHub: {branch_message}")
                failed = True
            else:
                ok, message = append_bet(config, payload)
                results.append(f"GitHub: {message}")
                failed |= not ok
        for line in results:
            (st.error if failed else st.success)(line)

    with st.expander("Paris enregistrés sur GitHub"):
        if not config.configured:
            st.caption("Configurez GITHUB_TOKEN pour lire le carnet durable.")
        else:
            frame, message = ledger_frame(config)
            st.caption(message)
            if not frame.empty:
                st.dataframe(frame, hide_index=True, use_container_width=True)


def render_maintenance_page(root: Path) -> None:
    st.title("Mise à jour des données et des modèles")
    st.caption(
        "Chaque tâche lance un script du dépôt avec **l'interpréteur qui fait tourner "
        f"cette app** (`{sys.executable}`). C'est ce qui garantit que les modèles "
        "produits soient lisibles ici: entraînés ailleurs, ils échouaient au chargement."
    )
    st.warning(
        "Ces tâches durent plusieurs minutes et bloquent l'interface pendant "
        "l'exécution. Respectez l'ordre: les données d'abord, les modèles ensuite — "
        "un modèle entraîné sur des données périmées le reste."
    )

    for index, task in enumerate(TASKS, start=1):
        with st.container(border=True):
            header, action = st.columns([4, 1])
            with header:
                st.markdown(f"**{index}. {task.label}** · _{task.minutes}_")
                st.caption(task.description)
            with action:
                launch = st.button("Lancer", key=f"run_{task.key}", use_container_width=True)
            status = pd.DataFrame(artefact_status(root, task))
            stale = status["Âge (jours)"].fillna(999).max() if len(status) else 0
            if (status["État"] == "absent").any():
                st.error("Artefact manquant: cette tâche n'a jamais été exécutée ici.")
            elif stale > 7:
                st.warning(f"Le plus ancien artefact date de {int(stale)} jours.")
            st.dataframe(status, hide_index=True, use_container_width=True)

            if launch:
                with st.spinner(f"{task.label} en cours… ne fermez pas l'onglet"):
                    result = run_task(root, task.key)
                if result["ok"]:
                    st.success(
                        f"Terminé en {result['seconds']:.0f} s. "
                        "Videz le cache des prédictions pour voir l'effet."
                    )
                else:
                    st.error(f"Échec (code {result.get('returncode', '?')}).")
                with st.expander("Sortie de la commande", expanded=not result["ok"]):
                    st.code(result["output"] or "(aucune sortie)")

    st.divider()
    if st.button("Vider le cache des prédictions"):
        st.cache_data.clear()
        st.success("Cache vidé: le prochain chargement ira rechercher les données.")
