import os
import json
import asyncio
import random
from pathlib import Path
from dotenv import load_dotenv
import discord
from discord import app_commands
from discord.ext import commands
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from google import genai

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("GUILD_ID"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ai_client = genai.Client(api_key=GEMINI_API_KEY)

intents = discord.Intents.default()
intents.guilds = True
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix="/", intents=intents)
scheduler = AsyncIOScheduler()

HISTORY_FILE = Path("dilemmas_history.json")


# ==============================================================================
# 1. GESTION DE L'HISTORIQUE ET VALIDATION SÉMANTIQUE DES DILEMMES
# ==============================================================================

def load_dilemma_history() -> list[dict]:
    """Charge la liste des dilemmes déjà soumis depuis le fichier JSON."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Erreur lecture historique dilemmes : {e}")
        return []


def save_dilemma_to_history(situation: str, opt_a: str, opt_b: str):
    """Enregistre le nouveau dilemme dans l'historique local."""
    history = load_dilemma_history()
    history.append({
        "situation": situation,
        "option_a": opt_a,
        "option_b": opt_b
    })
    if len(history) > 50:
        history = history[-50:]
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Erreur écriture historique dilemmes : {e}")


def check_semantic_similarity(candidate_summary: str, history: list[dict]) -> bool:
    """
    Vérifie si le dilemme candidat ressemble de près ou de loin à un dilemme existant.
    Renvoie True s'il est thématiquement proche (doit être jeté), False s'il est inédit.
    """
    if not history:
        return False

    past_topics = [f"- {h['situation']} (Choix: {h['option_a']} VS {h['option_b']})" for h in history[-20:]]
    past_topics_str = "\n".join(past_topics)

    prompt = f"""
Tu es un filtre anti-doublon et anti-redondance thématique impitoyable.
Voici la liste des dilemmes déjà soumis par le passé :
{past_topics_str}

Voici le NOUVEAU dilemme candidat proposé :
{candidate_summary}

ÉVALUATION :
Est-ce que ce candidat réutilise le même concept de fond, la même mécanique de honte, le même type de contrainte physique ou la même thématique qu'un des dilemmes passés (même formulé différemment ou avec d'autres mots) ?

Réponds STRICTEMENT par un seul mot :
- REJET (si c'est thématiquement proche, un dérivé ou une variante d'un sujet passé)
- VALIDE (si le sujet, le contexte et la mécanique sont 100% frais et inédits)
"""
    try:
        res = ai_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        verdict = res.text.strip().upper()
        return "REJET" in verdict
    except Exception as e:
        print(f"Erreur vérification similarité : {e}")
        return False


def generate_hardcore_dilemma() -> tuple[str, str, str]:
    """Génère un dilemme inédit, calibré 50/50, avec rejet automatique des idées proches."""
    history = load_dilemma_history()

    forbidden_list = ""
    if history:
        items = [f"{i}. {h['situation']} | Choix: {h['option_a']} VS {h['option_b']}" for i, h in enumerate(history[-20:], 1)]
        forbidden_list = "DILEMMES DÉJÀ PROPOSÉS (INTERDICTION STRICTE DE REPRENDRE LEURS THÈMES) :\n" + "\n".join(items)

    for attempt in range(1, 4):
        prompt = f"""
Tu es un concepteur de dilemmes moraux extrêmes et d'expériences de pensée.
Ton but absolu est de concevoir un dilemme "Tu préfères" INÉDIT, ultra-serré et divisant un groupe d'adultes rigoureusement à 50% / 50%.

{forbidden_list}

CONSIGNES DE CONCEPTION :
1. Originalité conceptuelle radicale : Explore des registres complètement absents des archives ci-dessus.
2. Balance 50/50 absolue : Les deux options doivent infliger un coût psychologique, social ou physique STRICTEMENT ÉQUIVALENT. Aucune des deux options ne doit être la porte de sortie facile.
3. Pas de contenu impliquant des mineurs.
4. Format de réponse STRICT (3 lignes exactes, rien d'autre) :
SITUATION: [Contexte dramatique ou absurde court]
OPTION_A: [Premier fardeau concret]
OPTION_B: [Second fardeau de poids rigoureusement identique]
"""
        try:
            response = ai_client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt
            )
            lines = [line.strip() for line in response.text.strip().splitlines() if line.strip()]
            situation, opt_a, opt_b = "Choix à balance égale :", "Option A", "Option B"
            for line in lines:
                if line.startswith("SITUATION:"):
                    situation = line.replace("SITUATION:", "").strip()
                elif line.startswith("OPTION_A:"):
                    opt_a = line.replace("OPTION_A:", "").strip()
                elif line.startswith("OPTION_B:"):
                    opt_b = line.replace("OPTION_B:", "").strip()

            candidate_summary = f"{situation} | {opt_a} VS {opt_b}"

            is_too_close = check_semantic_similarity(candidate_summary, history)
            if is_too_close:
                print(f"[REJET TENTATIVE {attempt}] Sujet sémantiquement trop proche d'un précédent. Nouveau tirage...")
                continue

            save_dilemma_to_history(situation, opt_a, opt_b)
            print(f"[VALIDÉ] Nouveau dilemme inédit retenu (tentative {attempt}).")
            return situation, opt_a, opt_b

        except Exception as e:
            print(f"Erreur tentative {attempt} : {e}")

    return (
        "Pour éviter une condamnation arbitraire, vous devez choisir l'une de ces deux sanctions :",
        "L'historique complet de vos recherches privées des 5 dernières années est lu en direct devant vos proches",
        "Chaque conversation que vous aurez pour le reste de votre vie sera diffusée sur haut-parleur"
    )


class PersistentDilemmaView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    def extract_data_and_vote(self, message: discord.Message, user: discord.Member, vote_for: str) -> discord.Embed:
        old_embed = message.embeds[0]
        description = old_embed.description

        field_a_val = old_embed.fields[0].value
        field_b_val = old_embed.fields[1].value

        raw_voters_a = [] if "Aucun vote" in field_a_val else [v.strip() for v in field_a_val.split(", ") if v.strip()]
        raw_voters_b = [] if "Aucun vote" in field_b_val else [v.strip() for v in field_b_val.split(", ") if v.strip()]

        voters_a = set(raw_voters_a)
        voters_b = set(raw_voters_b)

        user_mention = user.mention

        if vote_for == "A":
            voters_b.discard(user_mention)
            voters_a.add(user_mention)
        else:
            voters_a.discard(user_mention)
            voters_b.add(user_mention)

        new_embed = discord.Embed(
            title=old_embed.title,
            description=description,
            color=discord.Color.dark_red()
        )

        val_a = ", ".join(voters_a) if voters_a else "Aucun vote pour l'instant."
        val_b = ", ".join(voters_b) if voters_b else "Aucun vote pour l'instant."

        new_embed.add_field(name=f"🔴 Votes Option A ({len(voters_a)})", value=val_a, inline=False)
        new_embed.add_field(name=f"🔵 Votes Option B ({len(voters_b)})", value=val_b, inline=False)
        new_embed.set_footer(text="Cliquez ci-dessous pour voter ou modifier votre choix !")

        return new_embed

    @discord.ui.button(label="Voter A", style=discord.ButtonStyle.danger, custom_id="dilemma_btn_a")
    async def btn_a(self, interaction: discord.Interaction, button: discord.ui.Button):
        new_embed = self.extract_data_and_vote(interaction.message, interaction.user, "A")
        await interaction.response.edit_message(embed=new_embed, view=self)

    @discord.ui.button(label="Voter B", style=discord.ButtonStyle.primary, custom_id="dilemma_btn_b")
    async def btn_b(self, interaction: discord.Interaction, button: discord.ui.Button):
        new_embed = self.extract_data_and_vote(interaction.message, interaction.user, "B")
        await interaction.response.edit_message(embed=new_embed, view=self)


# ==============================================================================
# 2. LOGIQUE MATINALE (ENVOI INDIVIDUEL MP)
# ==============================================================================

async def get_comprehensive_user_context(guild: discord.Guild, member: discord.Member) -> str:
    user_messages = []
    mentions_and_replies = []

    for channel in guild.text_channels:
        perms = channel.permissions_for(guild.me)
        if not (perms.view_channel and perms.read_message_history):
            continue

        try:
            async for msg in channel.history(limit=100):
                content = msg.content.strip()
                if not content:
                    continue

                if msg.author.id == member.id and len(user_messages) < 10:
                    user_messages.append(f"- {content}")
                elif (member.mentioned_in(msg) or str(member.id) in msg.content) and len(mentions_and_replies) < 6:
                    mentions_and_replies.append(f"- {msg.author.display_name} a dit : \"{content}\"")

                if len(user_messages) >= 10 and len(mentions_and_replies) >= 6:
                    break
        except Exception:
            continue

        if len(user_messages) >= 10 and len(mentions_and_replies) >= 6:
            break

    context_parts = []
    if user_messages:
        context_parts.append("Messages récents postés par cette personne :\n" + "\n".join(user_messages))
    else:
        context_parts.append("Cette personne n'a presque rien posté récemment (membre silencieux).")

    if mentions_and_replies:
        context_parts.append("\nMessages où les autres membres lui parlent ou la mentionnent :\n" + "\n".join(mentions_and_replies))

    return "\n".join(context_parts)


def generate_morning_text(display_name: str, context: str, is_victim: bool) -> str:
    sweet_styles = [
        "coach de vie surmotivé façon TED Talk, mais avec un second degré bienveillant",
        "pote sincère qui pose les termes et prend le temps d'envoyer de la vraie bonne énergie",
        "observateur décalé qui fait une analyse psychologique élogieuse basée sur ses messages",
        "poète moderne du quotidien, mi-philosophique mi-drôle",
        "discours d'avant-match de vestiaire pour chauffer la personne avant sa journée"
    ]

    roast_styles = [
        "chroniqueur satirique impitoyable mais hilarant qui décortique son comportement",
        "pote dépité qui remet gentiment les pendules à l'heure avec beaucoup d'ironie",
        "enquêteur qui a analysé ses messages et en tire des conclusions désastreuses",
        "critique gastronomique/littéraire qui note très sévèrement la prestation de la personne sur le serveur",
        "discours faussement solennel pour lui décerner la palme de la flemme ou du flop"
    ]

    chosen_style = random.choice(roast_styles if is_victim else sweet_styles)

    common_instructions = f"""
Prénom / Nom d'appel à utiliser impérativement : '{display_name}'.
Voici le contexte des discussions sur le serveur :
---
{context}
---

CONSIGNES SUR L'IDENTITÉ ET LE GENRE :
1. Nom d'appel : Appelle la personne UNIQUEMENT par son nom d'affichage '{display_name}'. N'invente pas d'autre nom.
2. Genre et accords : Analyse les messages pour déduire si '{display_name}' est un homme ou une femme. Accorde TOUS tes adjectifs et participes passés en conséquence.
"""

    if is_victim:
        prompt = f"""
Tu es un pote sur un serveur Discord. Tu dois rédiger le tacle du matin destiné à '{display_name}'.
{common_instructions}

Ton angle d'attaque du jour : adopte un ton de **{chosen_style}**.

Consignes :
- Développe un message consistant (3 à 5 phrases, environ 50 à 90 mots). Ne sois pas trop bref, pose bien la vanne.
- Fais des références précises à ses messages récents ou à sa manière de s'exprimer.
- Si le contexte indique qu'il/elle n'a pas parlé, clashe-le/la sur son statut de fantôme passif sur le serveur.
- Reste dans le chambrage entre potes : drôle, piquant, mais sans haine ni vulgarité gratuite.
- Réponds UNIQUEMENT le texte du message, sans guillemets, sans titre.
"""
    else:
        prompt = f"""
Tu es un ami proche et chaleureux sur un serveur Discord. Tu dois rédiger un mot doux / message d'encouragement matinal personnalisé pour '{display_name}'.
{common_instructions}

Ton style du jour : adopte un ton de **{chosen_style}**.

Consignes :
- Écris un texte riche, vivant et sympa (3 à 5 phrases, environ 50 à 90 mots). Ne fais pas un message expéditif de deux lignes.
- Inspire-toi réellement de ce qu'il/elle raconte ou de ses passions pour que ce soit du sur-mesure.
- Si le contexte indique qu'il/elle n'a pas beaucoup parlé récemment, dis-lui avec humour et bienveillance qu'il/elle manque aux discussions.
- Sois naturel, évite le ton robotique ou corporate.
- Réponds UNIQUEMENT le texte du message, sans guillemets, sans titre.
"""

    try:
        response = ai_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Erreur API Gemini pour {display_name}: {e}")
        return "Passe une excellente journée pleine d'énergie !" if not is_victim else "C'est tombé sur toi ce matin... fais un effort aujourd'hui !"


async def run_daily_routine(dry_run: bool = False):
    mode = "[MODE SIMULATION / AUCUN ENVOI]" if dry_run else "[MODE RÉEL / ENVOI DM]"
    print(f"\n==================== DÉBUT ROUTINE MATINALE {mode} ====================")

    guild = bot.get_guild(GUILD_ID)
    if not guild:
        print(f"Erreur : Serveur {GUILD_ID} introuvable.")
        return

    members = [m for m in guild.members if not m.bot]
    if not members:
        print("Aucun membre trouvé.")
        return

    victim = random.choice(members)
    print(f"🎯 Victime sélectionnée : {victim.display_name} (@{victim.name})\n")

    for member in members:
        is_victim = (member.id == victim.id)
        role_label = "CLASH" if is_victim else "GENTIL"

        context = await get_comprehensive_user_context(guild, member)
        print(f"👤 Membre : {member.display_name} (@{member.name}) | Rôle : {role_label}")

        generated_text = generate_morning_text(member.display_name, context, is_victim)
        prefix = "💥 **Le tacle du matin :**\n" if is_victim else "☀️ **Bonjour !**\n"
        full_message = f"{prefix}{generated_text}"

        print(f"   ↳ Message généré :\n\"\"\"\n{full_message}\n\"\"\"")

        if not dry_run:
            try:
                await member.send(full_message)
                print("   ↳ Statut : [ENVOYÉ EN MP]")
            except discord.Forbidden:
                print("   ↳ Statut : [ÉCHEC - MP FERMÉS]")
            except Exception as e:
                print(f"   ↳ Statut : [ERREUR] {e}")
            await asyncio.sleep(2.5)
        else:
            print("   ↳ Statut : [SIMULATION - Non envoyé]")

        print("-" * 50)

    print(f"==================== FIN ROUTINE MATINALE {mode} ====================\n")


# ==============================================================================
# 3. INITIALISATION & COMMANDES SLASH
# ==============================================================================

@bot.event
async def on_ready():
    print(f"Bot connecté en tant que : {bot.user.name} ({bot.user.id})")

    bot.add_view(PersistentDilemmaView())

    guild_obj = discord.Object(id=GUILD_ID)
    bot.tree.copy_global_to(guild=guild_obj)
    synced = await bot.tree.sync(guild=guild_obj)
    print(f"{len(synced)} commande(s) slash synchronisée(s).")

    # Évite les doublons de jobs grâce à id et replace_existing
    scheduler.add_job(
        run_daily_routine,
        trigger=CronTrigger(hour=8, minute=0, timezone="Europe/Paris"),
        kwargs={"dry_run": False},
        id="daily_morning_routine",
        replace_existing=True
    )

    if not scheduler.running:
        scheduler.start()
        print("Planificateur matinal actif (08:00 Europe/Paris).")


@bot.tree.command(name="dilemme", description="Lance un dilemme cornélien inédit avec votes en direct")
async def dilemme(interaction: discord.Interaction):
    await interaction.response.defer()

    situation, opt_a, opt_b = generate_hardcore_dilemma()

    desc = (
        f"**{situation}**\n\n"
        f"🔴 **Option A :**\n{opt_a}\n\n"
        f"🔵 **Option B :**\n{opt_b}"
    )

    embed = discord.Embed(
        title="⚡ DILEMME CORNÉLIEN ⚡",
        description=desc,
        color=discord.Color.dark_red()
    )
    embed.add_field(name="🔴 Votes Option A (0)", value="Aucun vote pour l'instant.", inline=False)
    embed.add_field(name="🔵 Votes Option B (0)", value="Aucun vote pour l'instant.", inline=False)
    embed.set_footer(text="Cliquez ci-dessous pour voter ou modifier votre choix !")

    view = PersistentDilemmaView()
    await interaction.followup.send(embed=embed, view=view)


@bot.tree.command(name="test_matin", description="Simule la tournée matinale dans les logs sans envoyer de DM")
@app_commands.default_permissions(administrator=True)
async def test_matin(interaction: discord.Interaction):
    await interaction.response.send_message(
        "Simulation lancée ! Regarde les logs Square Cloud pour inspecter les messages.",
        ephemeral=True
    )
    await run_daily_routine(dry_run=True)


@bot.tree.command(name="envoyer_maintenant", description="Force la tournée réelle matinale en DM")
@app_commands.default_permissions(administrator=True)
async def envoyer_maintenant(interaction: discord.Interaction):
    await interaction.response.send_message(
        "Tournée réelle démarrée. Les DM sont en cours d'envoi.",
        ephemeral=True
    )
    await run_daily_routine(dry_run=False)

# ========================================================
# MODULE : QUESTIONS DE VITESSE & JUSTE PRIX (TIMER 10S)
# ========================================================

import re
import time
import asyncio
from difflib import SequenceMatcher
import discord
from discord import app_commands

CHAN_LOGS_ORGA_ID = 1551027822925971536  # ID de votre salon orga

SESSION_VITESSE = {
    "actif": False,
    "channel_id": None,
    "num_question": 0,
    "texte_question": "",
    "reponse_cible_brute": "",
    "reponse_cible_num": None,      # float ou None
    "top_depart": 0.0,
    "task_timer": None,
    "reponses_question": {}         # { user_id: { "membre": Member, "chrono": float, "texte": str, "val_num": float, "score": float } }
}


def est_role_orga(user: discord.Member) -> bool:
    """Vérifie si l'utilisateur a un rôle organisateur/admin."""
    if user.guild_permissions.administrator:
        return True
    roles_orga = {"orga", "organisateur", "admin", "mj", "maitre du jeu", "animation"}
    return any(r.name.lower() in roles_orga for r in user.roles)


def extraire_nombre(texte: str):
    """Tente d'extraire une valeur numérique flottante ou entière depuis un texte."""
    texte_propre = texte.replace(",", ".").replace(" ", "")
    match = re.search(r"[-+]?\d*\.?\d+", texte_propre)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


async def timer_10_secondes(channel: discord.TextChannel, num_q: int):
    """Attend 10 secondes puis clôture automatiquement la question avec le podium."""
    await asyncio.sleep(10)
    # Vérifie que la session active correspond bien toujours à cette question
    if SESSION_VITESSE["actif"] and SESSION_VITESSE["num_question"] == num_q:
        await cloturer_et_publier_resultats(channel)


async def demarrer_nouvelle_question(channel: discord.TextChannel, question_str: str, reponse_cible: str, auteur: discord.Member):
    """Initialise la question, stocke la réponse cible et lance le compte à rebours de 10s."""
    # Annulation d'un timer précédent si une question était encore en cours
    if SESSION_VITESSE["task_timer"] and not SESSION_VITESSE["task_timer"].done():
        SESSION_VITESSE["task_timer"].cancel()

    SESSION_VITESSE["actif"] = True
    SESSION_VITESSE["channel_id"] = channel.id
    SESSION_VITESSE["num_question"] += 1
    SESSION_VITESSE["texte_question"] = question_str
    SESSION_VITESSE["reponse_cible_brute"] = reponse_cible.strip()
    SESSION_VITESSE["reponse_cible_num"] = extraire_nombre(reponse_cible)
    SESSION_VITESSE["reponses_question"].clear()
    SESSION_VITESSE["top_depart"] = time.time()

    embed = discord.Embed(
        title=f"⚡ QUESTION #{SESSION_VITESSE['num_question']} — ⏳ 10 SECONDES !",
        description=(
            f"# {question_str}\n\n"
            "⏱️ **Vous avez 10 secondes chrono pour répondre dans ce chat !**\n"
            "🔒 *Vos messages sont masqués automatiquement à la milliseconde.*\n"
            "🎯 *Le candidat le plus proche l'emporte (départagé au chrono en cas d'égalité).*",
        ),
        color=discord.Color.red()
    )
    embed.set_footer(text=f"Lancée par {auteur.display_name} • Fin dans 10 secondes pile !")
    await channel.send(embed=embed)

    # Lancement du compte à rebours en tâche d'arrière-plan
    SESSION_VITESSE["task_timer"] = asyncio.create_task(timer_10_secondes(channel, SESSION_VITESSE["num_question"]))


async def cloturer_et_publier_resultats(channel: discord.TextChannel):
    """Arrête les réponses, classe les candidats selon la proximité puis le chrono, et publie le podium."""
    SESSION_VITESSE["actif"] = False

    reponses = list(SESSION_VITESSE["reponses_question"].values())
    cible_num = SESSION_VITESSE["reponse_cible_num"]
    cible_texte = SESSION_VITESSE["reponse_cible_brute"].lower()

    if cible_num is not None:
        # Tri numérique : le plus proche (écart absolu minimal), puis le plus rapide
        # Clé de tri : (a-t-il mis un nombre ? [0=oui, 1=non], écart absolu, chrono)
        for r in reponses:
            val = r["val_num"]
            r["ecart"] = abs(val - cible_num) if val is not None else float("inf")
        reponses.sort(key=lambda x: (0 if x["ecart"] != float("inf") else 1, x["ecart"], x["chrono"]))
    else:
        # Tri textuel : ressemblance maximale (ratio), puis le plus rapide
        for r in reponses:
            ratio = SequenceMatcher(None, cible_texte, r["texte"].lower()).ratio()
            r["ratio"] = ratio
        reponses.sort(key=lambda x: (-x["ratio"], x["chrono"]))

    # Affichage du récapitulatif
    medailles = ["🥇", "🥈", "🥉"]
    lignes = []
    for i, data in enumerate(reponses[:8]):
        symbole = medailles[i] if i < 3 else f"`#{i+1}`"
        nom = data["membre"].display_name
        chrono_str = f"`{data['chrono']:.2f}s`"

        if cible_num is not None:
            if data["ecart"] != float("inf"):
                info_ecart = f"Écart : **{data['ecart']:g}**" if data["ecart"] > 0 else "🎯 **PIED PLEIN !**"
                lignes.append(f"{symbole} **{nom}** — {data['texte']} ({info_ecart} en {chrono_str})")
            else:
                lignes.append(f"{symbole} **{nom}** — {data['texte']} *(non numérique)* en {chrono_str}")
        else:
            pct = int(data["ratio"] * 100)
            lignes.append(f"{symbole} **{nom}** — « {data['texte']} » ({pct}% match en {chrono_str})")

    texte_recap = "\n".join(lignes) if lignes else "*Aucun candidat n'a répondu dans les 10 secondes.*"

    embed_resultat = discord.Embed(
        title=f"⌛ TEMPS ÉCOULÉ — RÉSULTATS Q#{SESSION_VITESSE['num_question']}",
        description=(
            f"**Énoncé :** {SESSION_VITESSE['texte_question']}\n"
            f"🎯 **Réponse officielle :** `{SESSION_VITESSE['reponse_cible_brute']}`\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{texte_recap}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ),
        color=discord.Color.gold()
    )
    await channel.send(embed=embed_resultat)


# --------------------------------------------------------
# INTERCEPTION DU CHAT (!q et RÉPONSES DES CANDIDATS)
# --------------------------------------------------------

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    contenu = message.content.strip()

    # --- A. COMMANDE ORGA : !q QUESTION | REPONSE ---
    if contenu.lower().startswith("!q ") and isinstance(message.author, discord.Member) and est_role_orga(message.author):
        corps = contenu[3:].strip()
        try:
            await message.delete()  # Masque la question et la réponse pour les candidats
        except discord.DiscordException:
            pass

        if "|" in corps:
            question_texte, reponse_cible = corps.split("|", 1)
        else:
            question_texte, reponse_cible = corps, ""

        await demarrer_nouvelle_question(message.channel, question_texte.strip(), reponse_cible.strip(), message.author)
        return

    # --- B. RÉPONSE D'UN CANDIDAT PENDANT LES 10 SECONDES ---
    if SESSION_VITESSE["actif"] and message.channel.id == SESSION_VITESSE["channel_id"]:
        if not est_role_orga(message.author):
            # 1. Suppression instantanée de la réponse
            try:
                await message.delete()
            except discord.DiscordException as e:
                print(f"⚠️ Erreur suppression: {e}")

            # 2. Chrono et calcul
            chrono = time.time() - SESSION_VITESSE["top_depart"]
            uid = message.author.id

            # Une seule proposition par candidat par question
            if uid in SESSION_VITESSE["reponses_question"]:
                return

            val_num = extraire_nombre(contenu)

            SESSION_VITESSE["reponses_question"][uid] = {
                "membre": message.author,
                "chrono": chrono,
                "texte": contenu,
                "val_num": val_num
            }

            # 3. Alerte en temps réel chez les orgas
            salon_orga = message.guild.get_channel(CHAN_LOGS_ORGA_ID)
            if salon_orga:
                rang = len(SESSION_VITESSE["reponses_question"])
                embed_log = discord.Embed(
                    title=f"⚡ Q#{SESSION_VITESSE['num_question']} — Réponse #{rang} en `{chrono:.2f}s`",
                    description=(
                        f"**Candidat :** {message.author.mention} (`{message.author.display_name}`)\n"
                        f"**Proposition interceptée :** `{contenu}`\n"
                        f"**Cible :** `{SESSION_VITESSE['reponse_cible_brute']}`"
                    ),
                    color=discord.Color.green(),
                    timestamp=discord.utils.utcnow()
                )
                await salon_orga.send(embed=embed_log)

            return

    await bot.process_commands(message)


# Commande de secours si vous devez couper manuellement avant les 10s
@bot.tree.command(name="stop_vitesse", description="Interrompt les 10 secondes et force l'affichage du podium.")
@app_commands.check(est_orga_ou_admin)
async def stop_vitesse(interaction: discord.Interaction):
    if not SESSION_VITESSE["actif"]:
        await interaction.response.send_message("Aucune question active.", ephemeral=True)
        return

    if SESSION_VITESSE["task_timer"] and not SESSION_VITESSE["task_timer"].done():
        SESSION_VITESSE["task_timer"].cancel()

    await interaction.response.defer(ephemeral=True)
    await cloturer_et_publier_resultats(interaction.channel)
    await interaction.followup.send("🛑 Question clôturée manuellement.", ephemeral=True)

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
