import os
import asyncio
import random
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


async def get_comprehensive_user_context(guild: discord.Guild, member: discord.Member) -> str:
    """
    Récupère :
    1. Ce que la personne écrit (pour capter ses délires et sujets récents).
    2. Ce que les autres lui disent (pour déduire le genre et les accords).
    """
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

                # Messages écrits par le membre
                if msg.author.id == member.id and len(user_messages) < 10:
                    user_messages.append(f"- {content}")

                # Messages des autres qui le mentionnent ou lui répondent
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
    """Génère le texte du matin avec le display_name et déduction automatique du genre."""
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
2. Genre et accords : Analyse les messages pour déduire si '{display_name}' est un homme ou une femme (regarde les adjectifs employés par les autres ou par la personne elle-même). Accorde TOUS tes adjectifs et participes passés en conséquence (ex: 'prêt/prête', 'fatigué/fatiguée', 'motivé/motivée'). Si c'est ambigu, privilégie des tournures neutres ou masculines par défaut.
"""

    if is_victim:
        prompt = f"""
Tu es un pote sur un serveur Discord. Tu dois rédiger le tacle du matin destiné à '{display_name}'.
{common_instructions}

Ton angle d'attaque du jour : adopte un ton de **{chosen_style}**.

Consignes :
- Développe un message consistant et bien écrit (3 à 5 phrases, environ 50 à 90 mots). Ne sois pas trop bref, pose bien la vanne.
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

        # Contexte
        context = await get_comprehensive_user_context(guild, member)
        print(f"👤 Membre : {member.display_name} (@{member.name}) | Rôle : {role_label}")

        # Utilisation stricte du display_name
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


@bot.event
async def on_ready():
    print(f"Bot connecté en tant que : {bot.user.name} ({bot.user.id})")

    guild_obj = discord.Object(id=GUILD_ID)
    bot.tree.copy_global_to(guild=guild_obj)
    synced = await bot.tree.sync(guild=guild_obj)
    print(f"{len(synced)} commande(s) slash synchronisée(s) sur le serveur {GUILD_ID}.")

    scheduler.add_job(
        run_daily_routine,
        trigger=CronTrigger(hour=8, minute=0, timezone="Europe/Paris"),
        kwargs={"dry_run": False}
    )
    scheduler.start()
    print("Planificateur matinal actif (08:00 Europe/Paris).")


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
def generate_hardcore_dilemma() -> tuple[str, str, str]:
    """Génère un dilemme cornélien et extrême, et renvoie (titre/mise en situation, option_a, option_b)."""
    prompt = """
Génère un dilemme "Tu préfères" extrêmement difficile, absurde ou cornélien pour animer un débat animé entre potes.
Les deux options doivent être quasi impossibles à départager, très inconfortables, honteuses ou intenses.

Règles strictes :
- Pas de contenu impliquant des mineurs.
- Uniquement des situations fictives, des sacrifices moraux, des choix absurdes, ou de la honte sociale entre adultes.
- Format de réponse STRICTEMENT attendu (3 lignes, rien d'autre) :
SITUATION: [Une phrase courte qui pose le contexte dramatique ou absurde]
OPTION_A: [Le premier choix, percutant et précis]
OPTION_B: [Le second choix, tout aussi difficile ou horrible]
"""

    try:
        response = ai_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        lines = [line.strip() for line in response.text.strip().splitlines() if line.strip()]
        
        situation = "Le choix impossible du jour :"
        opt_a = "Option 1"
        opt_b = "Option 2"

        for line in lines:
            if line.startswith("SITUATION:"):
                situation = line.replace("SITUATION:", "").strip()
            elif line.startswith("OPTION_A:"):
                opt_a = line.replace("OPTION_A:", "").strip()
            elif line.startswith("OPTION_B:"):
                opt_b = line.replace("OPTION_B:", "").strip()

        return situation, opt_a, opt_b
    except Exception as e:
        print(f"Erreur génération dilemme : {e}")
        return (
            "Vous êtes coincés dans un ascenseur pour 48h sans issue :",
            "Devoir raconter ton pire secret inavouable à tout ton entourage en direct",
            "Manger uniquement de la nourriture pour chat pendant les 6 prochains mois"
        )

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
