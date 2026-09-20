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


async def get_recent_user_activity(guild: discord.Guild, user_id: int) -> str:
    """Scanne les salons textuels pour récupérer les derniers messages réels du membre."""
    collected_messages = []

    for channel in guild.text_channels:
        perms = channel.permissions_for(guild.me)
        if not (perms.view_channel and perms.read_message_history):
            continue

        try:
            async for msg in channel.history(limit=100):
                if msg.author.id == user_id and msg.content.strip():
                    collected_messages.append(msg.content.strip())
                    if len(collected_messages) >= 10:
                        break
        except Exception:
            continue

        if len(collected_messages) >= 10:
            break

    if not collected_messages:
        return "Aucun message récent trouvé dans les salons accessibles."

    return "\n".join([f"- {m}" for m in collected_messages])


import random

def generate_morning_text(username: str, context: str, is_victim: bool) -> str:
    """Génère un compliment ou un clash développé avec un style aléatoire pour surprendre."""
    
    # Styles variés pour les compliments
    sweet_styles = [
        "coach de vie surmotivé façon TED Talk, mais avec un second degré bienveillant",
        "pote sincère qui pose les termes et prend le temps d'envoyer de la vraie bonne énergie",
        "observateur décalé qui fait une analyse psychologique élogieuse basée sur ses messages",
        "poète moderne du quotidien, mi-philosophique mi-drôle",
        "discours d'avant-match de vestiaire pour chauffer la personne avant sa journée"
    ]

    # Styles variés pour les clashs
    roast_styles = [
        "chroniqueur satirique impitoyable mais hilarant qui décortique son comportement",
        "pote dépité qui remet gentiment les pendules à l'heure avec beaucoup d'ironie",
        "enquêteur qui a analysé ses messages et en tire des conclusions désastreuses",
        "critique gastronomique/littéraire qui note très sévèrement la prestation de la personne sur le serveur",
        "discours faussement solennel pour lui décerner la palme de la flemme ou du flop"
    ]

    chosen_style = random.choice(roast_styles if is_victim else sweet_styles)

    if is_victim:
        prompt = f"""
Tu es un pote sur un serveur Discord. Tu dois rédiger le tacle du matin destiné à '{username}'.
Voici ses derniers messages sur le serveur pour t'inspirer de son attitude, ses expressions ou ses délires :
---
{context}
---

Ton angle d'attaque du jour : adoptera un ton de **{chosen_style}**.

Consignes :
- Développe un message consistant et bien écrit (3 à 5 phrases, environ 50 à 90 mots). Ne sois pas trop bref, prends le temps de bien poser la vanne.
- Fais des références précises à ses messages récents ou à sa manière de s'exprimer sur le serveur.
- Si le contexte indique qu'il/elle n'a pas parlé, clashe-le/la longuement sur son statut de fantôme ou de spectateur passif de la vie du serveur.
- Reste dans le chambrage entre potes : drôle, créatif, piquant mais sans haine ni vulgarité gratuite.
- Réponds UNIQUEMENT le texte du message, sans guillemets, sans titre.
"""
    else:
        prompt = f"""
Tu es un ami proche et chaleureux sur un serveur Discord. Tu dois rédiger un mot doux / message d'encouragement matinal personnalisé pour '{username}'.
Voici ses derniers messages sur le serveur :
---
{context}
---

Ton style du jour : adopte un ton de **{chosen_style}**.

Consignes :
- Écris un texte riche, vivant et sympa (3 à 5 phrases, environ 50 à 90 mots). Ne fais pas un message expéditif de deux lignes.
- Inspire-toi réellement de ce qu'il/elle raconte, de ses passions ou de son humeur pour que la personne sente que c'est du 100% sur-mesure.
- Si le contexte indique qu'il/elle n'a pas beaucoup parlé récemment, dis-lui avec humour et bienveillance qu'il/elle manque aux discussions du serveur.
- Sois naturel, évite le ton robotique ou corpo : parle comme quelqu'un qui écrit un super message à un pote sur Discord.
- Réponds UNIQUEMENT le texte du message, sans guillemets, sans titre.
"""

    try:
        response = ai_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Erreur API Gemini pour {username}: {e}")
        return (
            "Passe une excellente journée ! Prends le temps de faire les choses bien aujourd'hui et n'oublie pas de t'hydrater, force pour tout ce qui t'attend !"
            if not is_victim
            else "C'est tombé sur toi ce matin... Honnêtement, regarde tes derniers messages, c'était écrit d'avance. Fais un effort aujourd'hui et essaie de relever le niveau !"
        )

async def run_daily_routine(dry_run: bool = False):
    mode = "[MODE SIMULATION / AUCUN ENVOI]" if dry_run else "[MODE RÉEL / ENVOI DM]"
    print(f"\n==================== DÉBUT ROUTINE {mode} ====================")

    guild = bot.get_guild(GUILD_ID)
    if not guild:
        print(f"Erreur : Impossible de trouver le serveur ID {GUILD_ID}.")
        return

    members = [m for m in guild.members if not m.bot]
    if not members:
        print("Aucun membre humain détecté sur le serveur.")
        return

    victim = random.choice(members)
    print(f"🎯 Victime sélectionnée : {victim.display_name} (@{victim.name})\n")

    for member in members:
        is_victim = (member.id == victim.id)
        role_label = "CLASH" if is_victim else "GENTIL"

        # Contexte
        context = await get_recent_user_activity(guild, member.id)
        nb_messages = len(context.splitlines()) if "Aucun message" not in context else 0
        print(f"👤 Membre : {member.display_name} (@{member.name}) | Rôle : {role_label}")
        print(f"   ↳ Messages analysés : {nb_messages}")

        # Génération
        generated_text = generate_morning_text(member.display_name, context, is_victim)
        prefix = "💥 **Le tacle du matin :**\n" if is_victim else "☀️ **Bonjour !**\n"
        full_message = f"{prefix}{generated_text}"

        print(f"   ↳ Message généré :\n\"\"\"\n{full_message}\n\"\"\"")

        # Envoi conditionné au mode
        if not dry_run:
            try:
                await member.send(full_message)
                print(f"   ↳ Statut : [ENVOYÉ EN MP]")
            except discord.Forbidden:
                print(f"   ↳ Statut : [ÉCHEC - MP FERMÉS]")
            except Exception as e:
                print(f"   ↳ Statut : [ERREUR] {e}")
            await asyncio.sleep(2.5)
        else:
            print("   ↳ Statut : [SIMULATION - Non envoyé]")

        print("-" * 50)

    print(f"==================== FIN ROUTINE {mode} ====================\n")


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
    print("Planificateur APScheduler actif (08:00 Europe/Paris).")


@bot.tree.command(name="test_matin", description="Simule la tournée dans les logs sans envoyer de DM")
@app_commands.default_permissions(administrator=True)
async def test_matin(interaction: discord.Interaction):
    await interaction.response.send_message(
        "Simulation lancée ! Regarde les **Logs** Square Cloud.",
        ephemeral=True
    )
    await run_daily_routine(dry_run=True)


@bot.tree.command(name="envoyer_maintenant", description="Force la tournée réelle avec envoi des DM")
@app_commands.default_permissions(administrator=True)
async def envoyer_maintenant(interaction: discord.Interaction):
    await interaction.response.send_message(
        "Tournée réelle démarrée.",
        ephemeral=True
    )
    await run_daily_routine(dry_run=False)


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
