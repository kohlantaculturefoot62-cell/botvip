import os
import asyncio
import random
from dotenv import load_dotenv
import discord
from discord.ext import commands
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from google import genai

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("GUILD_ID"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Client Gemini
ai_client = genai.Client(api_key=GEMINI_API_KEY)

# Client Discord avec les intents nécessaires
intents = discord.Intents.default()
intents.guilds = True
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
scheduler = AsyncIOScheduler()


async def get_recent_user_activity(guild: discord.Guild, user_id: int, limit_per_channel: int = 15) -> str:
    """Récupère les messages récents d'un membre pour donner du contexte à Gemini."""
    user_messages = []

    for channel in guild.text_channels:
        # Vérifie que le bot a les permissions de lecture
        permissions = channel.permissions_for(guild.me)
        if not (permissions.read_messages and permissions.read_message_history):
            continue

        try:
            async for msg in channel.history(limit=50):
                if msg.author.id == user_id and msg.content.strip():
                    user_messages.append(f"- {msg.content.strip()}")
                    if len(user_messages) >= limit_per_channel:
                        break
        except discord.DiscordException:
            continue

        if len(user_messages) >= limit_per_channel:
            break

    if not user_messages:
        return "Aucun message récent trouvé. La personne est plutôt discrète."

    return "\n".join(user_messages)


def generate_morning_text(username: str, context: str, is_victim: bool) -> str:
    """Génère le compliment ou le clash personnalisé via Gemini."""
    if is_victim:
        prompt = f"""
Tu es un bot Discord cynique mais drôle. Tu dois envoyer le clash du matin à l'utilisateur '{username}'.
Voici quelques-uns de ses messages récents sur le serveur pour t'inspirer de son attitude, ses délires ou ses sujets de discussion :
{context}

Consignes :
- Fais un clash / tacle humoristique court (max 2-3 phrases).
- Fais une référence subtile ou directe à ses messages ou sa manière de parler.
- Reste bon enfant, pas d'insultes graves ni de haine, juste du bon chambrage amical.
- Ne mets pas de guillemets autour de ta réponse.
"""
    else:
        prompt = f"""
Tu es un bot Discord bienveillant et chaleureux. Tu dois envoyer un mot doux / d'encouragement matinal personnalisé à '{username}'.
Voici quelques-uns de ses messages récents sur le serveur :
{context}

Consignes :
- Rédige un compliment ou un encouragement motivant et sympa (max 2-3 phrases).
- Personnalise-le en fonction de ce qu'il/elle raconte ou dégage dans ses messages.
- Sois naturel, pas trop formel, avec une pointe d'humour léger.
- Ne mets pas de guillemets autour de ta réponse.
"""

    try:
        response = ai_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Erreur Gemini pour {username}: {e}")
        return "Passe une excellente journée pleine d'énergie !" if not is_victim else "C'est tombé sur toi ce matin... fais un effort aujourd'hui !"


async def run_daily_routine():
    print("Démarrage de la tournée matinale...")
    guild = bot.get_guild(GUILD_ID)
    if not guild:
        print(f"Serveur avec l'ID {GUILD_ID} introuvable.")
        return

    # Récupération de tous les membres réels
    human_members = [m for m in guild.members if not m.bot]
    if not human_members:
        print("Aucun membre trouvé.")
        return

    victim = random.choice(human_members)
    print(f"Victime désignée pour le clash : {victim.display_name}")

    for member in human_members:
        is_victim = (member.id == victim.id)
        
        # Récupération du contexte des messages
        activity_context = await get_recent_user_activity(guild, member.id)

        # Génération du message avec Gemini
        text = generate_morning_text(member.display_name, activity_context, is_victim)
        
        prefix = "💥 **Le tacle du matin :**\n" if is_victim else "☀️ **Bonjour !**\n"
        full_message = f"{prefix}{text}"

        try:
            await member.send(full_message)
            print(f"Message envoyé à {member.name} ({'CLASH' if is_victim else 'GENTIL'})")
        except discord.Forbidden:
            print(f"DMs fermés pour {member.name}")
        except Exception as e:
            print(f"Erreur lors de l'envoi à {member.name}: {e}")

        # Pause pour éviter les limites de taux Discord (rate limits)
        await asyncio.sleep(2.5)

    print("Tournée matinale achevée avec succès !")


@bot.event
async def on_ready():
    print(f"Bot connecté sous {bot.user.name} (ID: {bot.user.id})")

    # Planification chaque matin à 8h00 (heure de Paris)
    scheduler.add_job(
        run_daily_routine,
        trigger=CronTrigger(hour=8, minute=0, timezone="Europe/Paris")
    )
    scheduler.start()
    print("Planificateur APScheduler démarré (08:00 Europe/Paris).")


# Commande de test pour déclencher manuellement sans attendre 8h
@bot.command(name="test_matin")
@commands.has_permissions(administrator=True)
async def test_matin(ctx):
    await ctx.send("Lancement manuel de la routine matinale...")
    await run_daily_routine()


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
