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

ai_client = genai.Client(api_key=GEMINI_API_KEY)

intents = discord.Intents.default()
intents.guilds = True
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
scheduler = AsyncIOScheduler()

async def get_recent_user_activity(guild: discord.Guild, user_id: int) -> str:
    """Scanne les salons textuels pour récupérer les derniers messages réels du membre."""
    collected_messages = []

    for channel in guild.text_channels:
        # Vérification des droits d'accès au salon
        perms = channel.permissions_for(guild.me)
        if not (perms.view_channel and perms.read_message_history):
            continue

        try:
            # On parcourt les 100 derniers messages du salon
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

def generate_morning_text(username: str, context: str, is_victim: bool) -> str:
    """Génère un texte unique basé sur les propos du membre."""
    if is_victim:
        prompt = f"""
Tu es un pote taquin et sarcastique sur un serveur Discord. Tu dois tacler avec humour '{username}'.
Voici ses derniers messages sur le serveur :
---
{context}
---

Consignes STRICTES :
- Fais un clash court (2 phrases max) qui fait DIRECTEMENT référence à ce qu'il/elle a dit ou à ses habitudes.
- Si le contexte indique qu'il n'a pas parlé, clashe-le sur le fait qu'il est un fantôme / un sous-marin qui ne sert à rien sur le serveur.
- Reste bon enfant, pas d'insultes graves.
- Réponds UNIQUEMENT le clash, sans guillemets ni introduction.
"""
    else:
        prompt = f"""
Tu es un pote bienveillant sur un serveur Discord. Tu dois donner de la force et souhaiter une bonne journée à '{username}'.
Voici ses derniers messages sur le serveur :
---
{context}
---

Consignes STRICTES :
- Fais un message motivant et sympa de 2 phrases max.
- Fais une référence subtile ou directe à ses discussions récentes ou ses centres d'intérêt.
- Si le contexte indique qu'il n'a pas parlé, dis-lui qu'on aimerait bien le voir parler un peu plus aujourd'hui.
- Ne sois pas un robot : parle comme un humain sur Discord.
- Réponds UNIQUEMENT le message, sans guillemets ni introduction.
"""

    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Erreur API Gemini pour {username}: {e}")
        return "Passe une excellente journée pleine d'énergie !" if not is_victim else "C'est tombé sur toi ce matin... fais un effort aujourd'hui !"

async def run_daily_routine():
    print("--- Démarrage de la distribution matinale ---")
    guild = bot.get_guild(GUILD_ID)
    if not guild:
        print(f"Erreur : Serveur ID {GUILD_ID} introuvable.")
        return

    # On charge l'ensemble des membres
    members = [m for m in guild.members if not m.bot]
    if not members:
        print("Aucun membre humain trouvé.")
        return

    victim = random.choice(members)
    print(f"Victime du jour : {victim.display_name} ({victim.name})")

    for member in members:
        is_victim = (member.id == victim.id)
        
        # 1. Contexte
        context = await get_recent_user_activity(guild, member.id)
        print(f"\n[SCAN] {member.name} : {len(context.splitlines())} ligne(s) de contexte trouvée(s).")

        # 2. Génération
        text = generate_morning_text(member.display_name, context, is_victim)
        print(f"[GEN] Pour {member.name} ({'CLASH' if is_victim else 'GENTIL'}) : {text}")

        prefix = "💥 **Le tacle du matin :**\n" if is_victim else "☀️ **Bonjour !**\n"
        full_message = f"{prefix}{text}"

        # 3. Envoi
        try:
            await member.send(full_message)
            print(f"[OK] Envoyé en MP à {member.name}")
        except discord.Forbidden:
            print(f"[BLOQUÉ] Impossible d'envoyer un DM à {member.name} (MP fermés)")
        except Exception as e:
            print(f"[ERREUR] Envoi échoué pour {member.name}: {e}")

        # Temporisation anti rate-limit
        await asyncio.sleep(2.5)

    print("\n--- Fin de la distribution matinale ---")

@bot.event
async def on_ready():
    print(f"Bot opérationnel : {bot.user.name} ({bot.user.id})")
    scheduler.add_job(
        run_daily_routine,
        trigger=CronTrigger(hour=8, minute=0, timezone="Europe/Paris")
    )
    scheduler.start()

@bot.command(name="test_matin")
@commands.has_permissions(administrator=True)
async def test_matin(ctx):
    await ctx.send("Lancement du test...")
    await run_daily_routine()

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
