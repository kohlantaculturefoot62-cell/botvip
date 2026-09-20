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


def generate_morning_text(username: str, context: str, is_victim: bool) -> str:
    """Génère un compliment ou un clash personnalisé via Gemini 3.5."""
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
- Reste bon enfant, pas d'insultes graves ni de haine.
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
- Fais une référence directe ou subtile à ses discussions récentes ou ses centres d'intérêt.
- Si le contexte indique qu'il n'a pas parlé, dis-lui qu'on aimerait bien le voir parler un peu plus aujourd'hui.
- Ne sois pas un robot : parle comme un humain sur Discord.
- Réponds UNIQUEMENT le message, sans guillemets ni introduction.
"""

    try:
        response = ai_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Erreur API Gemini pour {username}: {e}")
        return "Passe une excellente journée !" if not is_victim else "C'est tombé sur toi ce matin... fais un effort aujourd'hui !"


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
