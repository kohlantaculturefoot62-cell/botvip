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
from google.genai import types

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("GUILD_ID"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = "gemini-3.6-flash-lite"

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



# ==============================================================================
# 2. LOGIQUE MATINALE (ENVOI INDIVIDUEL MP)
# ==============================================================================



# ==============================================================================
# 3. RÉSUMÉ DES MESSAGES VOCAUX (GEMINI)
# ==============================================================================

VOICE_SUMMARY_PROMPT = """
Tu reçois un message vocal envoyé sur un serveur Discord entre amis.

Écoute-le en entier puis réponds STRICTEMENT dans ce format (sans autre texte) :

RÉSUMÉ: [1 à 3 phrases qui donnent l'essentiel : ce que la personne veut dire, demande ou annonce]
POINTS CLÉS: [liste à puces "- ..." de 2 à 5 éléments, UNIQUEMENT si le vocal est long ou contient plusieurs infos distinctes, sinon écris "-"]

Consignes :
- Écris en français, au style indirect ("Il/Elle dit que...", "Il/Elle demande si..."), sans inventer le genre si tu ne peux pas le déduire : utilise alors une tournure neutre.
- Reste fidèle au contenu : n'ajoute aucune information absente du vocal.
- Si l'audio est inaudible, vide ou sans parole, écris à la place de RÉSUMÉ : "Aucune parole exploitable dans ce vocal." et "-" pour les points clés.
"""

# Cache {message_id: texte du résumé} pour ne pas payer deux fois le même vocal.
voice_summary_cache: dict[int, str] = {}

# Taille max envoyée à Gemini en inline (la limite de la requête est de 20 Mo).
MAX_AUDIO_BYTES = 15 * 1024 * 1024


def get_voice_attachment(message: discord.Message) -> discord.Attachment | None:
    """Renvoie la pièce jointe audio/vocale du message, ou None s'il n'y en a pas."""
    for att in message.attachments:
        content_type = (att.content_type or "").lower()
        if content_type.startswith("audio/"):
            return att
    return None


async def summarize_voice_attachment(attachment: discord.Attachment) -> str:
    """Télécharge l'audio, l'envoie à Gemini et renvoie le résumé formaté."""
    if attachment.size > MAX_AUDIO_BYTES:
        raise ValueError("Ce vocal est trop volumineux pour être analysé.")

    audio_bytes = await attachment.read()
    mime_type = (attachment.content_type or "audio/ogg").split(";")[0].strip()

    # Client asynchrone : n'empêche pas le bot de répondre pendant l'appel Gemini.
    response = await ai_client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            VOICE_SUMMARY_PROMPT,
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        ],
    )

    text = (response.text or "").strip()
    if not text:
        raise ValueError("Gemini n'a renvoyé aucun résumé.")
    return text


def build_voice_summary_embed(summary: str, source: discord.Message) -> discord.Embed:
    """Construit l'embed Discord contenant le résumé du vocal."""
    summary_part = summary
    points_part = ""

    # Découpe le format "RÉSUMÉ: ... / POINTS CLÉS: ..." renvoyé par Gemini.
    if "POINTS CLÉS:" in summary:
        summary_part, points_part = summary.split("POINTS CLÉS:", 1)
    summary_part = summary_part.replace("RÉSUMÉ:", "").strip()
    points_part = points_part.strip()

    embed = discord.Embed(
        title="📝 Résumé du vocal",
        description=summary_part[:4000],
        color=discord.Color.blurple(),
    )
    if points_part and points_part != "-":
        embed.add_field(name="Points clés", value=points_part[:1024], inline=False)

    duration = None
    for att in source.attachments:
        att_duration = getattr(att, "duration", None)  # absent sur les anciennes versions de discord.py
        if att_duration is not None:
            duration = att_duration
            break
    footer = f"Vocal de {source.author.display_name}"
    if duration:
        footer += f" • {int(duration)} s"
    embed.set_footer(text=footer)
    return embed


async def summarize_voice_message(message: discord.Message) -> discord.Embed | None:
    """Point d'entrée commun (bouton + menu contextuel) : cache -> Gemini -> embed."""
    attachment = get_voice_attachment(message)
    if attachment is None:
        return None

    summary = voice_summary_cache.get(message.id)
    if summary is None:
        summary = await summarize_voice_attachment(attachment)
        voice_summary_cache[message.id] = summary
        # Évite que le cache grossisse indéfiniment.
        if len(voice_summary_cache) > 200:
            voice_summary_cache.pop(next(iter(voice_summary_cache)))

    return build_voice_summary_embed(summary, message)


class VoiceSummaryView(discord.ui.View):
    """Bouton persistant affiché sous chaque vocal détecté."""

    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="Résumer ce vocal", emoji="📝", style=discord.ButtonStyle.secondary, custom_id="voice_summary_btn")
    async def btn_summarize(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Le bouton est sur la réponse du bot : le vocal d'origine est le message référencé.
        ref = interaction.message.reference
        if ref is None or ref.message_id is None:
            await interaction.response.send_message("Impossible de retrouver le vocal d'origine.", ephemeral=True)
            return

        await interaction.response.defer(thinking=True)
        try:
            original = await interaction.channel.fetch_message(ref.message_id)
            embed = await summarize_voice_message(original)
        except discord.NotFound:
            await interaction.followup.send("Le vocal d'origine a été supprimé.", ephemeral=True)
            return
        except Exception as e:
            print(f"Erreur résumé vocal (bouton) : {e}")
            await interaction.followup.send("Je n'ai pas réussi à résumer ce vocal, réessaie dans un instant.", ephemeral=True)
            return

        if embed is None:
            await interaction.followup.send("Aucun vocal trouvé sur ce message.", ephemeral=True)
            return

        await interaction.followup.send(embed=embed)


# ==============================================================================
# 4. INITIALISATION & COMMANDES SLASH
# ==============================================================================

@bot.event
async def on_ready():
    print(f"Bot connecté en tant que : {bot.user.name} ({bot.user.id})")

    bot.add_view(VoiceSummaryView())

    guild_obj = discord.Object(id=GUILD_ID)
    bot.tree.copy_global_to(guild=guild_obj)
    synced = await bot.tree.sync(guild=guild_obj)
    print(f"{len(synced)} commande(s) slash synchronisée(s).")


@bot.listen("on_message")
async def on_voice_message(message: discord.Message):
    """Détecte les vocaux et propose le bouton « Résumer »."""
    if message.author.bot or message.guild is None or message.guild.id != GUILD_ID:
        return

    # Seuls les vrais messages vocaux Discord déclenchent le bouton (pas les fichiers audio classiques).
    if not message.flags.voice or get_voice_attachment(message) is None:
        return

    await message.reply("🎙️ Pas le temps d'écouter ?", view=VoiceSummaryView(), mention_author=False)


@bot.tree.context_menu(name="Résumer le vocal")
async def context_resume_vocal(interaction: discord.Interaction, message: discord.Message):
    """Clic droit sur un vocal > Applications > Résumer le vocal."""
    if get_voice_attachment(message) is None:
        await interaction.response.send_message("Ce message ne contient pas de vocal.", ephemeral=True)
        return

    await interaction.response.defer(thinking=True)
    try:
        embed = await summarize_voice_message(message)
    except Exception as e:
        print(f"Erreur résumé vocal (menu contextuel) : {e}")
        await interaction.followup.send("Je n'ai pas réussi à résumer ce vocal, réessaie dans un instant.", ephemeral=True)
        return

    await interaction.followup.send(embed=embed)


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
