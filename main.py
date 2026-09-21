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


# ==============================================================================
# 1. LOGIQUE MATINALE (COMPLIMENTS & CLASHS)
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
1. Nom d'appel : Appelle la personne UNIQUEMENT par son nom d'affichage '{display_name}'.
2. Genre et accords : Déduis si '{display_name}' est un homme ou une femme d'après les messages. Accorde TOUS tes adjectifs et participes passés en conséquence.
"""

    if is_victim:
        prompt = f"""
Tu es un pote sur un serveur Discord. Tu dois rédiger le tacle du matin destiné à '{display_name}'.
{common_instructions}
Ton angle d'attaque du jour : adopte un ton de **{chosen_style}**.

Consignes :
- Développe un message consistant (3 à 5 phrases, 50 à 90 mots).
- Fais des références précises à ses messages récents ou à ses habitudes.
- Si le contexte indique qu'il/elle n'a pas parlé, clashe-le/la sur son statut de fantôme passif.
- Reste dans le chambrage entre potes : drôle, piquant, sans haine.
- Réponds UNIQUEMENT le texte du message, sans guillemets, sans titre.
"""
    else:
        prompt = f"""
Tu es un ami proche sur un serveur Discord. Tu dois rédiger un mot doux personnalisé pour '{display_name}'.
{common_instructions}
Ton style du jour : adopte un ton de **{chosen_style}**.

Consignes :
- Écris un texte riche et sympa (3 à 5 phrases, 50 à 90 mots).
- Inspire-toi réellement de ce qu'il/elle raconte.
- Si le contexte indique qu'il/elle n'a pas beaucoup parlé, glisse gentiment qu'il/elle manque aux débats.
- Sois naturel, évite le ton robotique.
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
    mode = "[MODE SIMULATION]" if dry_run else "[MODE RÉEL]"
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
# 2. SYSTÈME DE DILEMME PERSISTANT AVEC AFFICHAGE DES VOTANTS
# ==============================================================================

def generate_hardcore_dilemma() -> tuple[str, str, str]:
    """Génère un dilemme cornélien et extrême."""
    prompt = """
Génère un dilemme "Tu préfères" extrêmement difficile, absurde ou cornélien pour animer un débat animé entre potes.
Les deux options doivent être quasi impossibles à départager, très inconfortables, honteuses ou intenses.

Règles strictes :
- Pas de contenu impliquant des mineurs.
- Uniquement des situations fictives, des sacrifices moraux, des choix absurdes ou de la honte sociale entre adultes.
- Format de réponse STRICTEMENT attendu (3 lignes) :
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
        situation, opt_a, opt_b = "Choix cornélien :", "Option A", "Option B"
        for line in lines:
            if line.startswith("SITUATION:"):
                situation = line.replace("SITUATION:", "").strip()
            elif line.startswith("OPTION_A:"):
                opt_a = line.replace("OPTION_A:", "").strip()
            elif line.startswith("OPTION_B:"):
                opt_b = line.replace("OPTION_B:", "").strip()
        return situation, opt_a, opt_b
    except Exception as e:
        print(f"Erreur dilemme : {e}")
        return (
            "Vous devez survivre sur une île déserte pendant 1 an :",
            "Devoir marcher pieds nus sur des Lego 30 minutes chaque matin",
            "N'avoir le droit de boire que de l'eau tiède avec du sel"
        )


class PersistentDilemmaView(discord.ui.View):
    def __init__(self):
        # timeout=None rend la vue permanente (ne meurt pas au redémarrage)
        super().__init__(timeout=None)

    def extract_data_and_vote(self, message: discord.Message, user: discord.Member, vote_for: str) -> discord.Embed:
        embed = message.embeds[0]
        situation = embed.description

        # Extraction des deux options et des votants depuis les fields de l'Embed
        opt_a_name = embed.fields[0].name
        opt_b_name = embed.fields[1].name

        # Parse les votants actuels (IDs de mention <@123...>)
        raw_voters_a = embed.fields[0].value.replace("Aucun vote pour l'instant.", "").split(", ")
        raw_voters_b = embed.fields[1].value.replace("Aucun vote pour l'instant.", "").split(", ")

        voters_a = set(v.strip() for v in raw_voters_a if v.strip())
        voters_b = set(v.strip() for v in raw_voters_b if v.strip())

        user_mention = user.mention

        if vote_for == "A":
            voters_b.discard(user_mention)
            voters_a.add(user_mention)
        else:
            voters_a.discard(user_mention)
            voters_b.add(user_mention)

        # Reconstruction de l'Embed
        new_embed = discord.Embed(
            title="⚡ DILEMME CORNÉLIEN ⚡",
            description=situation,
            color=discord.Color.dark_red()
        )
        val_a = ", ".join(voters_a) if voters_a else "Aucun vote pour l'instant."
        val_b = ", ".join(voters_b) if voters_b else "Aucun vote pour l'instant."

        new_embed.add_field(name=f"{opt_a_name.split(' (')[0]} ({len(voters_a)})", value=val_a, inline=False)
        new_embed.add_field(name=f"{opt_b_name.split(' (')[0]} ({len(voters_b)})", value=val_b, inline=False)
        new_embed.set_footer(text="Cliquez sur les boutons ci-dessous pour voter ou changer d'avis !")
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
# 3. COMMANDES ET INITIALISATION DU BOT
# ==============================================================================

@bot.event
async def on_ready():
    print(f"Bot connecté en tant que : {bot.user.name} ({bot.user.id})")

    # Enregistrement de la vue persistante pour qu'elle fonctionne même après redémarrage
    bot.add_view(PersistentDilemmaView())

    guild_obj = discord.Object(id=GUILD_ID)
    bot.tree.copy_global_to(guild=guild_obj)
    synced = await bot.tree.sync(guild=guild_obj)
    print(f"{len(synced)} commande(s) slash synchronisée(s).")

    scheduler.add_job(
        run_daily_routine,
        trigger=CronTrigger(hour=8, minute=0, timezone="Europe/Paris"),
        kwargs={"dry_run": False}
    )
    scheduler.start()
    print("Planificateur matinal actif (08:00 Europe/Paris).")


@bot.tree.command(name="dilemme", description="Lance un dilemme cornélien dans le salon avec votes en direct")
async def dilemme(interaction: discord.Interaction):
    await interaction.response.defer()

    situation, opt_a, opt_b = generate_hardcore_dilemma()

    embed = discord.Embed(
        title="⚡ DILEMME CORNÉLIEN ⚡",
        description=f"**{situation}**",
        color=discord.Color.dark_red()
    )
    embed.add_field(name=f"🔴 Option A : {opt_a} (0)", value="Aucun vote pour l'instant.", inline=False)
    embed.add_field(name=f"🔵 Option B : {opt_b} (0)", value="Aucun vote pour l'instant.", inline=False)
    embed.set_footer(text="Cliquez sur les boutons ci-dessous pour voter ou changer d'avis !")

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


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
